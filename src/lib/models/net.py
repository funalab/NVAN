import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import matplotlib.pylab as plt

from src.lib.models.function import mask_softmax, mask_mean, mask_max, seq_mask


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, num_classes, num_layers, hidden_dim, dropout, lossfun):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # self.word_embed = nn.Embedding(input_dim, embed_dim)
        # self.lstm = nn.LSTM(embed_dim, hidden_dim, dropout=dropout)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim*2, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.loss = lossfun

    def forward(self, input):
        # _, lstm_out = self.lstm(input.view(np.shape(input)[1], np.shape(input)[0], -1))
        lstm_out, _ = self.lstm(input.view(np.shape(input)[1], np.shape(input)[0], -1))
        # tag_space = self.hidden2tag(lstm_out[0].view(-1, self.hidden_dim))
        tag_space = self.hidden2tag(lstm_out[-1, :, :])
        tag_scores = self.softmax(tag_space)
        return tag_scores


class LSTMAttentionClassifier(nn.Module):
    def __init__(
            self,
            input_dim,
            embed_dim,
            num_classes,
            num_layers,
            hidden_dim,
            dropout,
            lossfun,
            phase='train'
            ):
        super(LSTMAttentionClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # self.word_embed = nn.Embedding(input_dim, embed_dim)
        # self.lstm = nn.LSTM(embed_dim, hidden_dim, dropout=dropout)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=False)
        self.hidden2tag = nn.Linear(hidden_dim, num_classes)
        # self.hidden2tag = nn.Linear(hidden_dim*2, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.loss = lossfun
        self.phase = phase

    def attention(self, lstm_out, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state, soft_attn_weights

    def forward(self, input):
        lstm_out, (final_hidden_state, _) = self.lstm(input.permute(1, 0, 2))
        # lstm_out, (final_hidden_state, _) = self.lstm(input.view(np.shape(input)[1], np.shape(input)[0], -1))
        lstm_out = lstm_out.permute(1, 0, 2)
        attn_out, attn_weights = self.attention(lstm_out, final_hidden_state)
        tag_space = self.hidden2tag(attn_out)
        tag_scores = self.softmax(tag_space)

        if self.phase == 'test':
            return tag_scores, attn_weights
        else:
            return tag_scores


class MuVAN(nn.Module):
    """
    Multi-view Attention Network for Multivariate Temporal Data

    Parameters
    ----------
    input_dim : dimention of multivariable
    embed_dim : hidden size
    num_classes : 
    num_layers : number of hidden layers. Default: 1
    hidden_dim : 
    dropout : dropout rate. Default: 0.5
    lossfun : 

    Inputs
    ------
    input: tensor, shaped [batch, max_step, input_size]

    Outputs
    -------
    output: tensor, shaped [batch, max_step, num_directions * hidden_size],
         tensor containing the output features (h_t) from the last layer
         of the LSTM, for each t.
    """

    def __init__(
            self,
            input_dim,
            embed_dim,
            num_classes,
            num_layers,
            hidden_dim,
            dropout,
            lossfun,
            phase='train'
            ):
        super(MuVAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.bgru = nn.LSTM(1, hidden_dim, num_layers, dropout=dropout, bidirectional=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.attn_fusion_1 = nn.Conv2d(2, 16, 5, 1, 2)
        self.attn_fusion_2 = nn.Conv2d(16, 32, 5, 1, 2)
        # self.affine = nn.Linear(hidden_dim * input_dim * 32 * 2, num_classes)
        self.affine = nn.Linear(int(hidden_dim * 2 / 4) * int(input_dim / 4) * 32, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.loss = lossfun
        self.phase = phase

    # def hybrid_focus_procedure(self, hidden_state):


    def attention(self, lstm_out, final_state):
        # final_state = final_state.transpose(0, 1).transpose(1, 2)
        final_state = final_state.permute(1, 2, 0)
        hidden = final_state.reshape(lstm_out.shape[0], -1)
        attn_weights = torch.bmm(lstm_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state, soft_attn_weights

    def forward(self, input):
        hidden_matrix, attn_matrix, attn_weights_matrix = [], [], []
        for v in range(self.input_dim):
            lstm_out, (final_hidden_state, _) = self.bgru(input[:,:,v].unsqueeze(2).permute(1, 0, 2))
            lstm_out = lstm_out.permute(1, 0, 2)
            attn_out, attn_weights = self.attention(lstm_out, final_hidden_state)

            hidden_matrix.append(lstm_out.unsqueeze(0))
            attn_matrix.append(attn_out.unsqueeze(0))
            attn_weights_matrix.append(attn_weights.unsqueeze(0))

        # hidden_matrix: [batch, view, time, dim]
        hidden_matrix = torch.cat(hidden_matrix).permute(1, 0, 2, 3)
        # attn_matrix: [batch, view, dim]
        attn_matrix = torch.cat(attn_matrix).permute(1, 0, 2)
        # attn_weights_matrix: [batch, view, time]
        attn_weights_matrix = torch.cat(attn_weights_matrix).permute(1, 0, 2)
        # print(hidden_matrix.shape)
        # print(attn_matrix.shape)
        # print(attn_weights_matrix.shape)

        cat_matrix = torch.cat([hidden_matrix[:,:,-1,:], attn_matrix]).view(hidden_matrix.shape[0], 2, self.input_dim, self.hidden_dim * 2)
        logit = self.pool(self.relu(self.attn_fusion_1(cat_matrix)))
        logit = self.pool(self.relu(self.attn_fusion_2(logit)))
        logit = logit.view(logit.size()[0], -1)
        logit = self.affine(logit)

        if self.phase == 'test':
            return logit, attn_weights_matrix
        else:
            return logit





class DynamicLSTM(nn.Module):
    """
    Dynamic LSTM module, which can handle variable length input sequence.

    Parameters
    ----------
    input_size : input size
    hidden_size : hidden size
    num_layers : number of hidden layers. Default: 1
    dropout : dropout rate. Default: 0.5
    bidirectional : If True, becomes a bidirectional RNN. Default: False.

    Inputs
    ------
    input: tensor, shaped [batch, max_step, input_size]
    seq_lens: tensor, shaped [batch], sequence lengths of batch

    Outputs
    -------
    output: tensor, shaped [batch, max_step, num_directions * hidden_size],
         tensor containing the output features (h_t) from the last layer
         of the LSTM, for each t.
    """

    def __init__(self, input_size, hidden_size=100,
                 num_layers=1, dropout=0., bidirectional=False):
        super(DynamicLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bias=True,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, seq_lens):
        # sort input by descending length
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = torch.index_select(x, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(seq_lens, dim=0, index=idx_sort)

        # pack input
        x_packed = pack_padded_sequence(
            x_sort, seq_lens_sort, batch_first=True)

        # pass through rnn
        y_packed, _ = self.lstm(x_packed)

        # unpack output
        y_sort, length = pad_packed_sequence(y_packed, batch_first=True)

        # unsort output to original order
        y = torch.index_select(y_sort, dim=0, index=idx_unsort)

        return y



class BiDirectionalLSTMClassifier(nn.Module):
    """Model for quora insincere question classification.
    """
    def __init__(self, input_dim, padding_idx, embed_dim, num_classes, num_layers, hidden_dim, dropout, lossfun):
        super(BiDirectionalLSTMClassifier, self).__init__()
        pretrained_embed = None

        if pretrained_embed is None:
            self.embed = nn.Embedding(input_dim, embed_dim)
        else:
            self.embed = nn.Embedding.from_pretrained(
                pretrained_embed, freeze=False)
        self.embed.padding_idx = padding_idx

        self.rnn = DynamicLSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout, bidirectional=True)

        self.fc_att = nn.Linear(hidden_dim * 2, 1)

        self.fc = nn.Linear(hidden_dim * 6, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_classes)

        #self.loss = nn.BCEWithLogitsLoss()
        self.loss = lossfun


    def forward(self, word_seq, seq_len):
        # mask
        max_seq_len = torch.max(seq_len)
        mask = seq_mask(seq_len, max_seq_len)  # [b,msl]

        # embed
        e = self.drop(self.embed(word_seq))  # [b,msl]->[b,msl,e]

        # bi-rnn
        r = self.rnn(e, seq_len)  # [b,msl,e]->[b,msl,h*2]

        # attention
        att = self.fc_att(r).squeeze(-1)  # [b,msl,h*2]->[b,msl]
        att = mask_softmax(att, mask)  # [b,msl]
        r_att = torch.sum(att.unsqueeze(-1) * r, dim=1)  # [b,h*2]

        # pooling
        r_avg = mask_mean(r, mask)  # [b,h*2]
        r_max = mask_max(r, mask)  # [b,h*2]
        r = torch.cat([r_avg, r_max, r_att], dim=-1)  # [b,h*6]

        # feed-forward
        f = self.drop(self.act(self.fc(r)))  # [b,h*6]->[b,h]
        logits = self.out(f).squeeze(-1)  # [b,h]->[b]

        return logits
