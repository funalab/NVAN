import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import matplotlib.pylab as plt

from src.lib.models.function import mask_softmax, mask_mean, mask_max, seq_mask


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_classes,
        num_layers,
        hidden_dim,
        dropout,
        lossfun,
        sharping_factor=None,
        phase='train'
        ):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim*2, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.loss = lossfun

    def forward(self, input):
        lstm_out, _ = self.lstm(input.view(input.shape[1], input.shape[0], -1))
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
            sharping_factor=None,
            phase='train'
            ):
        super(LSTMAttentionClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=False)
        self.hidden2tag = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.loss = lossfun
        self.phase = phase

    def attention(self, lstm_out, final_state):
        hidden = final_state[-1]
        attn_weights = torch.bmm(lstm_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state, soft_attn_weights

    def forward(self, input):
        lstm_out, (final_hidden_state, _) = self.lstm(input.permute(1, 0, 2))
        lstm_out = lstm_out.permute(1, 0, 2)
        attn_out, attn_weights = self.attention(lstm_out, final_hidden_state)
        tag_space = self.hidden2tag(attn_out)
        tag_scores = self.softmax(tag_space)

        if self.phase == 'test':
            return tag_scores, attn_weights
        else:
            return tag_scores


class LSTMMultiAttentionClassifier(nn.Module):
    """
    Multi-view Attention Network for Multivariate Temporal Data

    Parameters
    ----------
    input_dim : dimention of multivariable
    embed_dim : hidden size
    num_classes : number of the output classes
    num_layers : number of hidden layers
    hidden_dim : number of unit on the hidden layer
    dropout : dropout rate
    lossfun : loss function

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
            sharping_factor=None,
            phase='train'
            ):
        super(LSTMMultiAttentionClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.bgru = nn.LSTM(1, hidden_dim, num_layers, dropout=dropout, bidirectional=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.attn_fusion_1 = nn.Conv2d(2, 16, 5, 1, 2)
        self.attn_fusion_2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.affine = nn.Linear(int(hidden_dim * 2 / 4) * int(input_dim / 4) * 32, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.loss = lossfun
        self.phase = phase


    def attention(self, lstm_out):
        batch, time, dim = lstm_out.shape
        # attn_weights: [batch, time]
        attn_weights = torch.bmm(lstm_out, lstm_out.permute(0, 2, 1)[:,:,-1].unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        # new_hidden_state: [batch, dim]
        new_hidden_state = torch.bmm(lstm_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state, soft_attn_weights

    def forward(self, input):
        hidden_matrix, attn_matrix, attn_weights_matrix = [], [], []
        for v in range(self.input_dim):
            # lstm_out: [batch, dim, time]
            lstm_out, _ = self.bgru(input[:,:,v].unsqueeze(2).permute(1, 0, 2))
            # lstm_out: [batch, time, dim]
            lstm_out = lstm_out.permute(1, 0, 2)
            # attn_out: [batch, dim], attn_weights: [batch, time]
            attn_out, attn_weights = self.attention(lstm_out)

            hidden_matrix.append(lstm_out.unsqueeze(0))
            attn_matrix.append(attn_out.unsqueeze(0))
            attn_weights_matrix.append(attn_weights.unsqueeze(0))

        # hidden_matrix: [batch, view, time, dim]
        hidden_matrix = torch.cat(hidden_matrix).permute(1, 0, 2, 3)
        # attn_matrix: [batch, view, dim]
        attn_matrix = torch.cat(attn_matrix).permute(1, 0, 2)
        # attn_weights_matrix: [batch, view, time]
        attn_weights_matrix = torch.cat(attn_weights_matrix).permute(1, 0, 2)

        cat_matrix = torch.cat([hidden_matrix[:,:,-1,:], attn_matrix]).view(hidden_matrix.shape[0], 2, self.input_dim, self.hidden_dim * 2)
        logit = self.pool(self.relu(self.attn_fusion_1(cat_matrix)))
        logit = self.pool(self.relu(self.attn_fusion_2(logit)))
        logit = logit.view(logit.size()[0], -1)
        logit = self.affine(logit)

        if self.phase == 'test':
            return logit, attn_weights_matrix
        else:
            return logit


class MuVAN(nn.Module):
    """
    Multi-view Attention Network for Multivariate Temporal Data

    Parameters
    ----------
    input_dim : dimention of multivariable
    embed_dim : hidden size
    num_classes : number of the output classes
    num_layers : number of hidden layers
    hidden_dim : number of unit on the hidden layer
    dropout : dropout rate
    lossfun : loss function

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
            sharping_factor,
            phase='train'
            ):
        super(MuVAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.bgru = nn.LSTM(1, hidden_dim, num_layers, dropout=dropout, bidirectional=True)

        # self.multi_view_attention = self.location_based_attention
        self.multi_view_attention = self.context_based_attention

        # location-based attention
        if self.multi_view_attention == self.location_based_attention:
            self.w_e = nn.Linear(hidden_dim * 2, 1)

        # context-based attention
        if self.multi_view_attention == self.context_based_attention:
            self.k = 128
            self.w_a = nn.Linear(self.k, input_dim)
            self.w_b = nn.Linear(self.k, input_dim)
            self.w_c = nn.Linear(self.k, input_dim)
            self.w_d = nn.Linear(input_dim, input_dim)
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            self.w_sc_i = nn.Conv2d(1, 1, (1, hidden_dim * 2 - self.k + 1), 1, 0)
            self.w_sc_t = nn.Conv2d(1, 1, (1, hidden_dim * 2 - self.k + 1), 1, 0)
            self.w_cc_1 = nn.Conv2d(1, 1, (1, hidden_dim * 2 - self.k + 1), 1, 0)
            self.w_cc_2 = nn.Conv2d(1, 1, (1, hidden_dim * 2 - self.k + 1), 1, 0)

        # hybrid_focus_procedure
        self.eps = 0.00001
        self.sharpening_factor = sharping_factor

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.attn_fusion_1 = nn.Conv2d(2, 16, 5, 1, 2)
        self.attn_fusion_2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.affine = nn.Linear(int(hidden_dim * 2 / 4) * int(input_dim / 4) * 32, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.loss = lossfun
        self.phase = phase


    def location_based_attention(self, hidden_matrix):
        # hidden_matrix: [batch, view, time, dim]
        _, _, time, _ = hidden_matrix.shape
        e_v = []
        for v in range(self.input_dim):
            e_t = []
            for t in range(time-1):
                e_t.append(self.w_e(hidden_matrix[:,v,t]).squeeze(1))
            e_v.append(torch.stack(e_t).permute(1, 0))
        return torch.stack(e_v).permute(1, 0, 2)

    def context_based_attention(self, hidden_matrix):
        # hidden_matrix: [batch, view, time, dim]
        batch_size, _, time_point, _ = hidden_matrix.shape
        ''' local self-context '''
        h_t = []
        for v in range(self.input_dim):
            h_t.append(self.w_sc_t(hidden_matrix[:,v,-1].view(hidden_matrix.shape[0],1,1,-1)))
        # h_t: [batch, 1, 1, k]
        h_t = torch.sum(torch.stack(h_t), dim=0)

        e_ti = []
        for t in range(time_point - 1):
            ''' target self-context '''
            h_i, h_ti = [], []
            for v in range(self.input_dim):
                h_i.append(self.w_sc_i(hidden_matrix[:,v,t].view(batch_size,1,1,-1)))
            h_i = torch.sum(torch.stack(h_i), dim=0)

            ''' cross-context '''
            h_ti = \
                self.w_cc_1(hidden_matrix[:,v,-1].view(batch_size,1,1,-1)) + \
                self.w_cc_2(hidden_matrix[:,v,t].view(batch_size,1,1,-1))

            ''' previous score information from attention matrix '''
            # hidden_matrix: [batch, view, time, dim]
            if t == 0:
                attn_weights = torch.sum(hidden_matrix[:,:,0] * hidden_matrix[:,:,-1], 2)
            else:
                attn_weights = torch.sum(hidden_matrix[:,:,t-1] * hidden_matrix[:,:,-1], 2)
            soft_attn_weights = self.softmax(attn_weights).unsqueeze(1).unsqueeze(2)

            ''' merge '''
            e_ti.append(torch.tanh(self.w_a(h_t) + self.w_b(h_i) + self.w_c(h_ti) + self.w_d(soft_attn_weights)))

        # e_ti: [batch, view, time]
        e_ti = torch.stack(e_ti).view(time_point-1, batch_size, self.input_dim).permute(1, 2, 0)
        return e_ti

    def hybrid_focus_procedure(self, energy_matrix):
        batch, view, time = energy_matrix.shape
        # beta_top: [batch, time]
        beta_top = torch.sum(self.relu(energy_matrix), dim=1)
        # beta: [batch, time]
        beta = torch.div(beta_top, torch.sum(beta_top, dim=1).unsqueeze(1) + self.eps)
        # e_sig: [batch, view, time]
        e_sig = torch.sigmoid(energy_matrix)
        # e_hat: [batch, view, time]
        e_hat = torch.div(e_sig, torch.sum(e_sig, dim=1).unsqueeze(1) + self.eps)
        e_hat = torch.mul(e_hat, beta.unsqueeze(1))
        e_hat = torch.mul(e_hat, self.sharpening_factor)
        # attention_matrix: [batch, view, time]
        attention_matrix = self.softmax(e_hat.reshape(batch, view*time)).view(batch, view, time)
        return attention_matrix

    def attentional_feature_fusion(self, hidden_matrix, attention_matrix):
        # context_matrix: [batch, view, dim]
        context_matrix = torch.matmul(attention_matrix.unsqueeze(2), hidden_matrix[:,:,:-1]).squeeze(2)
        # cat_matrix: [batch, channel, view, dim]
        cat_matrix = torch.stack([hidden_matrix[:,:,-1,:], context_matrix]).permute(1, 0, 2, 3)
        logit = self.pool(self.relu(self.attn_fusion_1(cat_matrix)))
        logit = self.pool(self.relu(self.attn_fusion_2(logit)))
        logit = self.affine(logit.view(logit.size()[0], -1))
        return logit

    def forward(self, input):
        hidden_matrix = []
        for v in range(self.input_dim):
            # lstm_out: [batch, time, dim]
            lstm_out, _ = self.bgru(input[:,:,v].unsqueeze(2))
            hidden_matrix.append(lstm_out)

        # hidden_matrix: [batch, view, time, dim]
        hidden_matrix = torch.stack(hidden_matrix).permute(1, 0, 2, 3)
        # energy_matrix: [batch, view, time]
        energy_matrix = self.multi_view_attention(hidden_matrix)
        # attention_matrix: [batch, view, time]
        attention_matrix = self.hybrid_focus_procedure(energy_matrix)
        # context_matrix: [batch, view, dim]
        logit = self.attentional_feature_fusion(hidden_matrix, attention_matrix)

        if self.phase == 'test':
            return logit, attention_matrix
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
