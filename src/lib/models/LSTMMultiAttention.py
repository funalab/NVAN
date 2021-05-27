import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMMultiAttention(nn.Module):
    """
    Multi-view Attention Network for Multivariate Temporal Data

    Parameters
    ----------
    input_dim : dimention of multivariable
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
            num_classes,
            num_layers,
            hidden_dim,
            dropout,
            lossfun,
            phase='train'
            ):
        super(LSTMMultiAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.bgru = nn.LSTM(1, hidden_dim, num_layers, dropout=dropout, bidirectional=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.attn_fusion_1 = nn.Conv2d(2, 16, 5, 1, 2)
        self.attn_fusion_2 = nn.Conv2d(16, 32, 5, 1, 2)
        if isinstance(lossfun, nn.CrossEntropyLoss):  # Multi-class classification
            self.affine = nn.Linear(int(hidden_dim * 2 / 4) * int(input_dim / 4) * 32, num_classes)
        elif isinstance(lossfun, nn.BCEWithLogitsLoss):
            self.affine = nn.Linear(int(hidden_dim * 2 / 4) * int(input_dim / 4) * 32, 1)
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

        if self.phase == 'test' or self.phase == 'validation':
            return logit, attn_weights_matrix
        else:
            return logit
