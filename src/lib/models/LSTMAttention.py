import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAttention(nn.Module):
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
        super(LSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=True)
        if isinstance(lossfun, nn.CrossEntropyLoss):  # Multi-class classification
            self.affine = nn.Linear(hidden_dim*2, num_classes)
        elif isinstance(lossfun, nn.BCEWithLogitsLoss):
            self.affine = nn.Linear(hidden_dim*2, 1)
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
        lstm_out, _ = self.lstm(input.permute(1, 0, 2))
        lstm_out = lstm_out.permute(1, 0, 2)
        attn_out, attn_weights = self.attention(lstm_out)
        logits = self.affine(attn_out)

        if self.phase == 'test' or self.phase == 'validation':
            return logits, None
        else:
            return logits
