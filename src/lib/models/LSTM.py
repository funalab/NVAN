import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
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
        super(LSTM, self).__init__()
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
