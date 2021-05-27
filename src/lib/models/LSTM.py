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
        if isinstance(lossfun, nn.CrossEntropyLoss):  # Multi-class classification
            self.affine = nn.Linear(hidden_dim*2, num_classes)
        elif isinstance(lossfun, nn.BCEWithLogitsLoss):
            self.affine = nn.Linear(hidden_dim*2, 1)
        self.softmax = nn.Softmax(dim=1)
        self.loss = lossfun
        self.phase = phase

    def forward(self, input):
        lstm_out, _ = self.lstm(input.view(input.shape[1], input.shape[0], -1))
        logits = self.affine(lstm_out[-1, :, :])

        if self.phase == 'test' or self.phase == 'validation':
            return logits, None
        else:
            return logits
