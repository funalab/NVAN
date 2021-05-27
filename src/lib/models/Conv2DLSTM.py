import copy
from copy import deepcopy
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DLSTM(nn.Module):
    """
    2D Convolution LSTM Network to predict born or abort embryo

    Parameters
    ----------
    input_dim : dimention of multivariable
    num_classes : number of the output classes
    num_layers : number of hidden layers
    hidden_dim : number of unit on the hidden layer
    dropout : dropout rate
    lossfun : loss function
    phase : specify phase of training [train, validation, test]

    Inputs
    ------
    input: tensor, shaped [batch, time, view]

    Outputs
    -------
    output: tensor, shaped [batch, num_classes]
    """

    def __init__(
            self,
            input_dim,
            num_classes,
            num_layers,
            hidden_dim,
            dropout,
            ip_size,
            lossfun,
            phase='train'
            ):
        super(Conv2DLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(1, 8, 5, 1, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.lstm = nn.LSTM(int((ip_size[0]/4) * (ip_size[1]/4) * 16),
                            hidden_dim, num_layers, dropout=dropout, bidirectional=True)
        if isinstance(lossfun, nn.CrossEntropyLoss):  # Multi-class classification
            self.affine = nn.Linear(hidden_dim * 2, num_classes)
        elif isinstance(lossfun, nn.BCEWithLogitsLoss):
            self.affine = nn.Linear(hidden_dim * 2, 1)
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
        # input: [batch, time, 1, z, y, x]
        batch, time, _, _ = input.shape
        input = input.unsqueeze(2)
        hidden_vec = []
        for t in range(time):
            h = self.pool(self.relu(self.bn1(self.conv1(input[:,t]))))
            h = self.pool(self.relu(self.bn2(self.conv2(h))))
            hidden_vec.append(h)
        hidden_vec = torch.stack(hidden_vec).permute(1, 0, 2, 3, 4).view(batch, time, -1)
        lstm_out, _ = self.lstm(hidden_vec)
        attn_out, attn_weights = self.attention(lstm_out)
        logits = self.affine(attn_out)

        if self.phase == 'test' or self.phase == 'validation':
            return logits, None
        else:
            return logits
