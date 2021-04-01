import copy
from copy import deepcopy
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv5DLSTM(nn.Module):
    """
    2D + 3D Convolution LSTM Network to predict born or abort embryo

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
        super(Conv5DLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv2d1 = nn.Conv2d(1, 8, 5, 1, 2)
        self.conv2d2 = nn.Conv2d(8, 16, 5, 1, 2)
        self.bn2d1 = nn.BatchNorm2d(8)
        self.bn2d2 = nn.BatchNorm2d(16)
        self.conv3d1 = nn.Conv3d(1, 8, 5, 1, 2)
        self.conv3d2 = nn.Conv3d(8, 16, 5, 1, 2)
        self.bn3d1 = nn.BatchNorm3d(8)
        self.bn3d2 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU()
        self.pool2d = nn.MaxPool2d(2, stride=2)
        self.pool3d = nn.MaxPool3d(2, stride=2)
        self.lstm = nn.LSTM(int((ip_size[0]/4) * (ip_size[1]/4) * (ip_size[2]/4) * 16 + (ip_size[1]/4) * (ip_size[2]/4) * 16),
                            hidden_dim, num_layers, dropout=dropout, bidirectional=True)
        self.affine = nn.Linear(hidden_dim * 2, num_classes)
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
        image_2d = input[:,:,0].unsqueeze(2)
        image_3d = input[:,:,1:].unsqueeze(2)
        batch, time, _, _, _ = image_2d.shape
        hidden_vec_2d, hidden_vec_3d = [], []
        for t in range(time):
            h = self.pool2d(self.relu(self.bn2d1(self.conv2d1(image_2d[:,t]))))
            h = self.pool2d(self.relu(self.bn2d2(self.conv2d2(h))))
            hidden_vec_2d.append(h)
            h = self.pool3d(self.relu(self.bn3d1(self.conv3d1(image_3d[:,t]))))
            h = self.pool3d(self.relu(self.bn3d2(self.conv3d2(h))))
            hidden_vec_3d.append(h)
        hidden_vec_2d = torch.stack(hidden_vec_2d).permute(1, 0, 2, 3, 4).view(batch, time, -1)
        hidden_vec_3d = torch.stack(hidden_vec_3d).permute(1, 0, 2, 3, 4, 5).view(batch, time, -1)
        hidden_vec = torch.cat((hidden_vec_2d, hidden_vec_3d), dim=2)
        lstm_out, _ = self.lstm(hidden_vec)
        attn_out, attn_weights = self.attention(lstm_out)
        logits = self.affine(attn_out)

        if self.phase == 'test':
            return logits, None
        else:
            return logits
