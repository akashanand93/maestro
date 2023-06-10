import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Use this line if you want to visualize.py the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in

        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.Attention = nn.Linear(self.channels, 1)
        # self.BatchNorm = nn.BatchNorm1d(self.channels)

    def forward(self, x):

        # # apply batch norm 1D
        # x = x.permute(0, 2, 1)
        # x = self.BatchNorm(x)
        # x = x.permute(0, 2, 1)

        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        
        x = x.permute(0, 2, 1)
        x = self.Linear(x)  # [Batch, Channel, Output length]
        x = x.permute(0, 2, 1)  # [Batch, Output length, Channel]

        x = x + seq_last

        x = x.reshape(-1, self.channels)
        x = self.Attention(x)  # [Batch, Output length, 1]
        x = x.reshape(-1, self.pred_len)
        return x  # [Batch, Output length, Channel]
    