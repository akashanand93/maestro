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
        self.individual = False
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.Attention = nn.Linear(self.channels, 1)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = x.permute(0, 2, 1)
            x = self.Linear(x)  # [Batch, Channel, Output length]
            x = x.permute(0, 2, 1)  # [Batch, Output length, Channel]
            x = x + seq_last
            x = x.reshape(-1, self.channels)
            # apply tanh activation
            x = torch.tanh(x)
            x = self.Attention(x)  # [Batch, Output length, 1]
            x = x.reshape(-1, self.pred_len)
        return x  # [Batch, Output length, Channel]