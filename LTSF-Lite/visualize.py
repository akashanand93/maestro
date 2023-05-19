import os
import sys
import json
import torch
# import MSE loss
import torch.nn as nn
import numpy as np
from models import dlinear
import matplotlib.pyplot as plt
from dataloader import data_provider
from sklearn.preprocessing import StandardScaler

title_fields = ['exchange', 'name', 'instrument_token']

loss_fn = nn.MSELoss()

class Config:
    def __init__(self, seq_len, pred_len, channels, individual):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = channels
        self.individual = individual

class Visualize:

    def __init__(self, args, setting, mode='train'):

        self.mode = mode
        self.setting = setting
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.enc_in
        self.config = Config(self.seq_len, self.pred_len, self.channels, args.individual)
        self.checkpoint_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
        self.model = dlinear.DLinear(self.config)
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.dataloader, _ = data_provider(args, mode)
        self.scaler = StandardScaler()

    def plot(self, idx, stocks, sort=None):

        x, y = self.dataloader[idx] # x is seq_len*channel, y is pred_len*channel
        y_pred = self.model(torch.from_numpy(x).unsqueeze(0).float())
        y_pred = y_pred[0].detach().squeeze().numpy()

        y_ = np.transpose(y)
        y_pred_ = np.transpose(y_pred)

        # cerate loss for each stock
        loss = []
        for stock in stocks:
            loss.append(loss_fn(torch.from_numpy(y_pred_[stock]).unsqueeze(0), torch.from_numpy(y_[stock]).unsqueeze(0)).item())

        # permute x and y_pred to be channel*seq_len
        x = np.transpose(self.dataloader.inverse_transform(x))
        y = np.transpose(self.dataloader.inverse_transform(y))
        y_pred = np.transpose(self.dataloader.inverse_transform(y_pred))

        full_pred = np.concatenate((x, y_pred), axis=1)
        full_data = np.concatenate((x, y), axis=1)

        # sort the loss in descending order with the index
        loss = dict(zip(stocks, loss))

        if sort not in ['asc', 'desc']:
            print("Please enter a valid sorting order: asc or desc")
            return

        if sort == 'asc':
            loss = {k: v for k, v in sorted(loss.items(), key=lambda item: item[1], reverse=False)}
        elif sort == 'desc':
            loss = {k: v for k, v in sorted(loss.items(), key=lambda item: item[1], reverse=True)}

        # create a figure for each stock in sorting order
        for stock in loss:
            title = 'Loss: {}'.format(round(loss[stock], 5))
            plt.figure(figsize=(11, 3)); plt.plot(full_data[stock], label='real'); plt.plot(full_pred[stock], label='model'); plt.legend(); plt.title(title); plt.show()



