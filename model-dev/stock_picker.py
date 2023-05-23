# this scripts calculates loss for each stock and pick top k best stocks according to loss

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from models import dlinear
from decider import Decider
import matplotlib.pyplot as plt
from dataloader import data_provider
from datetime import datetime
from sklearn.preprocessing import StandardScaler


# create a MSE loss function wjhich returns loss for each batch elem
loss_fn = nn.MSELoss(reduction='none')

class Config:
    def __init__(self, seq_len, pred_len, channels, individual, decomp_kernal):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = channels
        self.individual = individual
        self.decomp_kernal = decomp_kernal

class StockPicker:

    def __init__(self, args, setting, weights, mode='val'):

        self.mode = mode
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.enc_in
        self.config = Config(self.seq_len, self.pred_len, self.channels, args.individual, args.decomp_kernal)

        # load model and weights
        self.checkpoint_path = os.path.join(args.checkpoints, setting, weights)
        self.model = dlinear.DLinear(self.config)
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        print("Load model from {}".format(self.checkpoint_path))

        # data loader object
        self.dataloader, _ = data_provider(args, mode)

        # scaler for inverse transform
        self.scaler = StandardScaler()

    def pick_stocks(self, stocks, k, step=4):

        best_stocks = []
        num_stocks = len(stocks)
        loss_list = []

        for i in range(0, len(self.dataloader), step):

            x, y = self.dataloader[i]  # x is seq_len*channel, y is pred_len*channel

            y_pred = self.model(torch.from_numpy(x).unsqueeze(0).float())
            y_pred = y_pred[0].detach().squeeze().numpy()

            y = np.transpose(y)
            y_pred = np.transpose(y_pred)

            loss = loss_fn(torch.from_numpy(y_pred), torch.from_numpy(y))
            loss = torch.mean(loss, dim=1)[stocks]
            loss_list.append(loss.numpy().tolist())

        loss_list = np.array(loss_list)
        print(loss_list.shape)
        #     loss_list += loss.numpy()
        #
        # # create a mapping between stock name and loss
        # loss_dict = dict(zip(stocks, loss_list))
        # # sort the loss dict
        # loss_dict = {k: v for k, v in sorted(loss_dict.items(), key=lambda item: item[1])}
        # # pick top k stocks
        # best_stocks = list(loss_dict.keys())[:k]
        #
        # return best_stocks
