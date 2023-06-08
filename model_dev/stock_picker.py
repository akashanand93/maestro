# this scripts calculates loss for each stock and pick top k best stocks according to loss

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from model_dev.dataloader import data_provider
from datetime import datetime
from model_dev.utills import load_model
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
        Model = load_model(args.model)
        self.model = Model(self.config)
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        print("Load model from {}".format(self.checkpoint_path))

        # data loader object
        self.dataloader, _ = data_provider(args, mode)

        # scaler for inverse transform
        self.scaler = StandardScaler()

    def pick_stocks(self, step=4, sort_by='loss_mean'):

        best_stocks = []
        loss_list = []

        for i in range(0, len(self.dataloader), step):

            x, y = self.dataloader[i]  # x is seq_len*channel, y is pred_len*channel

            y_pred = self.model(torch.from_numpy(x).unsqueeze(0).float())
            y_pred = y_pred[0].detach().squeeze().numpy()

            y = np.transpose(y)
            y_pred = np.transpose(y_pred)

            loss = loss_fn(torch.from_numpy(y_pred), torch.from_numpy(y))
            loss = torch.mean(loss, dim=1)
            # select loss for given stocks

            loss_list.append(loss.numpy())

        loss_list = np.array(loss_list) # shape is time_step*stock_num
        # reverse the loss list to be shape of stock_num*time_step
        loss_list = np.transpose(loss_list)

        mean_loss_list = np.mean(loss_list, axis=1)
        std_loss_list = np.std(loss_list, axis=1)

        # create a dict, where key is stock and value is a map containg mean and std
        loss_dict = {}
        for i in range(len(mean_loss_list)):
            loss_dict[i] = {'loss_mean': mean_loss_list[i], 'loss_std': std_loss_list[i]}

        # sort the dict by loss_mean
        sorted_loss_dict = sorted(loss_dict.items(), key=lambda x: x[1][sort_by], reverse=False)
        # return stock names
        return sorted_loss_dict
