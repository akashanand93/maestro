import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
from model_dev.decider import Decider
import matplotlib.pyplot as plt
from model_dev.dataloader import data_provider
from datetime import datetime
from model_dev.utills import get_stock_meta
from model_dev.utills import load_model
from sklearn.preprocessing import StandardScaler


def convert_date_format(date_string):

    dt = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    new_date_string = dt.strftime("%a %H:%M")
    return new_date_string


class Visualize:

    def __init__(self, args, setting, weights, mode='train', title_meta=True, decision_log=False, min_cutoff=0):

        self.title_fields = ['exchange', 'name']
        self.loss_fn = nn.MSELoss()

        self.mode = mode
        self.target = args.target
        self.title_meta = title_meta
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.enc_in
        self.decision_log = decision_log

        # load model and weights
        self.checkpoint_path = os.path.join(args.checkpoints, setting, weights)
        self.model = load_model(args)
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        print("Load model from {}".format(self.checkpoint_path))
        self.model.eval()

        # data loader object
        self.dataloader, _ = data_provider(args, mode, return_date=True)

        # startegy decider object
        # self.decider = Decider(0, args.seq_len, args.pred_len, log=decision_log, min_cutoff=min_cutoff)

        if self.title_meta:
            self.index_to_column = get_stock_meta

    def plot(self, idx, plot=True, plt_len=100):

        x, y, date_info = self.dataloader[idx]  # x is seq_len*channel, y is pred_len*channel

        y_pred = self.model(torch.from_numpy(x).unsqueeze(0).float())
        y_pred = y_pred[0].detach().squeeze().numpy()
        
        # # permute x and y_pred to be channel*seq_len
        # y_ = np.transpose(y)
        # y_pred_ = np.transpose(y_pred)

        y_ = y.copy()
        y_pred_ = y_pred.copy()

        x = self.dataloader.inverse_transform(x)
        y = self.dataloader.inverse_transform(y)

        # make y_pred (60*8) by repeating it channel times
        y_pred = np.repeat(y_pred, self.channels).reshape(self.pred_len, self.channels)
        y_pred = self.dataloader.inverse_transform(y_pred)

        full_pred = np.concatenate((x[:,self.target], y_pred[:,self.target]))
        full_data = np.concatenate((x[:,self.target], y[:,self.target]))

        loss = self.loss_fn(torch.from_numpy(y_pred_), torch.from_numpy(y_[:,self.target])).item()

        if plot:
            title = 'loss: {}'.format(loss)
            plt.figure(figsize=(11, 3)); plt.plot(full_data, label='real');
            plt.plot(full_pred, label='model'); plt.title(title)
            plt.legend(); plt.show()

        return y_pred_, y_[:,self.target], loss
    


def craete_heatmap(stocks_matrix, names):

        # Assuming your matrix is called 'stocks'
    # Scale down the size while maintaining the aspect ratio
    width = stocks_matrix.shape[1]//5
    height = stocks_matrix.shape[0]//5

    # Set the figure size
    plt.figure(figsize=(width, height))

    # Create a heatmap
    ax = sns.heatmap(stocks_matrix, cmap='RdBu', square=True, vmin=-1, vmax=1)
    plt.yticks(rotation=0)
    plt.xticks(np.arange(stocks_matrix.shape[1]), names, rotation=90)
    plt.tick_params(axis='x', labelsize=8)

    # plot names on y axis
    plt.yticks(np.arange(stocks_matrix.shape[0]), names[:stocks_matrix.shape[0]], rotation=0)
    plt.tick_params(axis='y', labelsize=8)

    # Label the colorbar
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('value')

    # Give your heatmap a title
    plt.title('Stocks Matrix Heatmap')

    plt.show()