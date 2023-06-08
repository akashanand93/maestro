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


def convert_date_format(date_string):

    dt = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    new_date_string = dt.strftime("%a %H:%M")
    return new_date_string


class Config:
    def __init__(self, seq_len, pred_len, channels, individual, decomp_kernal):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = channels
        self.individual = individual
        self.decomp_kernal = decomp_kernal


class Scatter:

    def __init__(self, args, setting, weights, mode='train', title_meta=True):

        self.title_fields = ['exchange', 'name']
        self.loss_fn = nn.MSELoss()

        self.mode = mode
        self.title_meta = title_meta
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.enc_in
        self.target = args.target
        self.config = Config(self.seq_len, self.pred_len, self.channels, args.individual, args.decomp_kernal)

        # load model and weights
        self.checkpoint_path = os.path.join(args.checkpoints, setting, weights)

        Model = load_model(args.model)
        self.model = Model(self.config)
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        print("Load model from {}".format(self.checkpoint_path))

        # data loader object
        self.dataloader, _ = data_provider(args, mode, return_date=True)

        if self.title_meta:
            # load instrument config for stock meta
            instrument_file = "{}/{}".format(args.root_path, args.instrument_file)
            instrumen_config = json.load(open(instrument_file, "r"))
            self.instrumen_config = {i['instrument_token']: i for i in instrumen_config}
            # read columns header from data_path which is csv
            data_path = "{}/{}".format(args.root_path, args.data_path)
            columns = pd.read_csv(data_path, nrows=1).columns[1:]
            self.index_to_column = {i: self.instrumen_config[int(columns[i])] for i in range(len(columns))}

    def plot(self, segment=None):

        stock = self.target
        # run the model on each datapoint sequentially
        self.model.eval()
        full_signal = np.zeros(len(self.dataloader) + self.seq_len + self.pred_len - 1)
        full_pred = [[] for i in range(len(full_signal))]

        day_list = []
        for minute in range(len(self.dataloader)):

            x, y, date_info = self.dataloader[minute]   # x is seq_len*channel, y is pred_len*channel

            date_info = [convert_date_format(date).split()[0] for date in date_info]
            day_list.append(date_info[0])

            y_pred = self.model(torch.from_numpy(x).unsqueeze(0).float())
            y_pred = y_pred[0].detach().squeeze().numpy()

            # inverse transform
            y_pred = np.repeat(y_pred, self.channels).reshape(self.pred_len, self.channels)
            y_pred = self.dataloader.inverse_transform(y_pred)
            y = self.dataloader.inverse_transform(y)
            x = self.dataloader.inverse_transform(x)

            # select stock from x, y, y_pred
            x = x[:, stock]
            y = y[:, stock]
            y_pred = y_pred[:, stock]

            full_signal[minute] = x[0]
            for pred_minute in range(self.pred_len):
                full_pred[minute + self.seq_len + pred_minute].append(y_pred[pred_minute])

            # if the last step, add last seq_len data to the end of full_signal
            if minute == len(self.dataloader) - 1:
                full_signal[minute + 1: minute + self.seq_len] = x[1:]
                full_signal[minute + self.seq_len:] = y

        if segment is not None:
            full_signal = full_signal[segment[0]: segment[1]]
            full_pred = full_pred[segment[0]: segment[1]]
            day_list = day_list[segment[0]: segment[1]]

        scatter_x = []
        scatter_y = []
        for minute in range(len(full_pred)):
            for pred in full_pred[minute]:
                scatter_x.append(minute)
                scatter_y.append(pred)

        fig, ax = plt.subplots();
        fig.set_size_inches(18, 4.5)
        ax.plot(full_signal, label='signal')
        ax.scatter(scatter_x, scatter_y, label='prediction', color='red', s=1)

        # plot vertical lines for each beginning of the day
        for i in range(1, len(day_list)):
            if day_list[i - 1] != day_list[i]:
                ax.axvline(x=i, color='red', linestyle='--', linewidth=0.6)
                ax.text(i, 0.02, day_list[i], transform=ax.get_xaxis_transform(), rotation=0, size=10,
                        color='green')

        plt.show()

        return full_pred





