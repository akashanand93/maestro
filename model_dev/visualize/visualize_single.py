import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from model_dev.decider import Decider
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
        self.config = Config(self.seq_len, self.pred_len, self.channels, args.individual, args.decomp_kernal)

        # load model and weights
        self.checkpoint_path = os.path.join(args.checkpoints, setting, weights)
        Model = load_model(args.model)
        self.model = Model(self.config)
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        print("Load model from {}".format(self.checkpoint_path))
        self.model.eval()

        # data loader object
        self.dataloader, _ = data_provider(args, mode, return_date=True)

        # startegy decider object
        # self.decider = Decider(0, args.seq_len, args.pred_len, log=decision_log, min_cutoff=min_cutoff)

        if self.title_meta:
            # load instrument config for stock meta
            instrument_file = "{}/{}".format(args.root_path, args.instrument_file)
            instrumen_config = json.load(open(instrument_file, "r"))
            self.instrumen_config = {i['instrument_token']: i for i in instrumen_config}
            # read columns header from data_path which is csv
            data_path = "{}/{}".format(args.root_path, args.data_path)
            columns = pd.read_csv(data_path, nrows=1).columns[1:]
            self.index_to_column = {i: self.instrumen_config[int(columns[i])] for i in range(len(columns))}

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