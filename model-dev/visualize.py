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


def draw_barchart_single(profits, day_times):

    assert len(profits) == len(day_times), "Lists must have the same length"
    fig, ax = plt.subplots()
    ax.bar(day_times, profits)
    fig.set_size_inches(18, 4)

    # create a vertical line at start of the each day in plot
    for i in range(1, len(day_times)):
        if day_times[i-1].split()[2].split('|')[-1] != day_times[i].split()[2].split('|')[-1]:
            ax.axvline(x=i, color='k', linestyle='--', linewidth=1)
            ax.text(i, 0.9, day_times[i].split()[2].split('|')[-1], transform=ax.get_xaxis_transform(), rotation=90, size=12)

    for i in range(1, len(day_times)):
        if day_times[i-1].split()[3].split('|')[-1] != day_times[i].split()[3].split('|')[-1]:
            ax.axvline(x=i, color='k', linestyle='--', linewidth=1)
            ax.text(i, 0.9, day_times[i].split()[3].split('|')[-1], transform=ax.get_xaxis_transform(), rotation=90, size=12)

    ax.set_title("Profits by Day-Time")
    ax.set_xlabel("Day-Time")
    ax.set_ylabel("Profits")
    plt.xticks(rotation=90)
    plt.show()


title_fields = ['exchange', 'name']

loss_fn = nn.MSELoss()

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

    def __init__(self, args, setting, weights, mode='train', title_meta=True, decision_log=False):

        self.mode = mode
        self.title_meta = title_meta
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.enc_in
        self.decision_log = decision_log
        self.config = Config(self.seq_len, self.pred_len, self.channels, args.individual, args.decomp_kernal)

        # load model and weights
        self.checkpoint_path = os.path.join(args.checkpoints, setting, weights)
        self.model = dlinear.DLinear(self.config)
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        print("Load model from {}".format(self.checkpoint_path))

        # data loader object
        self.dataloader, _ = data_provider(args, mode, return_date=True)

        # scaler for inverse transform
        self.scaler = StandardScaler()

        # startegy decider object
        self.decider = Decider(0, args.seq_len, args.pred_len, log=decision_log)

        if self.title_meta:
            # load instrument config for stock meta
            instrument_file = "{}/{}".format(args.root_path, args.instrument_file)
            instrumen_config = json.load(open(instrument_file, "r"))
            self.instrumen_config = {i['instrument_token']: i for i in instrumen_config}
            # read columns header from data_path which is csv
            data_path = "{}/{}".format(args.root_path, args.data_path)
            columns = pd.read_csv(data_path, nrows=1).columns[1:]
            self.index_to_column = {i: self.instrumen_config[int(columns[i])] for i in range(len(columns))}


    def plot(self, idx, stocks, plot=True, print_net_profit=True):

        x, y, date_info = self.dataloader[idx]  # x is seq_len*channel, y is pred_len*channel

        date_info = [convert_date_format(date) for date in date_info]
        start_date, end_date, pred_start_date, pred_end_date = date_info
        date_str = "Date range: {} - {} | Prediction range: {} - {}".format(start_date, end_date, pred_start_date,pred_end_date)

        y_pred = self.model(torch.from_numpy(x).unsqueeze(0).float())
        y_pred = y_pred[0].detach().squeeze().numpy()

        y_ = np.transpose(y)
        y_pred_ = np.transpose(y_pred)

        # permute x and y_pred to be channel*seq_len
        x = np.transpose(self.dataloader.inverse_transform(x))
        y = np.transpose(self.dataloader.inverse_transform(y))
        y_pred = np.transpose(self.dataloader.inverse_transform(y_pred))

        full_pred = np.concatenate((x, y_pred), axis=1)
        full_data = np.concatenate((x, y), axis=1)
        net_profit = 0
        fund_utilization = 0
        num_transactions = 0
        total_commision = 0

        for i, stock in enumerate(stocks):

            stock_meta_str = ''
            if self.title_meta:
                stock_meta = self.index_to_column[stock]
                stock_meta_str = ' | '.join(['{}: {}'.format(field, stock_meta[field]) for field in title_fields])

            loss = loss_fn(torch.from_numpy(y_pred_[stock]).unsqueeze(0), torch.from_numpy(y_[stock]).unsqueeze(0)).item()
            decision, profit, fund_utilized, commision = self.decider.decide(y_pred[stock], y[stock], x[stock][-1])
            fund_utilization += fund_utilized
            net_profit += profit
            total_commision += commision
            num_transactions += 1 if decision != 'hold' else 0

            if plot:
                title = '{}\n\n{} | loss: {:.4f} | decision: {}, profit: {} | {}'.format(stock_meta_str, i, loss, decision, profit, date_str)
                plt.figure(figsize=(11, 3)); plt.plot(full_data[stock], label='real');
                plt.plot(full_pred[stock], label='model');
                plt.legend(); plt.title(title, fontsize=10); plt.show()

            if print_net_profit:
                print("net profit: {}".format(net_profit))

        return net_profit, date_info, fund_utilization, num_transactions, total_commision
