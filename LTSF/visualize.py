import os
import sys
import json
import torch
import numpy as np
from models import dlinear
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maestro.dataloader.dataloader import HistoricalDataCSV

title_fields = ['exchange', 'name', 'instrument_token']

class Config:
    def __init__(self, seq_len, pred_len, channels, individual):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = channels
        self.individual = individual

class Visualize:

    def __init__(self, seq_len, pred_len, channels, individual, checkpoint):

        path_config = os.path.join(os.getcwd(), "config.json")
        path_config = json.load(open(path_config, "r"))
        self.base_code = path_config["base_code"]
        self.data_dir = path_config["data_dir"]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual
        self.config = Config(seq_len, pred_len, channels, individual)
        self.checkpoint_path = os.path.join(self.base_code, "LTSF/checkpoints/", checkpoint, "checkpoint.pth")
        self.model = dlinear.DLinear(self.config)
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.dataloader = HistoricalDataCSV(self.data_dir, mode='train', sequence_length=seq_len+pred_len, predict_window=10, price_cutoff=1, meta=True)


    def plot(self, idx, stocks):

        data = self.dataloader[idx][0]
        x = data[:, :self.seq_len]
        metadata = self.dataloader[idx][2]
        y_pred = self.model(torch.from_numpy(x).permute(1, 0).unsqueeze(0).float())
        y_pred = y_pred[0].detach().squeeze().permute(1, 0).numpy()
        full_pred = np.concatenate((x, y_pred), axis=1)

        for stock in stocks:
            title = metadata[stock]
            # filter out the fields we don't want to display
            title = {k: v for k, v in title.items() if k in title_fields}
            # format dict into pretty string
            title = ' | '.join('{}: {}'.format(k, v) for k, v in title.items())
            plt.figure(figsize=(11, 3)); plt.plot(data[stock], label='real'); plt.plot(full_pred[stock], label='model'); plt.legend(); plt.title(title); plt.show()





