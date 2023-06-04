import os
import sys
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from utills import detect_constant_price
warnings.filterwarnings('ignore')

class Transoform(object):

    def __init__(self, config_file, channel='open'):

        self.channel = channel
        config = json.load(open(config_file, 'r'))
        self.data_dir = config['data_dir']
        self.inp_data = "{}/{}".format(self.data_dir, config['raw_data_csv'])
        self.transform_map = {'ltsf': self.ltsf, 'backtrader': self.backtrader}
        self.all_channels = ['open', 'high', 'low', 'close', 'volume']

    def ltsf(self, nan_threshold=0.97, mean_thresh_price=5, mean_trade_thresh=20000000):

        for file in os.listdir(self.inp_data):

            print("\n-------------Processing file: {}------------\n".format(file))
            cols = pd.read_csv("{}/{}".format(self.inp_data, file)).columns
            price_cols = [i for i in cols if i.endswith(self.channel)]
            vol_cols = [i for i in cols if i.endswith('volume')]
            df = pd.read_csv("{}/{}".format(self.inp_data, file), usecols=price_cols + vol_cols + ['date'])

            # create list ofcolumns wihtout channel
            cols = [i.split('_')[0] for i in price_cols]
            cols_to_drop = []
            for col in cols:
                mean_trade_value = df['{}_volume'.format(col)].mean() * df['{}_avg'.format(col)].mean()
                if mean_trade_value <= mean_trade_thresh:
                    cols_to_drop.append(col)

            # drop all volumne columns
            df.drop(columns=vol_cols, inplace=True)
            # drop cols_to_drop with channel
            print("Dropping low trade columns, current shape: {}".format(df.shape))
            df.drop(columns=["{}_{}".format(i, self.channel) for i in cols_to_drop], inplace=True)

            # rename xyz_<channel> columns to xyz
            df.columns = [i.split('_')[0] if i.endswith(self.channel) else i for i in df.columns]

            print("Dropping high nan columns, current shape: {}".format(df.shape))
            df.dropna(axis=1, thresh=int(df.shape[0] * nan_threshold), inplace=True)

            print("Dropping low price columns, current shape: {}".format(df.shape))
            df = df.drop([i for i in df.columns if df[i].dtype == np.float64 and df[i].mean() <= mean_thresh_price], axis=1)

            print("Interpolating remaining NaNs, current nan count: {}".format(df.isnull().sum().sum()))
            df = df.interpolate(method='linear', axis=0, limit_direction='both')
            print("Final shape: {}".format(df.shape))
            df.to_csv("{}/ltsf/{}".format(self.data_dir, file), index=False)

    def backtrader(self):

        backtrader_dir = "{}/{}".format(self.data_dir, 'backtrader')
        if not os.path.exists(backtrader_dir):
            os.makedirs(backtrader_dir)

        inp_df = pd.read_csv(self.inp_data, parse_dates=['datetime'])

        # for each instrument create a new df
        for instrument in tqdm(inp_df['instrument'].unique()):
            df = inp_df[inp_df['instrument'] == instrument]
            df.drop('instrument', axis=1, inplace=True)
            df['date'] = df['datetime'].dt.date
            df['time'] = df['datetime'].dt.time
            df.drop('datetime', axis=1, inplace=True)
            df = df[['date', 'time'] + self.all_channels]
            df.to_csv("{}/{}.csv".format(backtrader_dir, instrument), index=False)

def parse_args():

    parser = argparse.ArgumentParser(description='json to csv')
    parser.add_argument('--config', help='configuration file', required=True)
    parser.add_argument('--trans_typ', help='type of transform', default='ltsf')
    parser.add_argument('--channel', help='channel to consider', default='avg')
    args = parser.parse_args()
    return args


# main function
if __name__ == '__main__':

    args = parse_args()
    config_file = args.config
    trans_typ = args.trans_typ.lower()
    channel = args.channel.lower()
    transform = Transoform(config_file, channel=channel)
    transform.transform_map[trans_typ]()