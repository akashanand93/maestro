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

    def ltsf(self):

        for file in os.listdir(self.inp_data):
            cols = pd.read_csv("{}/{}".format(self.inp_data, file)).columns
            desired_cols = [i for i in cols if i.endswith(self.channel)] + ['date']
            df = pd.read_csv("{}/{}".format(self.inp_data, file), usecols=desired_cols)
            # rename xyz_<channel> columns to xyz
            df.columns = [i.split('_')[0] if i.endswith(self.channel) else i for i in df.columns]
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