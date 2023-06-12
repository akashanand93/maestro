import os
import json
import torch
import argparse
from torch import nn
from expirement import ExpMain
import matplotlib.pyplot as plt
from dataloader import data_provider
from utills import read_default_args


parent_to_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_file = '{}/configs/config_shiva.json'.format(parent_to_parent_dir)
config = json.load(open(config_file))
data_dir = config['data_dir']
raw_dir = "{}/{}".format(data_dir, config['raw_data_dir'])
csv_dir = "{}/{}".format(data_dir, config['raw_data_csv'])
ltsf = "{}/ltsf".format(data_dir)

default_args = read_default_args()

args = {
   'root_path': ltsf,
   'checkpoints': '{}/checkpoints/'.format(data_dir),
   'data_path': '03_23.csv',
   'seq_len': 120,
   'pred_len': 30,
   'batch_size': 16,
   'learning_rate': 0.05,
   'train_only': False,
   'train_epochs': 20,
   'data_segment': None,
   'model': 'nlinear_attention',
   'enc_in': 397,
   'patience': 5,
   'target': 0,
   'stocks': None
}

for key, value in args.items():
    default_args[key] = value

args = argparse.Namespace(**default_args)

# create a new args parse
parser = argparse.ArgumentParser('Training script for MAESTRO')
parser.add_argument('--target', type=int, required=False, help='Target index', default=None)
parser.add_argument('--config', type=str, default=config_file, help='Config file path')
parser.add_argument('--correlated_stocks', type=str, required=False, default=None, help='Correlated stocks file path')
parser.add_argument('--min_correlation', type=int, required=False, default=0, help='minimum number of correlated stocks')


def train(target, suffix=''):

    print("\n-------------Training for target {}-------------\n".format(target))
    args.target = target
    exp = ExpMain(args)
    setting = 'mod_{}_sl{}_pl{}_ds_{}_tg_{}_ch_{}_{}'.format(args.model, args.seq_len, args.pred_len, args.data_path.split('.')[0], args.target, args.enc_in, suffix)
    print(exp.train(setting))


def train_correlated(sudo_target, correlated_stocks, suffix=''):

    print("\n-------------Training for target {}, corelated stocks {}-------------\n".format(sudo_target, correlated_stocks))
    args.target = 0
    args.stocks = correlated_stocks
    args.enc_in = len(correlated_stocks)
    exp = ExpMain(args)
    setting = 'mod_{}_sl{}_pl{}_ds_{}_tg_{}_ch_{}_{}'.format(args.model, args.seq_len, args.pred_len, args.data_path.split('.')[0], sudo_target, args.enc_in, suffix)
    print(exp.train(setting))


if __name__ == "__main__":
    
    args_train = parser.parse_args()
    target = args_train.target
    correlated_stocks = args_train.correlated_stocks
    
    if target is not None:
        train(target)

    elif correlated_stocks is not None:

        correlated_stocks = json.load(open(correlated_stocks, 'r'))
        correlated_stocks = {int(k): v for k, v in correlated_stocks.items() if len(v) >= args_train.min_correlation}
        print("Found {} stocks with minimum {} correlated stocks".format(len(correlated_stocks), args_train.min_correlation))

        for target, correlated in correlated_stocks.items():
            # remove target from correlated stocks and add it at firet position
            correlated = [target] + [c for c in correlated if c != target]
            train_correlated(target, correlated, suffix='corr_k50')

            
    
