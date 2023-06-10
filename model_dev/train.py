import os
import json
import torch
import argparse
from torch import nn
from expirement import ExpMain
import matplotlib.pyplot as plt
from dataloader import data_provider
from utills import read_default_args


config_file = '/Users/shiva/Desktop/maestro/configs/config_shiva.json'
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
   'enc_in': 270,
   'patience': 3,
   'target': 0,
   'stocks': None
}

for key, value in args.items():
    default_args[key] = value

args = argparse.Namespace(**default_args)

# create a new args parse
parser = argparse.ArgumentParser('Training script for MAESTRO')
parser.add_argument('--target', type=int, required=True, help='Target index')


if __name__ == "__main__":
    
    args_train = parser.parse_args()

    print("\n-------------Training for target {}-------------\n".format(args_train.target))
    args.target = args_train.target
    exp = ExpMain(args)
    setting = 'mod_{}_sl{}_pl{}_ds_{}_tg_{}_ch_{}'.format(args.model, args.seq_len, args.pred_len, args.data_path.split('.')[0], args.target, args.enc_in)
    print(exp.train(setting))
