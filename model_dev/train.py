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
   'seq_len': 600,
   'pred_len': 60,
   'batch_size': 16,
   'learning_rate': 0.025,
   'train_only': False,
   'train_epochs': 20,
   'data_segment': None,
   'model': 'nlinear_attention',
   'enc_in': 270,
   'patience': 5,
   'target': 0
}

for key, value in args.items():
    default_args[key] = value

args = argparse.Namespace(**default_args)
# setting = 'mod_{}_sl{}_pl{}_ds_{}_tg_{}'.format(args.model, args.seq_len, args.pred_len, args.data_path.split('.')[0], args.target)


if __name__ == "__main__":
    
    # exp = ExpMain(args)
    # exp.test(setting, 'train')

    for i in range(args.enc_in):
        print("\n-------------Training for target {}-------------\n".format(i))
        args.target = i
        exp = ExpMain(args)
        setting = 'mod_{}_sl{}_pl{}_ds_{}_tg_{}'.format(args.model, args.seq_len, args.pred_len, args.data_path.split('.')[0], args.target)
        exp.train(setting)
        del exp