import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import reduce
from utills import detect_constant_price


def json_to_df(instrument, data):

    data_dict = {}
    for records in data:
        for property, value in records.items():
            if property == 'date':
                # remove the timezone info from datetime string
                value = value[:-6].replace('T', ' ')
                data_dict.setdefault(property, [])
                data_dict[property].append(value)
            else:
                column = "{}_{}".format(instrument, property)
                data_dict.setdefault(column, [])
                data_dict[column].append(value)
        property = 'avg'
        column = "{}_{}".format(instrument, property)
        data_dict.setdefault(column, [])
        data_dict[column].append(np.mean([records['open'], records['high'], records['low'], records['close']]))
    df = pd.DataFrame(data_dict)
    return df


def datagen(config_file, date):

    config = json.load(open(config_file, 'r'))
    base_dir = config['data_dir']
    inp_dir = "{}/{}".format(base_dir, config['raw_data_dir'])
    out_dir = "{}/{}".format(base_dir, config['raw_data_csv'])

    if not date:
        # file names are like <stock>_03_23.json, find out all the unique dates like `03_23`
        all_dates = set(['_'.join(i.split('.')[0].split('_')[1:]) for i in os.listdir(inp_dir)])
        print("Dates found: {}".format(all_dates))
    else:
        all_dates = [date]

    for date in all_dates:

        print("--------------Processing date: {}------------\n".format(date))

        merged_df = pd.DataFrame()
        files = [i for i in os.listdir(inp_dir) if i.split('.')[0].endswith(date)]

        for file in tqdm(files):
            instrument = file.split('.')[0].split('_')[0]
            json_data = json.load(open("{}/{}".format(inp_dir, file), 'r'))
            if len(json_data) < 10000:
                continue
            df = json_to_df(instrument, json_data)
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='date', how='outer')

        merged_df = merged_df.sort_values(by=['date'])
        print("Merged DFs, Shape: {}".format(merged_df.shape))
        merged_df.to_csv("{}/{}.csv".format(out_dir, date), index=False)


# write argparse function
def parse_args():
    parser = argparse.ArgumentParser(description='json to csv')
    parser.add_argument('--config', help='configuration file', required=True)
    parser.add_argument('--date', required=False, default=None, help='date files to be processed, if not provided, all dates will be processed')
    args = parser.parse_args()
    return args


# main function
if __name__ == '__main__':

    args = parse_args()
    config_file = args.config
    date = args.date
    datagen(config_file, date)
