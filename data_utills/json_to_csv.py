import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utills import detect_constant_price


def json_to_csv(inp_file, output_file):

    print("reading json file from {}".format(inp_file))
    if inp_file.endswith('.pickle'):
        json_data = pickle.load(open(inp_file, 'rb'))
    elif inp_file.endswith('.json'):
        json_data = json.load(open(inp_file, 'r'))
    else:
        raise Exception("json file should be either json or pickle")

    with open(output_file, 'w') as outfile:
        outfile.write("instrument,datetime,open,high,low,close,volume\n")
        for instrument, values in tqdm(json_data.items()):
            # write header
            for value in values:
                date = value['date']
                if inp_file.endswith('.json'):
                    # remove timezone info from datetime string
                    date = date[:-6]
                else:
                    date = date.tz_localize(None)

                outfile.write(
                    "{},{},{},{},{},{},{}\n".format(int(instrument), date, value['open'], value['high'], value['low'], value['close'], int(value['volume'])))

    print("csv file saved at {}".format(output_file))


def datagen(config_file, filter, read_csv, train_split=0.8, val_split=0.1):

    config = json.load(open(config_file, 'r'))
    data_dir = config['data_dir']
    inp_file = "{}/{}".format(data_dir, config['raw_data'])
    output_file = "{}/{}".format(data_dir, config['raw_data_csv'])
    filterd_data = "{}/{}".format(data_dir, config['filterd_data'])

    if read_csv:
        df = pd.read_csv(output_file, parse_dates=['datetime'])
        print("\nfile size: {} Mb".format(sys.getsizeof(df)/1024/1024))
    else:
        json_to_csv(inp_file, output_file)
        return

    if filter:
        # take instrument with all valid date present
        valid_time_start = '09:15:00'
        valid_time_end = '15:29:00'
        all_valid_minutes = pd.date_range(start=valid_time_start, end=valid_time_end, freq='1min').time
        all_valid_minutes = pd.Series(all_valid_minutes)

        print('\nfiltering out instruments with missing minutes')
        print('size before filtering: {}, instruments: {}'.format(len(df), df['instrument'].nunique()))
        df = df.groupby(['instrument', df['datetime'].dt.date]).filter(lambda x: all_valid_minutes.isin(x['datetime'].dt.time).all())
        print('size of dataframe after filtering: {}, instruments: {}'.format(len(df), df['instrument'].nunique()))

        print('\nfiltering out instruments with missing days')
        all_valid_days = df.groupby('instrument')['datetime'].apply(lambda x: x.dt.date.unique()).apply(pd.Series).stack().reset_index(level=1, drop=True)
        all_valid_days = pd.Series(all_valid_days.unique())
        print('size before filtering: {}, instruments: {}'.format(len(df), df['instrument'].nunique()))
        df = df.groupby('instrument').filter(lambda x: all_valid_days.isin(x['datetime'].dt.date).all())
        print('size of dataframe after filtering: {}, instruments: {}'.format(len(df), df['instrument'].nunique()))

        print('\nfiltering out instruments where volume is zero')
        print('size before filtering: {}, instruments: {}'.format(len(df), df['instrument'].nunique()))
        df = df.groupby('instrument').filter(lambda x: x['volume'].sum() != 0)
        print('size of dataframe after filtering: {}, instruments: {}'.format(len(df), df['instrument'].nunique()))

        # drop constant open price instrument completely
        print('\nfiltering out instruments with constant open price')
        print('size before filtering: {}, instruments: {}'.format(len(df), df['instrument'].nunique()))
        df = df.groupby('instrument').filter(lambda x: detect_constant_price(x['open'], duration=3) == True)
        print('size of dataframe after filtering: {}, instruments: {}'.format(len(df), df['instrument'].nunique()))

        df.to_csv(filterd_data, index=False)


# write argparse function
def parse_args():
    parser = argparse.ArgumentParser(description='json to csv')
    parser.add_argument('--config', help='configuration file', required=True)
    parser.add_argument('--filter', help='split the data into train, val and test', default=1, type=int)
    parser.add_argument('--read_csv', help='read csv file', default=1, type=int)
    args = parser.parse_args()
    return args


# main function
if __name__ == '__main__':

    args = parse_args()
    filter = args.filter
    config_file = args.config
    read_csv = args.read_csv
    datagen(config_file, filter=filter, read_csv=read_csv)