import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def json_to_csv(config_file, split, sort, save_raw, train_split=0.8, val_split=0.1):

    config = json.load(open(config_file, 'r'))
    data_dir = config['data_dir']
    json_file = config['data_json_path']
    output_file = config['data_csv_path']

    # load the json file
    print("reading json file from {}".format(json_file))
    json_data = pickle.load(open(json_file, 'rb'))

    data_list = []
    for instrument, values in tqdm(json_data.items()):
        data_row = []
        for value in values:
            date = value['date']
            data_row.append([instrument, date, value['open'], value['high'], value['low'], value['close'], value['volume']])
        data_list.append(data_row)

    # create a dataframe
    df = pd.DataFrame(np.concatenate(data_list), columns=['instrument', 'datetime', 'open', 'high', 'low', 'close', 'volume'])

    # remove the timezone info from datetime string
    df['datetime'] = df['datetime'].apply(lambda x: x.replace(tzinfo=None))
    # convert instrument, year, month, day, hour, minute, volume to int
    df[['instrument', 'volume']] = df[['instrument', 'volume']].astype(int)

    if sort:
        print('sorting the dataframe')
        df = df.sort_values(by=['instrument', 'datetime'])

    if save_raw:
        print("saving raw data, size: {}".format(len(df)))
        df.to_csv(output_file, index=False)

    if split:
        # take instrument with all valid date present
        valid_time_start = '09:15:00'
        valid_time_end = '15:29:00'
        all_valid_minutes = pd.date_range(start=valid_time_start, end=valid_time_end, freq='1min').time
        all_valid_minutes = pd.Series(all_valid_minutes)

        # filter only instrument in all valid minutes are present, for one given day only
        print('filtering out instruments with missing minutes')
        print('size before filtering: {}, instruments: {}'.format(len(df), df['instrument'].nunique()))
        df = df.groupby(['instrument', df['datetime'].dt.date]).filter(lambda x: all_valid_minutes.isin(x['datetime'].dt.time).all())
        print('size of dataframe after filtering: {}, instruments: {}'.format(len(df), df['instrument'].nunique()))

        # take instrument with all valid date present
        # pick a instrument with max number of days, and create a list of all days from it
        print('filtering out instruments with missing days')
        all_valid_days = df.groupby('instrument')['datetime'].apply(lambda x: x.dt.date.unique()).apply(pd.Series).stack().reset_index(level=1, drop=True)
        all_valid_days = pd.Series(all_valid_days.unique())
        print('size of dataframe before filtering: {}'.format(len(df)))
        df = df.groupby('instrument').filter(lambda x: all_valid_days.isin(x['datetime'].dt.date).all())
        print('size of dataframe after filtering: {}'.format(len(df)))
        print('number of instruments in final dataframe: {}'.format(df['instrument'].nunique()))

        # seperate out end data for test and val as per day basis for each instrument
        print('splitting the data into train, val and test')
        # get the unique instruments
        dates = df['datetime'].dt.date.unique()

        num_days = len(dates)
        num_train_days = int(num_days * train_split)
        num_val_days = int(num_days * val_split)

        # get the train, val and test dates
        train_dates = dates[:num_train_days]
        val_dates = dates[num_train_days:num_train_days+num_val_days]
        test_dates = dates[num_train_days+num_val_days:]

        # get the train, val and test dataframes
        train_df = df[df['datetime'].dt.date.isin(train_dates)]
        val_df = df[df['datetime'].dt.date.isin(val_dates)]
        test_df = df[df['datetime'].dt.date.isin(test_dates)]

        # save the train, val and test dataframes
        train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

        print('train size: {}, val size: {}, test size: {}'.format(len(train_df), len(val_df), len(test_df)))


# write argparse function
def parse_args():
    parser = argparse.ArgumentParser(description='json to csv')
    parser.add_argument('--config', help='configuration file', required=True)
    parser.add_argument('--sort', help='sort the data', default=1, type=int)
    parser.add_argument('--save_raw', help='save the raw data', default=0, type=int)
    parser.add_argument('--split', help='split the data into train, val and test', default=1, type=int)
    args = parser.parse_args()
    return args


# main function
if __name__ == '__main__':

    args = parse_args()
    sort = args.sort
    split = args.split
    save_raw = args.save_raw
    config_file = args.config
    json_to_csv(config_file, split=split, sort=sort, save_raw=save_raw)