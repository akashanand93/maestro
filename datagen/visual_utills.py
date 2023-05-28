import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt

date_format = "%Y-%m-%d %H:%M:%S"
days = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
sort_by_options = ['mean_vol', 'mean_price', 'mean_trade_value', 'price_std', 'vol_std', 'missing_values']


def property_map(df, cols, sort_by, reverse):

    """
    Returns a map of column name to its mean volume, mean price, mean trade value, price std, volume std etc
    """

    col_volume_map = {}
    for col in cols:
        vol_col = '{}_volume'.format(col)
        price_col = '{}_avg'.format(col)
        mean_vol = int(df[vol_col].mean())
        mean_price = round(df[price_col].mean(), 2)
        mean_trade_value = int(mean_vol * mean_price)
        price_std = round(df[price_col].std(), 2)
        vol_std = round(df[vol_col].std(), 2)
        missing_values = df[vol_col].isnull().sum()
        col_volume_map[col] = {'mean_vol': mean_vol, 'mean_price': mean_price, 'mean_trade_value': mean_trade_value, 'price_std': price_std, 'vol_std': vol_std, 'missing_values': missing_values}

    if sort_by:
        col_volume_map = {k: v for k, v in sorted(col_volume_map.items(), key=lambda item: item[1][sort_by], reverse=reverse)}
    return col_volume_map


def create_day_list(df):

    """
    Returns a list of days of the week for each datetime in the dataframe, sequentially
    """

    datetime_list = df['date'].to_list()
    day_list = []
    for i in datetime_list:
        datetime_obj = datetime.strptime(i, date_format)
        day_of_week = days[datetime_obj.weekday()]
        day_list.append(day_of_week)
    return day_list


def beautify_str(str):

    """
    Returns a beautified string like 'mean_price' to 'Mean Price'
    """
    return ' '.join([i.capitalize() for i in str.split('_')])


class VisualizeData:

    def __init__(self, config, df, sort_by=None, reverse=False):

        self.df = df
        self.sort_by = sort_by
        self.config = json.load(open(config))
        self.data_dir = self.config['data_dir']
        self.raw_dir = "{}/{}".format(self.data_dir, self.config['raw_data_dir'])
        self.csv_dir = "{}/{}".format(self.data_dir, self.config['raw_data_csv'])
        self.ltsf = "{}/ltsf".format(self.data_dir)

        # load instrument meta
        instrument_file = "{}/{}".format(self.data_dir, self.config['stock_meta'])
        instrumen_config = json.load(open(instrument_file, "r"))
        self.instrumen_config = {i['instrument_token']: i for i in instrumen_config}

        if sort_by and sort_by not in sort_by_options:
            raise Exception("sort_by should be one of {}".format(sort_by_options))

        self.cols = list(set([i.split('_')[0] for i in df.columns if i != 'date']))
        self.col_volume_map = property_map(df, self.cols, sort_by, reverse)
        self.day_list = create_day_list(df)

    def craete_title(self, col):

        name = self.instrumen_config.get(int(col), {}).get('name', 'Unknown')

        title = "\n{} | {}".format(name, col)
        if self.sort_by:
            title += " | {}: {}".format(beautify_str(self.sort_by), self.col_volume_map[col][self.sort_by])
        title += "\n\n"

        for k, v in self.col_volume_map[col].items():
            if k != 'missing_values':
                title += "{}: {} | ".format(beautify_str(k), v)
        title = title.strip(' | ')
        return title

    def plot(self, day_separation=True, samples=None, name_start=None):

        sample_count = 0
        for i, col in enumerate(self.col_volume_map):

            name = self.instrumen_config.get(int(col), {}).get('name', 'Unknown')
            if name_start and not name.startswith(name_start):
                continue

            if samples and sample_count >= samples:
                break

            sample_count += 1
            fig, ax = plt.subplots(); fig.set_size_inches(18, 4.5); ax.tick_params(axis='y', colors='black')
            ax.plot(self.df['{}_avg'.format(col)].to_list())
            ax.set_title(self.craete_title(col), fontsize=16, y=1.025, color='black')

            if day_separation:
                # plot vertical lines for each beginning of the day
                for i in range(1, len(self.day_list)):
                    if self.day_list[i - 1] != self.day_list[i]:
                        ax.axvline(x=i, color='red', linestyle='--', linewidth=0.6)
                        ax.text(i, 0.02, self.day_list[i], transform=ax.get_xaxis_transform(), rotation=0, size=10, color='green')

            plt.show()


def help():

    """
    This functions prints the user manual of the visualizer
    """
    print("--------------------Class Parameters--------------------\n")
    print("sort_by             Sort by mean_vol, mean_price, mean_trade_value, price_std, vol_std, missing_values")
    print("reverse             Reverse the sort order")
    print("\n--------------------Plot Parameters--------------------\n")
    print("samples             Number of samples to plot")
    print("name_start          Plot only those stocks whose name starts with this string")
    print("day_separation      Plot vertical lines for each day")
