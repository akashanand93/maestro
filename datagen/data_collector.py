import os
import sys
import json
import time
import math
import tqdm
import argparse
from datetime import datetime

# get the parent to parent directry of this script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maestro.connector.kite_connector import KiteConnector
from maestro.historical.fetch_historical_data import HistoricalDataFetcher, Interval, get_today, get_date_before_today
from maestro.instruments.kite_instruments import TradingInstruments, ExchangeType

class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super(DateTimeEncoder, self).default(o)

def collect(raw_data, total_days, days_per_file):

    connector = KiteConnector()
    instruments = TradingInstruments(connector, live=True)
    nse_instruments = instruments.get_instruments_by_exchange(ExchangeType.NSE)
    print("Total NSE Instruments: {}".format(len(nse_instruments)))
    history_fetcher = HistoricalDataFetcher(connector)

    to_date = get_today()
    for i in range(days_per_file, total_days, days_per_file):

        from_date = get_date_before_today(i)
        flag = 0
        name = '{}_{}'.format(from_date[5:7], from_date[2:4])

        print("----------Generating data from {} to {}-----------\n".format(from_date, to_date))

        while 1:

            for instrument in tqdm.tqdm(nse_instruments):
                token = instrument["instrument_token"]
                file_name = '{}_{}.json'.format(token, name)
                if file_name in os.listdir(raw_data):
                    continue
                try:
                    data = history_fetcher.fetch_historical_data(token, from_date, to_date, Interval.MINUTE)
                    flag = 0
                except:
                    time.sleep(5)
                    print("Error, will run the loop again")
                    flag = 1
                    break
                json.dump(data, open("{}/{}".format(raw_data, file_name), 'w'), indent=4, cls=DateTimeEncoder)
            if flag == 1:
                continue
            break

        to_date = from_date

# create argparse
def argparser():
    parser = argparse.ArgumentParser(description='data collector')
    parser.add_argument('--config', help='configuration file', required=True)
    parser.add_argument('--total_days', help='total days', required=False, default=360)
    parser.add_argument('--days_per_file', help='days per file', required=False, default=60)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = argparser()
    config_file = args.config
    total_days = int(args.total_days)
    days_per_file = int(args.days_per_file)
    config = json.load(open(config_file, 'r'))
    data_dir = config['data_dir']
    raw_data = "{}/{}".format(data_dir, config['raw_data_dir'])
    collect(raw_data, total_days, days_per_file)



