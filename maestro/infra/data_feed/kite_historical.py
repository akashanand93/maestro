import backtrader as bt
from maestro.kite.connector.kite_connector import KiteConnector
from maestro.kite.historical.fetch_historical_data import HistoricalDataFetcher, Interval
import pytz
import datetime

import pandas as pd

import backtrader as bt
import datetime

default_from_date = (datetime.datetime.today() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
default_to_date = datetime.datetime.today().strftime("%Y-%m-%d")

class ZerodhaHistoricalData(bt.feeds.DataBase):
    params = (
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1),
        ('timeframe', bt.TimeFrame.Minutes),
        ('compression', 1),
        ('kite_connector', None),
        ('from_date', default_from_date),
        ('to_date', default_to_date),
        ('instrument_token', None),
    )

    def start(self):
        super(ZerodhaHistoricalData, self).start()
        if self.p.kite_connector is None:
            raise ValueError("data_fetcher cannot be None")
        if self.p.instrument_token is None:
            raise ValueError("instrument_token cannot be None")
        data_fetcher = HistoricalDataFetcher(self.p.kite_connector)
        if self.p.from_date is None:
            self.p.from_date = default_from_date
        if self.p.to_date is None:
            self.p.to_date = default_to_date
        data_json = data_fetcher.fetch_historical_data(
            instrument_token = self.p.instrument_token,
            from_date = self.p.from_date, 
            to_date = self.p.to_date, 
            interval = Interval.MINUTE)
        self.p.dataname = pd.DataFrame(data_json)
        self._idx = 0

    def _load(self):
        if self._idx >= len(self.p.dataname):
            return False

        row = self.p.dataname.iloc[self._idx]

        self.lines.datetime[0] = bt.date2num(row['date'].to_pydatetime())
        self.lines.open[0] = row['open']
        self.lines.high[0] = row['high']
        self.lines.low[0] = row['low']
        self.lines.close[0] = row['close']
        self.lines.volume[0] = row['volume']
        self.lines.openinterest[0] = 0.0

        self._idx += 1
        return True
