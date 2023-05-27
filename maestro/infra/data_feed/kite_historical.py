import backtrader as bt
from maestro.kite.connector.kite_connector import KiteConnector
from maestro.kite.historical.fetch_historical_data import HistoricalDataFetcher, Interval
import pytz
import datetime

class ZerodhaData(bt.feeds.DataBase):
    params = (
        ("token", ""),
        ("fromdate", datetime.datetime.utcnow()),
        ("todate", datetime.datetime.utcnow()),
        ("interval", "5minute"),
        ("timezone", pytz.UTC), # Zerodha data is in IST
        ("compression", 1)
    )
    
    def start(self):
        self.kiteconnector = KiteConnector()
        self.historical_data_fetcher = HistoricalDataFetcher(self.kiteconnector)
        
    
    def _load(self):
        if self.p.historical:
            data = self.kite.ltp(self.p.token)
            # Add data processing code here
        else:
            data = self.kite.ltp(self.p.token)
            # Add data processing code here
        return True
