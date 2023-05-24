import datetime
import backtrader.feeds as btfeed
import datetime as dt
import backtrader as bt

class CustomDataFeed(btfeed.GenericCSVData):

  params = (
    ('nullvalue', 0.0),
    ('dtformat', ('%Y-%m-%d %H:%M:%S')),
    ('timeframe',bt.TimeFrame.Minutes),
    ('datetime', 0),
    ('high', 2),
    ('low', 3),
    ('open', 1),
    ('close', 4),
    ('volume', 5),
    ('openinterest', -1)
  )
