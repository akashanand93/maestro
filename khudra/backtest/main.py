from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from dotenv import load_dotenv

load_dotenv()


from maestro.infra.data_feed.custom_csv import CustomDataFeed
from maestro.infra.strategies.dummy_strategy import DummyStrategy
from maestro.infra.commission.kite_platform import ZerodhaCommission
import datetime

# Import the backtrader platform
import backtrader as bt

if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    cerebro.addstrategy(DummyStrategy)

    # Create a Data Feed
    data = CustomDataFeed(
        dataname="/Users/akashanand/repo/marketmaestro/maestro/data/backtrader/1675521.csv", 
        name="1675521")

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Add the custom commission calculation
    cerebro.broker.addcommissioninfo(ZerodhaCommission())

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())