import backtrader as bt

class BacktestOrchestrator:
    def __init__(self, strategy:bt.Strategy, data_feed, commission, initial_cash):
        self.strategy = strategy
        self.data_feed = data_feed
        self.commission = commission
        self.initial_cash = initial_cash

    def run(self):
        # Create a cerebro entity
        cerebro = bt.Cerebro()
        # Add data feed
        cerebro.adddata(self.data_feed)
        # Add strategy
        cerebro.addstrategy(self.strategy)
        # Set initial cash
        cerebro.broker.setcash(self.initial_cash)
        # Add commission
        cerebro.broker.addcommissioninfo(self.commission)
        # Print out the starting conditions
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
        # Run over everything
        cerebro.run()
        # Print out the final result
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())