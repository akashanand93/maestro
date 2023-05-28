from maestro.infra.orchestrator.backtest_orchestrator import BacktestOrchestrator
from maestro.infra.data_feed.kite_historical import ZerodhaHistoricalData
from maestro.infra.data_feed.custom_csv import CustomDataFeed
from maestro.infra.commission.kite_platform import ZerodhaCommission
from maestro.infra.strategies.dummy_strategy import DummyStrategy


def get_backtest_orchestrator(config, connector):
    initial_cash = config['initial_cash']
    data_feed = get_data_feed(config['data_feed'], connector=connector)
    commission = get_commission(config['commission'])
    startegy = get_strategy(config['strategy'])
    return BacktestOrchestrator(strategy=startegy,
                                data_feed=data_feed,
                                commission=commission,
                                initial_cash=initial_cash)

def get_data_feed(config, connector=None):
    permissible_data_feed_types = ['kite_historical', 'csv_file']
    instrument_token = config['instrument_token']
    data_feed_type = config['type']
    if data_feed_type == 'kite_historical':
        start_date = config['start_date']
        end_date = config['end_date']
        if start_date == "":
            start_date = None
        if end_date == "":
            end_date = None
        return ZerodhaHistoricalData(kite_connector = connector,
                                     instrument_token = instrument_token,
                                     from_date = start_date,
                                     to_date = end_date)
    elif data_feed_type == 'csv_file':
        return CustomDataFeed(config['file_path'], config['name'])
    else:
        raise ValueError("Invalid data feed type: {}, Allowed types: ".format(data_feed_type, 
                                                                              permissible_data_feed_types))
    
def get_commission(config):
    permissible_commission_types = ['kite_platform']
    commission_type = config['type']
    if commission_type == 'kite_platform':
        return ZerodhaCommission()
    else:
        raise ValueError("Invalid commission type: {}, Allowed types: ".format(commission_type, 
                                                                                permissible_commission_types))
    
def get_strategy(config):
    permissible_strategy_types = ['dummy']
    strategy_type = config['type']
    if strategy_type == 'dummy':
        return DummyStrategy()
    else:
        raise ValueError("Invalid strategy type: {}, Allowed types: ".format(strategy_type, 
                                                                              permissible_strategy_types))
    

