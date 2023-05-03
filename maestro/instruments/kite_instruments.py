# Class to get all trading instruments either from the kite API or from the json file saved in data folder
from maestro.connector.kite_connector import KiteConnector
import enum
import logging
import json

INSTRUMENTS_JSON_FILE = "/Users/akashanand/repo/maestro/data/instruments.json"


class InstrumentKeys(enum.Enum):
    EXCHANGE="exchange"
    EXCHANGE_TOKEN="exchange_token"
    INSTRUMENT_TOKEN="instrument_token"
    INSTRUMENT_TYPE="instrument_type"
    TRADING_SEGMENT="segment"
    TRADING_SYMBOL="tradingsymbol"
    INSTRUMENT_NAME="name"


class ExchangeType(enum.Enum):
    NSE="NSE"
    BSE="BSE"
    NFO="NFO"
    CDS="CDS"
    MCX="MCX"
    BCD="BCD"


class ExchangeSegment(enum.Enum):
    NFO_OPT="NFO-OPT"
    BSE="BSE"
    CDS_OPT="CDS-OPT"
    NSE="NSE"
    MCX_OPT="MCX-OPT"
    BCD_OPT="BCD-OPT"
    NFO_FUT="NFO-FUT"
    BCD="BCD"
    MCX_FUT="MCX-FUT"
    CDS_FUT="CDS-FUT"
    INDICES="INDICES"
    BCD_FUT="BCD-FUT"


class InstrumentType(enum.Enum):
    CE="CE"
    PE="PE"
    EQ="EQ"
    FUT="FUT"


class Instrument(object):
    def __init__(self, instrument):
        self.instrument_token=instrument["instrument_token"]
        self.exchange_token=instrument["exchange_token"]
        self.trading_symbol=instrument["tradingsymbol"]
        self.name=instrument["name"]
        self.last_price=instrument["last_price"]
        self.expiry=instrument["expiry"]
        self.expiry=instrument["expiry"]
        self.strike=instrument["strike"]
        self.tick_size=instrument["tick_size"]
        self.lot_size=instrument["lot_size"]
        self.instrument_type=instrument["instrument_type"]
        self.segment=instrument["segment"]
        self.exchange=instrument["exchange"]



class TradingInstruments(object):
    def __init__(self, kite_connector: KiteConnector, live: bool = False):
        self.kite_connector = kite_connector
        self.trader = kite_connector.trader
        self.api_key = kite_connector.api_key
        self.access_token = kite_connector.access_token
        if live:
            logging.debug("Going to fetch instruments from kite API")
            self.instruments = self._get_instruments_live()
            logging.debug("Fetched instruments from kite API")
        else:
            logging.debug("Going to fetch instruments from file")
            self.instruments = self._get_instruments_file()
            logging.debug("Fetched instruments from file")
        self.structured_instruments = [Instrument(instrument) for instrument in self.instruments]

    def _get_instruments_live(self):
        """
        :return: instruments
        """
        # Get the instruments
        instruments = self.trader.instruments()
        return instruments

    def _get_instruments_file(self):
        """
        :return: instruments
        """
        # Get the instruments
        with open(INSTRUMENTS_JSON_FILE, 'r') as infile:
            instruments = json.load(infile)
        return instruments

    def get_instruments(self):
        """
        :return: instruments
        """
        return self.instruments

    def get_instruments_by_symbol(self, symbol):
        """
        :param symbol: symbol
        :return: instrument
        """
        out = []
        for instrument in self.instruments:
            if instrument[InstrumentKeys.TRADING_SYMBOL.value] == symbol:
                out.append(instrument)
        return out

    def get_instrument_by_token(self, token: int):
        """
        :param token: token
        :return: instrument
        """
        for instrument in self.instruments:
            if instrument[InstrumentKeys.INSTRUMENT_TOKEN.value] == token:
                return instrument
        return None

    def get_instruments_by_name(self, name: str):
        """
        :param name: name
        :return: instrument
        """
        out = []
        for instrument in self.instruments:
            if instrument[InstrumentKeys.INSTRUMENT_NAME.value] == name:
                out.append(instrument)
        return out

    def get_instruments_by_exchange_token(self, exchange_token:str):
        """
        :param exchange_token: exchange token
        :return: instrument
        """
        out = []
        for instrument in self.instruments:
            if instrument[InstrumentKeys.EXCHANGE_TOKEN.value] == exchange_token:
                out.append(instrument)
        return out

    def get_instruments_by_exchange(self, exchange: ExchangeType, instrument_list: list = None):
        """
        :param exchange: exchange
        :return: instrument
        """
        out = []
        if instrument_list is None:
            instrument_list = self.instruments
        for instrument in instrument_list:
            if instrument[InstrumentKeys.EXCHANGE.value] == exchange.value:
                out.append(instrument)
        return out

    def find_instruments(self, substring: str):
        """
        :param substring: substring
        :return: instrument
        """
        out = []
        for instrument in self.instruments:
            if substring.lower() in instrument[InstrumentKeys.INSTRUMENT_NAME.value].lower():
                out.append(instrument)
            elif substring.lower() in instrument[InstrumentKeys.TRADING_SYMBOL.value].lower():
                out.append(instrument)
        return out