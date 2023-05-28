from maestro.kite.connector.kite_connector import KiteConnector
import datetime
import enum
import logging

def get_today():
    return datetime.datetime.today().strftime("%Y-%m-%d")

def get_date_before_today(days: int):
    return (datetime.datetime.today() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")

class Interval(enum.Enum):
    MINUTE = "minute"
    THREE_MINUTE = "3minute"
    FIVE_MINUTE = "5minute"
    TEN_MINUTE = "10minute"
    FIFTEEN_MINUTE = "15minute"
    THIRTY_MINUTE = "30minute"
    SIXTY_MINUTE = "60minute"
    DAY = "day"


# The maximum duration for each interval
"""
The kite historical API allows only the following intervals for the correspoding intervals
            Minute : 60 days
            3minute : 100 days
            5minute : 100 days
            10minute : 100 days
            15minute : 200 days
            30minute : 200 days
            60minute : 400 days
            Day : 2000 days
"""
MAX_DURATION_MAPPING = {
    Interval.MINUTE.value: 60,
    Interval.THREE_MINUTE.value: 100,
    Interval.FIVE_MINUTE.value: 100,
    Interval.TEN_MINUTE.value: 100,
    Interval.FIFTEEN_MINUTE.value: 200,
    Interval.THIRTY_MINUTE.value: 200,
    Interval.SIXTY_MINUTE.value: 400,
    Interval.DAY.value: 2000
}


# Create a class to fetch historical data
# The class will use KiteConnector to access the kiteconnect object along with the API_KEY and ACCESS_TOKEN

class HistoricalDataFetcher(object):
    def __init__(self, kite_connector) -> None:
        self.kite_connector = kite_connector
        self.trader = kite_connector.trader
        self.api_key = kite_connector.api_key
        self.access_token = kite_connector.access_token


    def _check_date_validity(self, date):
        """
        :param date: date
        :return: True if valid, False otherwise
        """
        # Add some checks. The date format should be YYYY-MM-DD
        try:
            datetime.datetime.strptime(date, '%Y-%m-%d')
            return True
        except ValueError:
            return False


    def _check_duration_validity(self, from_date: str, to_date: str, interval: Interval):
        """
        Function to check the duration validity
        """
        logging.debug("To date: {} and from date {}".format(to_date, from_date))
        from_date = datetime.datetime.strptime(from_date, '%Y-%m-%d')
        to_date = datetime.datetime.strptime(to_date, '%Y-%m-%d')
        # Calculate the duration in days
        duration = (to_date - from_date).days
        if duration > MAX_DURATION_MAPPING[interval.value]:
            return False, duration
        if duration <= 0:
            return False, duration
        return True, duration


    def fetch_historical_data(self, instrument_token, from_date: str, to_date: str, interval: Interval):
        """
        :param instrument_token: instrument token
        :param from_date: from date
        :param to_date: to date
        :param interval: interval
        :return: historical data
        """
        # Add some checks.
        if not self._check_date_validity(from_date):
            raise ValueError("Invalid from date")
        if not self._check_date_validity(to_date):
            raise ValueError("Invalid to date") 
        is_duration_valid, duration = self._check_duration_validity(from_date, to_date, interval)
        if not is_duration_valid:
            raise ValueError("Invalid duration. Permissible {} actual {}".format(MAX_DURATION_MAPPING[interval.value], duration))

        data = self.trader.historical_data(instrument_token, from_date, to_date, interval.value)
        return data