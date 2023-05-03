import torch
from torch.utils.data import Dataset, DataLoader
import logging
import datetime

class HistoricalData(Dataset):
    def __init__(self, data_directory, sequence_length_in_minutes=120, predict_after_in_minutes=5):
        import pickle
        with open(data_directory, "rb") as f:
            data = pickle.load(f)
            self.data = data
        self.sequence_length_in_minutes = sequence_length_in_minutes
        self.instruments = list(self.data.keys())
        self.num_instruments = len(self.instruments)
        self.predict_after_in_minutes = predict_after_in_minutes
        self.instrument_index = {}
        for i, instrument in enumerate(self.instruments):
            self.instrument_index[instrument] = i
        logging.info("Loaded data for {} instruments".format(self.num_instruments))
        logging.info("Starting to arrange data")
        self._arrange_data()
        logging.info("Finished arranging data")
        logging.info("Number of date data points: {}".format(len(self.data)))

    def _create_day_end_data(self, date):
        return {
            'date': date,
            'is_day_end': 1,
            'open': [0.0 for i in range(self.num_instruments)], 
            'high': [0.0 for i in range(self.num_instruments)], 
            'low': [0.0 for i in range(self.num_instruments)], 
            'close': [0.0 for i in range(self.num_instruments)], 
            'volume': [0.0 for i in range(self.num_instruments)]
            }
    
    def _create_dummy_arranged_data(self):
        all_dates = set()
        for instrument in self.data.keys():
            for element in self.data[instrument]:
                date_obj = element["date"]
                all_dates.add(date_obj)
        all_dates = sorted(list(all_dates))
        all_dates_with_day_end = []
        
        for idx, date in enumerate(all_dates):
            if idx == 0:
                last_date = date
            if date.day != last_date.day:
                all_dates_with_day_end.append(last_date + datetime.timedelta(minutes=1))
                is_day_end.append(1)
            all_dates_with_day_end.append(date)
            is_day_end.append(0)
            last_date = date
        return all_dates_with_day_end, is_day_end


    def _arrange_data(self):
        # Arrange the data date wise so that we can use it for time series analysis
        # Originally self.data is dictionary of instruments and the values is an array of elements
        # Each element is a dictionary with keys: date, open, high, low, close, volume
        # First arrange all the instruments date wise
        date_wise_data = []
        all_dates = {}
        for instrument in self.data.keys():
            for element in self.data[instrument]:
                date_obj = element["date"]
                date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                if date not in all_dates:
                    date_wise_data.append({
                        'date': date_obj,
                        'is_day_end': 0,
                        'open': [0.0 for i in range(self.num_instruments)], 
                        'high': [0.0 for i in range(self.num_instruments)], 
                        'low': [0.0 for i in range(self.num_instruments)], 
                        'close': [0.0 for i in range(self.num_instruments)], 
                        'volume': [0.0 for i in range(self.num_instruments)]
                        })
                    all_dates[date] = len(date_wise_data) - 1
                idx = all_dates[date]
                date_wise_data[idx]['open'][self.instrument_index[instrument]] = element['open']
                date_wise_data[idx]['high'][self.instrument_index[instrument]] = element['high']
                date_wise_data[idx]['low'][self.instrument_index[instrument]] = element['low']
                date_wise_data[idx]['close'][self.instrument_index[instrument]] = element['close']
                date_wise_data[idx]['volume'][self.instrument_index[instrument]] = element['volume']

        # Now we have each instrument data in date_wise_data. Lets create an array of all data sorted by dates
        date_wise_data = sorted(date_wise_data, key=lambda x: x['date'])
        # Now insert day end data
        for i in range(len(date_wise_data)):
            if i == 0:
                last_date = date_wise_data[i]['date']
            if date_wise_data[i]['date'].day != last_date.day:
                date_wise_data.insert(i, self._create_day_end_data(last_date))
                last_date = date_wise_data[i]['date']
        self.data = date_wise_data
        

    
    def __len__(self):
        return len(self.data) - self.predict_after_in_minutes - self.sequence_length_in_minutes
    
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.sequence_length_in_minutes
        if end_idx > (len(self.data) - self.predict_after_in_minutes):
            end_idx = len(self.data)
        # Get the temp data
        temp_data = self.data[start_idx:end_idx]

