import numpy as np
import pandas as pd
import talib

from data_sources import get_resampled_prices_in_range


class Data:
    def __init__(self, start_time, end_time, transaction_currency, counter_currency, resample_period, horizon,
                 start_cash, start_crypto, source):
        self.start_time = start_time
        self.end_time = end_time
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.resample_period = resample_period
        self.horizon = horizon
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.source = source

        self.price_data = get_resampled_prices_in_range(start_time, end_time, transaction_currency, counter_currency, resample_period)
        self.rsi_data = talib.RSI(np.array(self.price_data.close_price, dtype=float), timeperiod=14)
        self.sma50_data = talib.SMA(np.array(self.price_data.close_price, dtype=float), timeperiod=50)
        self.ema50_data = talib.EMA(np.array(self.price_data.close_price, dtype=float), timeperiod=50)
        self.sma200_data = talib.SMA(np.array(self.price_data.close_price, dtype=float), timeperiod=200)
        self.ema200_data = talib.EMA(np.array(self.price_data.close_price, dtype=float), timeperiod=200)
        self.prices = self.price_data.as_matrix(columns=["close_price"])
        self.timestamps = pd.to_datetime(self.price_data.index.values, unit='s')
        assert len(self.prices) == len(self.timestamps)