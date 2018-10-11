import numpy as np
import pandas as pd
import talib
import logging
from dateutil import parser
from tick_provider import PriceDataframeTickProvider
from backtester_ticks import TickDrivenBacktester
from data_sources import get_resampled_prices_in_range, get_timestamp_n_ticks_earlier
from charting import time_series_chart


# temporarily suspend strategies logging warnings: buy&hold strategy triggers warnings
# as our buy has to be triggered AFTER the minimum strategy initialization period
# determined by the longest_function_history_size parameter of the used grammar
strategy_logger = logging.getLogger("strategies")
strategy_logger.setLevel(logging.ERROR)

TICKS_FOR_PRECOMPUTE = 200


class Data:

    def _parse_time(self, time_input):
        if isinstance(time_input, str):
            time_object = parser.parse(time_input)
            return time_object.timestamp()
        return time_input

    def __init__(self, start_time, end_time, transaction_currency, counter_currency, resample_period, start_cash,
                 start_crypto, source):
        self.start_time = self._parse_time(start_time)
        self.end_time = self._parse_time(end_time)
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.resample_period = resample_period
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.source = source

        self.precalc_start_time = get_timestamp_n_ticks_earlier(self.start_time, TICKS_FOR_PRECOMPUTE, transaction_currency,
                                                                counter_currency, source, resample_period)

        self.price_data = get_resampled_prices_in_range\
            (self.precalc_start_time, self.end_time, transaction_currency, counter_currency, resample_period)

        self.price_data = self.price_data[~self.price_data.index.duplicated(keep='first')]


        # do some sanity checks with data
        if not self.price_data.empty and self.price_data.iloc[0].name > self.start_time + 60*60*2:
            raise Exception(f"The retrieved price data for {transaction_currency}-{counter_currency} starts "
                            f"{(self.price_data.iloc[0].name - self.start_time)/60:.2f} minutes after "
                            f"the set start time!")

        if not self.price_data.empty and self.end_time - self.price_data.iloc[-1].name > 60*60*2:
            raise Exception(f"The retrieved price data for {transaction_currency}-{counter_currency} ends "
                            f"{(self.end_time - self.price_data.iloc[-1].name)/60:.2f} minutes before "
                            f"the set end time!")

        prices = np.array(self.price_data.close_price, dtype=float)
        self.rsi_data = talib.RSI(prices, timeperiod=14)[TICKS_FOR_PRECOMPUTE:]
        self.sma20_data = talib.SMA(prices, timeperiod=20)[TICKS_FOR_PRECOMPUTE:]
        self.ema20_data = talib.EMA(prices, timeperiod=20)[TICKS_FOR_PRECOMPUTE:]
        self.sma50_data = talib.SMA(prices, timeperiod=50)[TICKS_FOR_PRECOMPUTE:]
        self.ema50_data = talib.EMA(prices, timeperiod=50)[TICKS_FOR_PRECOMPUTE:]
        self.sma200_data = talib.SMA(prices, timeperiod=200)[TICKS_FOR_PRECOMPUTE:]
        self.ema200_data = talib.EMA(prices, timeperiod=200)[TICKS_FOR_PRECOMPUTE:]
        self.macd, self.macd_signal, self.macd_hist = talib.MACD(
            prices, fastperiod=12, slowperiod=26, signalperiod=9)
        self.macd = self.macd[TICKS_FOR_PRECOMPUTE:]
        self.macd_signal = self.macd_signal[TICKS_FOR_PRECOMPUTE:]
        self.macd_hist = self.macd_hist[TICKS_FOR_PRECOMPUTE:]
        self.adx = talib.ADX(np.array(self.price_data.high_price, dtype=float),
                             np.array(self.price_data.low_price, dtype=float),
                             np.array(self.price_data.close_price, dtype=float))[TICKS_FOR_PRECOMPUTE:]

        self.price_data = self.price_data.iloc[TICKS_FOR_PRECOMPUTE:]
        self.prices = self.price_data.as_matrix(columns=["close_price"])
        self.timestamps = pd.to_datetime(self.price_data.index.values, unit='s')
        assert len(self.prices) == len(self.timestamps)


    def to_dataframe(self):
        df = self.price_data.copy(deep=True)
        df['RSI'] = pd.Series(self.rsi_data, index=df.index)
        df['SMA20'] = pd.Series(self.sma20_data, index=df.index)
        df['SMA50'] = pd.Series(self.sma50_data, index=df.index)
        df['SMA200'] = pd.Series(self.sma200_data, index=df.index)
        df['EMA20'] = pd.Series(self.ema20_data, index=df.index)
        df['EMA50'] = pd.Series(self.ema50_data, index=df.index)
        df['EMA200'] = pd.Series(self.ema200_data, index=df.index)
        df['ADX'] = pd.Series(self.adx, index=df.index)


    def __str__(self):
        return f"{self.transaction_currency}-{self.counter_currency}-{int(self.start_time)}-{int(self.end_time)}"

    def build_buy_and_hold_benchmark(self):

        benchmark = TickDrivenBacktester.build_benchmark(
            transaction_currency=self.transaction_currency,
            counter_currency=self.counter_currency,
            start_cash=self.start_cash,
            start_crypto=self.start_crypto,
            start_time=self.start_time,
            end_time=self.end_time,
            source=self.source,
            tick_provider=PriceDataframeTickProvider(self.price_data)
        )
        return benchmark

    def _filter_fields(self, fields, individual_str):
        filtered_dict = {}
        for field in fields:
            if not field.lower() in individual_str and field != "Close price" and field != "MACD signal":
                continue
            if field == "MACD signal" and "macd" not in individual_str:
                continue
            filtered_dict[field] = fields[field]
        return filtered_dict


    def plot(self, orders=None, individual_str=None):
        timestamps = self.price_data.index
        data_primary_axis = {
            "Close price" : self.price_data.close_price,
            "SMA50": self.sma50_data,
            "EMA50": self.ema50_data,
            "SMA200": self.sma200_data,
            "EMA200": self.ema200_data,

        }

        data_secondary_axis = {
            "ADX": self.adx,
            "MACD": self.macd,
            "MACD signal": self.macd_signal,
            "RSI": self.rsi_data
        }

        if individual_str is not None:
            data_primary_axis = self._filter_fields(data_primary_axis, individual_str)
            data_secondary_axis = self._filter_fields(data_secondary_axis, individual_str)



        time_series_chart(timestamps, series_dict_primary=data_primary_axis, series_dict_secondary=data_secondary_axis,
                          title=f"{self.transaction_currency} - {self.counter_currency}", orders=orders)


