import numpy as np
import pandas as pd
import talib
from dateutil import parser
from tick_provider import PriceDataframeTickProvider
from backtester_ticks import TickDrivenBacktester
from data_sources import get_resampled_prices_in_range
import logging

# temporarily suspend strategies logging warnings: buy&hold strategy triggers warnings
# as our buy has to be triggered AFTER the minimum strategy initialization period
# determined by the longest_function_history_size parameter of the used grammar
strategy_logger = logging.getLogger("strategies")
strategy_logger.setLevel(logging.ERROR)


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

        self.price_data = get_resampled_prices_in_range(self.start_time, self.end_time, transaction_currency, counter_currency, resample_period)
        # do some sanity checks with data
        if not self.price_data.empty and self.price_data.iloc[0].name > self.start_time + 60*60*2:
            raise Exception(f"The retrieved price data starts "
                            f"{(self.price_data.iloc[0].name - self.start_time)/60:.2f} minutes after "
                            f"the set start time!")

        if not self.price_data.empty and self.end_time - self.price_data.iloc[-1].name > 60*60*2:
            raise Exception(f"The retrieved price data ends "
                            f"{(self.end_time - self.price_data.iloc[-1].name)/60:.2f} minutes before "
                            f"the set end time!")

        self.rsi_data = talib.RSI(np.array(self.price_data.close_price, dtype=float), timeperiod=14)
        self.sma50_data = talib.SMA(np.array(self.price_data.close_price, dtype=float), timeperiod=50)
        self.ema50_data = talib.EMA(np.array(self.price_data.close_price, dtype=float), timeperiod=50)
        self.sma200_data = talib.SMA(np.array(self.price_data.close_price, dtype=float), timeperiod=200)
        self.ema200_data = talib.EMA(np.array(self.price_data.close_price, dtype=float), timeperiod=200)
        self.prices = self.price_data.as_matrix(columns=["close_price"])
        self.timestamps = pd.to_datetime(self.price_data.index.values, unit='s')
        assert len(self.prices) == len(self.timestamps)


    def to_dataframe(self):
        df = self.price_data.copy(deep=True)
        df['RSI'] = pd.Series(self.rsi_data, index=df.index)
        df['SMA50'] = pd.Series(self.sma50_data, index=df.index)
        df['SMA200'] = pd.Series(self.sma200_data, index=df.index)
        df['EMA50'] = pd.Series(self.ema50_data, index=df.index)
        df['EMA200'] = pd.Series(self.ema200_data, index=df.index)
        return df

    def __str__(self):
        return f"{self.transaction_currency}-{self.counter_currency}-{int(self.start_time)}-{int(self.end_time)}"

    def build_buy_and_hold_benchmark(self, num_ticks_to_skip):

        # need to skip first num_ticks_to_skip, as strategies don't generate signals before all
        # the historical data is available to them
        benchmark_start_time = self.price_data.iloc[num_ticks_to_skip].name

        benchmark = TickDrivenBacktester.build_benchmark(
            transaction_currency=self.transaction_currency,
            counter_currency=self.counter_currency,
            start_cash=self.start_cash,
            start_crypto=self.start_crypto,
            start_time=benchmark_start_time,
            end_time=self.end_time,
            source=self.source,
            tick_provider=PriceDataframeTickProvider(self.price_data)
        )
        return benchmark
