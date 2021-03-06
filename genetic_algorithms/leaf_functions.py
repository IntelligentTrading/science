import numpy as np
import inspect
from collections import namedtuple


class FunctionProvider:

    def if_then_else(self, input, output1, output2):
        try:
            return output1 if input else output2
        except:
            return output1

    @classmethod
    def buy(self):
        pass

    @classmethod
    def sell(self):
        pass

    @classmethod
    def ignore(self):
        pass

    def identity(self, x):
        return x


DataKey = namedtuple('DataKey', 'source resample_period transaction_currency counter_currency start_time end_time')


class TAProvider(FunctionProvider):

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return("TAprovider")

    def _get_timestamp_index(self, input):
        assert self.data.price_data.index.get_loc(input[0]) == np.where(self.data.price_data.index == input[0])[0][0]
        return self.data.price_data.index.get_loc(input[0])

    def rsi(self, input):
        return self.data.rsi_data[self._get_timestamp_index(input)]

    def rsi_lt_20(self, input):
        return self.data.rsi_data[self._get_timestamp_index(input)] < 20

    def rsi_lt_25(self, input):
        return self.data.rsi_data[self._get_timestamp_index(input)] < 25

    def rsi_lt_30(self, input):
        return self.data.rsi_data[self._get_timestamp_index(input)] < 30

    def rsi_gt_70(self, input):
        return self.data.rsi_data[self._get_timestamp_index(input)] > 70

    def rsi_gt_75(self, input):
        return self.data.rsi_data[self._get_timestamp_index(input)] > 75

    def rsi_gt_80(self, input):
        return self.data.rsi_data[self._get_timestamp_index(input)] > 80

    def macd_bullish(self, input):
        return self._crosses_from_below(self.data.macd, self.data.macd_signal, input)
        #index = self._get_timestamp_index(input)
        #if index == 0:
        #    return False
        #return self.data.macd[index] > self.data.macd_signal[index] \
        #       and self.data.macd[index-1] <= self.data.macd_signal[index-1]

    def macd_bearish(self, input):
        return self._crosses_from_above(self.data.macd, self.data.macd_signal, input)
        # index = self._get_timestamp_index(input)
        # if index == 0:
        #     return False
        # return self.data.macd[index] < self.data.macd_signal[index] \
        #       and self.data.macd[index-1] >= self.data.macd_signal[index-1]

    def ema_bullish_cross(self, input):
        return self._crosses_from_above(self.data.ema55_data, self.data.ema21_data, input)

    def ema_bearish_cross(self, input):
        return self._crosses_from_below(self.data.ema55_data, self.data.ema21_data, input)

    def bbands_bearish_cross(self, input):
        return self._crosses_from_below(self.data.prices, self.data.bb_up, input)

    def bbands_bullish_cross(self, input):
        return self._crosses_from_above(self.data.prices, self.data.bb_low, input)

    def bbands_squeeze_bullish(self, input):
        index = self._get_timestamp_index(input)
        return self.data.bb_width[index] <= self.data.min_bbw_180[index] \
               and self._crosses_from_below(self.data.prices, self.data.bb_up, input)

    def bbands_squeeze_bearish(self, input):
        index = self._get_timestamp_index(input)
        return self.data.bb_width[index] <= self.data.min_bbw_180[index] \
               and self._crosses_from_above(self.data.prices, self.data.bb_low, input)

    def bbands_price_gt_up(self, input):
        index = self._get_timestamp_index(input)
        return self.data.prices[index] > self.data.bb_up[index]

    def bbands_price_lt_low(self, input):
        index = self._get_timestamp_index(input)
        return self.data.prices[index] < self.data.bb_low[index]

    def slowd_gt_80(self, input):
        return self.data.slowd[self._get_timestamp_index(input)] > 80

    def slowd_lt_20(self, input):
        return self.data.slowd[self._get_timestamp_index(input)] < 20

    def candlestick_momentum_buy(self, input):
        index = self._get_timestamp_index(input)
        if index == 0:
            return False
        return self.data.prices[index] > 0.5 * self.data.prices[index-1]

    def candlestick_momentum_sell(self, input):
        index = self._get_timestamp_index(input)
        if index == 0:
            return False
        return self.data.prices[index] < 0.5 * self.data.prices[index-1]

    def _crosses_from_below(self, data, other, input):
        index = self._get_timestamp_index(input)
        if index == 0:
            return False
        return data[index] > other[index] and data[index-1] <= other[index-1]

    def _crosses_from_above(self, data, other, input):
        index = self._get_timestamp_index(input)
        if index == 0:
            return False
        return data[index] < other[index] and data[index-1] >= other[index-1]

    def adx(self, input):
        return self.data.adx[self._get_timestamp_index(input)]

    def sma20(self, input):
        return self.data.sma20_data[self._get_timestamp_index(input)]

    def ema20(self, input):
        return self.data.ema20_data[self._get_timestamp_index(input)]

    def sma50(self, input):
        return self.data.sma50_data[self._get_timestamp_index(input)]

    def ema50(self, input):
        return self.data.ema50_data[self._get_timestamp_index(input)]

    def sma200(self, input):
        return self.data.sma200_data[self._get_timestamp_index(input)]

    def ema200(self, input):
        return self.data.ema200_data[self._get_timestamp_index(input)]

    def price(self, input):
        timestamp = input[0]
        return self.data.price_data.loc[timestamp, "close_price"]

    def volume(self, input):
        return self.data.close_volume_data[self._get_timestamp_index(input)]

    def sma50_volume(self, input):
        return self.data.sma50_volume_data[self._get_timestamp_index(input)]

    def volume_cross_up(self, input):
        return self._crosses_from_below(self.data.close_volume_data, self.data.sma50_volume_data, input)

    def volume_cross_down(self, input):
        return self._crosses_from_above(self.data.close_volume_data, self.data.sma50_volume_data, input)

    def macd_stoch_sell(self, input):
        index = self._get_timestamp_index(input)
        return self.data.macd_hist[index] < 0 and self.data.slowd[index] > 71

    def macd_stoch_buy(self, input):
        index = self._get_timestamp_index(input)
        return self.data.macd_hist[index] > 0 and self.data.slowd[index] < 29



class TAProviderCollection(FunctionProvider):

    def __init__(self, data_collection):
        self.providers = {DataKey(data.source, data.resample_period, data.transaction_currency, data.counter_currency,
                                  data.start_time, data.end_time): TAProvider(data) for data in data_collection}
        # create methods
        members = inspect.getmembers(TAProvider, predicate=inspect.isfunction)
        base_members = inspect.getmembers(FunctionProvider)
        for member in members:
            if member in base_members:
                continue

            function_name = member[0]
            if function_name.startswith('__'):
                continue

            setattr(TAProviderCollection, function_name, self._create_function(function_name))


    def _create_function(self, function_name):
        exec(f'''
def {function_name}(self, input):
    timestamp, transaction_currency, counter_currency = input
    provider = self.get_provider(timestamp, transaction_currency, counter_currency)
    return provider.{function_name}([timestamp])
''')
        return locals()[f'{function_name}']

    def get_provider(self, timestamp, transaction_currency, counter_currency):
        for key in self.providers.keys():
            if key.transaction_currency == transaction_currency \
                    and key.counter_currency == counter_currency \
                    and key.end_time >= timestamp >= key.start_time:
                return self.providers[key]
        raise Exception(f'No data loaded for {transaction_currency} - {counter_currency} at {timestamp}!')

    def __str__(self):
        return ','.join([f'{key.transaction_currency}-{key.counter_currency}-{key.start_time}-{key.end_time}'
                         for key in self.providers.keys()])


