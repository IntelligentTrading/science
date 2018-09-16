import numpy as np
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

DataKey = namedtuple('DataKey', 'transaction_currency counter_currency start_time end_time')

class TAProvider(FunctionProvider):

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return("TAprovider")

    def rsi(self, input):
        timestamp = input[0]
        timestamp_index = np.where(self.data.price_data.index == timestamp)[0]
        return self.data.rsi_data[timestamp_index]


    def sma50(self, input):
        timestamp = input[0]
        timestamp_index = np.where(self.data.price_data.index == timestamp)[0]
        return self.data.sma50_data[timestamp_index]


    def ema50(self, input):
        timestamp = input[0]
        timestamp_index = np.where(self.data.price_data.index == timestamp)[0]
        return self.data.ema50_data[timestamp_index]


    def sma200(self, input):
        timestamp = input[0]
        timestamp_index = np.where(self.data.price_data.index == timestamp)[0]
        return self.data.sma200_data[timestamp_index]


    def ema200(self, input):
        timestamp = input[0]
        timestamp_index = np.where(self.data.price_data.index == timestamp)[0]
        return self.data.ema200_data[timestamp_index]


    def price(self, input):
        timestamp = input[0]
        try:
            return self.data.price_data.loc[timestamp,"close_price"]
        except:
            print('OMG!!!')


class TAProviderCollection(TAProvider):
    # TODO: figure out how to write this shorter and DRY

    def __init__(self, data_collection):
        self.providers = {DataKey(data.transaction_currency, data.counter_currency,
                                  data.start_time, data.end_time) : TAProvider(data) for data in data_collection}

    def __str__(self):
        return("TAproviderCollection")


    def get_provider(self, timestamp, transaction_currency, counter_currency):
        for key in self.providers.keys():
            if key.transaction_currency == transaction_currency \
                    and key.counter_currency == counter_currency \
                    and key.end_time >= timestamp >= key.start_time:
                return self.providers[key]
        raise Exception(f'No data loaded for {transaction_currency} {counter_currency} at {timestamp}!')


    def rsi(self, input):
        timestamp, transaction_currency, counter_currency = input
        return self.get_provider(timestamp, transaction_currency, counter_currency).rsi([timestamp])


    def sma50(self, input):
        timestamp, transaction_currency, counter_currency = input
        return self.get_provider(timestamp, transaction_currency, counter_currency).sma50([timestamp])


    def ema50(self, input):
        timestamp, transaction_currency, counter_currency = input
        return self.get_provider(timestamp, transaction_currency, counter_currency).ema50([timestamp])


    def sma200(self, input):
        timestamp, transaction_currency, counter_currency = input
        return self.get_provider(timestamp, transaction_currency, counter_currency).sma200([timestamp])


    def ema200(self, input):
        timestamp, transaction_currency, counter_currency = input
        return self.get_provider(timestamp, transaction_currency, counter_currency).ema200([timestamp])


    def price(self, input):
        timestamp, transaction_currency, counter_currency = input
        return self.get_provider(timestamp, transaction_currency, counter_currency).price([timestamp])