import numpy as np


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
        return self.data.price_data.loc[timestamp,"close_price"]


class TAProviderCollection(TAProvider):
    # TODO: figure out how to write this shorter and DRY

    def __init__(self, data_collection):
        self.providers = {(data.transaction_currency, data.counter_currency) : TAProvider(data) for data in data_collection}
        # TODO: ensure that same currency pairs can be used for training and validation

    def __str__(self):
        return("TAproviderCollection")

    def rsi(self, input):
        timestamp, transaction_currency, counter_currency = input
        return self.providers[(transaction_currency, counter_currency)].rsi([timestamp])


    def sma50(self, input):
        timestamp, transaction_currency, counter_currency = input
        return self.providers[(transaction_currency, counter_currency)].sma50([timestamp])


    def ema50(self, input):
        timestamp, transaction_currency, counter_currency = input
        return self.providers[(transaction_currency, counter_currency)].ema50([timestamp])

    def sma200(self, input):
        timestamp, transaction_currency, counter_currency = input
        return self.providers[(transaction_currency, counter_currency)].sma200([timestamp])


    def ema200(self, input):
        timestamp, transaction_currency, counter_currency = input
        return self.providers[(transaction_currency, counter_currency)].ema200([timestamp])


    def price(self, input):
        timestamp, transaction_currency, counter_currency = input
        return self.providers[(transaction_currency, counter_currency)].price([timestamp])