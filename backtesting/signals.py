from utils import datetime_from_timestamp

class Signal:
    def __init__(self, signal_type, trend, horizon, strength_value, strength_max,
                 price, price_change, timestamp, rsi_value, transaction_currency, counter_currency):
        self.signal_type = signal_type
        self.trend = trend
        self.horizon = horizon
        self.strength_value = strength_value
        self.strength_max = strength_max
        self.price = price
        self.price_change = price_change
        self.timestamp = timestamp
        self.rsi_value = rsi_value
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency

    def __str__(self):
        return ("{} trend={} horizon={} timestamp={}".format(self.signal_type, self.trend,
                                                             self.horizon, datetime_from_timestamp(self.timestamp)))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__









