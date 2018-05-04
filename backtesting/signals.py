from enum import Enum

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

    @staticmethod
    def get_signal_name(signal_type, strength_value):
        if signal_type == SignalType.SMA:
            signal_name = "SMA "
            if strength_value == 1:
                signal_name += "short"
            elif strength_value == 2:
                signal_name += "medium"
            elif strength_value == 3:
                signal_name += "long"
            else:
                signal_name += "any"
        else:
            signal_name = signal_type.value
        return signal_name

    def __str__(self):
        return ("{} trend={} horizon={} timestamp={} rsi_value={}".format(Signal.get_signal_name(self.signal_type, self.strength_value),
                                                             self.trend, self.horizon, datetime_from_timestamp(self.timestamp), self.rsi_value))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class SignalType(Enum):
    RSI = "RSI"
    kumo_breakout = "kumo_breakout"
    SMA = "SMA"
    EMA = "EMA"
    RSI_Cumulative = "RSI_cumulative"