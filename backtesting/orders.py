from enum import Enum
from utils import datetime_from_timestamp


class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"


class Order:
    def __init__(self, order_type, transaction_currency, counter_currency, timestamp, value, unit_price,
                 transaction_cost_percent):
        self.order_type = order_type
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.timestamp = timestamp
        self.value = value
        self.unit_price = unit_price
        self.transaction_cost_percent = transaction_cost_percent

    def execute(self):
        if self.order_type == OrderType.BUY:
            return (self.value * (1 - self.transaction_cost_percent)) / self.unit_price, -self.value
        elif self.order_type == OrderType.SELL:
            return -self.value, (self.value * self.unit_price) * (1 - self.transaction_cost_percent)

    def __str__(self):
        delta_currency, delta_cash = self.execute()
        return "{0}  \t {1: <16} \t cash_balance -> {2:13.2f} {3} \t currency_balance -> {4:13.6f} {5} \t (1 {6} = {7:.8f} {8})". format(
            datetime_from_timestamp(self.timestamp),
            self.order_type,
            delta_cash,
            self.counter_currency,
            delta_currency,
            self.transaction_currency,
            self.transaction_currency,
            self.unit_price,
            self.counter_currency
        )


class BuyOrder:
    def __init__(self, transaction_currency, counter_currency, timestamp, cash, unit_price, transaction_cost):
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.timestamp = timestamp
        self.cash = cash
        self.unit_price = unit_price
        self.transaction_cost_percent = transaction_cost

    def execute(self):
        bought_crypto = (self.cash * (1-self.transaction_cost_percent)) / self.unit_price
        delta_cash = -self.cash
        return bought_crypto, delta_cash

    def __str__(self):
        delta_currency, delta_cash = self.execute()
        return "{0}  \t {1: <16} \t cash_balance -> {2:13.2f} {3} \t currency_balance -> {4:13.6f} {5} \t (1 {6} = {7:.8f} {8})". format(
            datetime_from_timestamp(self.timestamp),
            "BUY",
            delta_cash,
            self.counter_currency,
            delta_currency,
            self.transaction_currency,
            self.transaction_currency,
            self.unit_price,
            self.counter_currency
        )


class SellOrder:
    def __init__(self, transaction_currency, counter_currency, timestamp, crypto, unit_price, transaction_cost):
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.timestamp = timestamp
        self.crypto = crypto
        self.unit_price = unit_price
        self.transaction_cost_percent = transaction_cost

    def execute(self):
        delta_cash = (self.crypto * self.unit_price) * (1-self.transaction_cost_percent)
        delta_crypto = -self.crypto
        return delta_crypto, delta_cash

    def __str__(self):
        delta_currency, delta_cash = self.execute()
        return "{0}  \t {1: <16} \t cash_balance -> {2:13.2f} {3} \t currency_balance -> {4:13.6f} {5} \t (1 {6} = {7:.8f} {8})". format(
            datetime_from_timestamp(self.timestamp),
            "SELL",
            delta_cash,
            self.counter_currency,
            delta_currency,
            self.transaction_currency,
            self.transaction_currency,
            self.unit_price,
            self.counter_currency
        )
