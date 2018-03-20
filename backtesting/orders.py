from enum import Enum
from utils import datetime_from_timestamp


class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"


class Order:
    def __init__(self, order_type, transaction_currency, counter_currency, timestamp, amount, unit_price):
        self.order_type = order_type
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.timestamp = timestamp
        self.amount = amount
        self.unit_price = unit_price

    def execute(self):
        if self.order_type == OrderType.BUY:
            return self.amount, -self.unit_price * self.amount
        elif self.order_type == OrderType.SELL:
            return -self.amount, self.unit_price * self.amount

    def __str__(self):
        delta_currency, delta_cash = self.execute()
        return "{0}  \t {1: <16} \t cash_balance -> {2:13.2f} {3} \t currency_balance -> {4:13.6f} {5} ". format(
            datetime_from_timestamp(self.timestamp),
            self.order_type,
            delta_cash,
            self.counter_currency,
            delta_currency,
            self.transaction_currency
        )
