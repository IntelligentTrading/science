import mysql.connector
from signals import RSISignal
from config import database_config
from utils import datetime_from_timestamp, MAX_TIME
from enum import Enum


class CounterCurrency(Enum):
    BTC = 0
    ETH = 1
    USDT = 2
    XMR = 3

#(BTC, ETH, USDT, XMR) = list(range(4))

signal_query = """ SELECT trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value 
            FROM signal_signal 
            WHERE   signal_signal.signal=%s AND 
                    transaction_currency=%s AND 
                    counter_currency=%s AND
                    timestamp >= %s AND
                    timestamp <= %s;"""

most_recent_price_query = """SELECT price FROM indicator_price 
                            WHERE timestamp = 
                                (SELECT MAX(timestamp) 
                                FROM indicator_price 
                                FROM indicator_price 
                                WHERE transaction_currency = "%s") 
                            AND transaction_currency = "%s"
                            AND counter_currency = %s;"""

price_query = """SELECT price FROM indicator_price 
                            WHERE transaction_currency = %s
                            AND timestamp = %s
                            AND counter_currency = %s;"""


def get_rsi_signals(transaction_currency, start_time=0, end_time=MAX_TIME, counter_currency="BTC"):
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    cursor.execute(signal_query, params=("RSI", transaction_currency, counter_currency_id, start_time, end_time))
    signals = []
    for (trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value) in cursor:
        # TODO: fill this field properly
        price_USDT = None # get_price_USDT(price, timestamp, transaction_currency, counter_currency)
        signals.append(RSISignal(trend, horizon, strength_value, strength_max,
                                 price/1E8, price_USDT, price_change, timestamp, rsi_value, transaction_currency, counter_currency))
    return signals


def get_value_of_currency(currency, timestamp, counter_currency="BTC", normalize=False):
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    cursor.execute(price_query, params=(currency, timestamp, counter_currency_id))
    price = cursor.fetchall()
    if cursor.rowcount == 0:
        return None

    assert cursor.rowcount == 1
    price = price[0][0]
    if normalize:
        return price / 1E8
    else:
        return price


def BTC_to_USDT(timestamp):
    return get_value_of_currency("BTC", timestamp, "USDT")


def get_value_of_currency_in_USDT(currency, timestamp):
    value_in_USDT = get_value_of_currency(currency, timestamp, "USDT")
    if value_in_USDT == None:
        value_in_BTC = get_value_of_currency(currency, timestamp, "BTC")
        if value_in_BTC == None:
            return None
        value_of_BTC_in_USDT = BTC_to_USDT(timestamp)
        if value_of_BTC_in_USDT == None:
            return None
        return value_in_BTC * value_of_BTC_in_USDT
    else:
        return value_in_USDT


def get_price_USDT(price, timestamp, transaction_currency, counter_currency):
    if counter_currency == "USDT":
        return price
    elif counter_currency == "BTC":
        return BTC_to_USDT(timestamp) * get_value_of_currency(transaction_currency, timestamp, "BTC") * price


if __name__ == "__main__":
    timestamp = "1513183102.19288"
    print(get_value_of_currency("OMG", "1513183102.19288", "BTC"))
    print(datetime_from_timestamp(float(timestamp)), BTC_to_USDT("1513183102.19288"))
    print(get_value_of_currency_in_USDT("OMG", timestamp))



