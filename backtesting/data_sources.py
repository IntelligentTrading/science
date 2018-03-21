import mysql.connector
from signals import RSISignal
from config import database_config
from utils import datetime_from_timestamp
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
                                WHERE transaction_currency = "%s") 
                            AND transaction_currency = "%s"
                            AND counter_currency = %s;"""

price_query = """SELECT price FROM indicator_price 
                            WHERE transaction_currency = %s
                            AND timestamp = %s
                            AND counter_currency = %s;"""

timestamp_range_query = """SELECT MIN(timestamp), MAX(timestamp) FROM indicator_price WHERE counter_currency = 2;"""


def get_rsi_signals(transaction_currency, start_time, end_time, counter_currency="BTC"):
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    cursor.execute(signal_query, params=("RSI", transaction_currency, counter_currency_id, start_time, end_time))
    signals = []
    for (trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value) in cursor:
        signals.append(RSISignal(trend, horizon, strength_value, strength_max,
                                 price/1E8,  price_change, timestamp, rsi_value, transaction_currency, counter_currency))
    return signals


def get_timestamp_range(counter_currency=CounterCurrency.USDT.value):
    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    cursor.execute(timestamp_range_query, params=(counter_currency))
    (start, end) = cursor.fetchone()
    return start, end


def get_price(currency, timestamp, counter_currency="BTC", normalize=True):
    if currency == counter_currency:
        return 1
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    cursor.execute(price_query, params=(currency, timestamp, counter_currency_id))
    price = cursor.fetchall()
    if cursor.rowcount == 0:
        print("WARNING: No data for value of {} in {} on {}".format(currency, counter_currency, datetime_from_timestamp(timestamp)))
        return None

    assert cursor.rowcount == 1
    price = price[0][0]
    if normalize:
        return price / 1E8
    else:
        return price


def convert_value_to_USDT(value, timestamp, transaction_currency):
    value_BTC_in_USDT = get_price("BTC", timestamp, "USDT")
    value_transaction_currency_in_BTC = get_price(transaction_currency, timestamp, "BTC")
    if value_BTC_in_USDT == None or value_transaction_currency_in_BTC == None:
        return None
    else:
        return value_BTC_in_USDT * value_transaction_currency_in_BTC * value


if __name__ == "__main__":
    timestamp = "1513183102.19288"
    print(get_price("OMG", "1513183102.19288", "BTC"))




