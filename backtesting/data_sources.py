import mysql.connector
from signals import Signal
from config import database_config
from utils import datetime_from_timestamp
from enum import Enum


class NoPriceDataException(Exception):
    pass


class CounterCurrency(Enum):
    BTC = 0
    ETH = 1
    USDT = 2
    XMR = 3


class SignalType(Enum):
    RSI = "RSI"
    kumo_breakout = "kumo_breakout"
    SMA = "SMA"
    EMA = "EMA"
    RSI_Cumulative = "RSI_cumulative"

#(BTC, ETH, USDT, XMR) = list(range(4))

signal_query = """ SELECT trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value 
            FROM signal_signal 
            WHERE   signal_signal.signal=%s AND 
                    transaction_currency=%s AND 
                    counter_currency=%s AND
                    timestamp >= %s AND
                    timestamp <= %s
            ORDER BY timestamp;"""

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

trading_against_counter_query = """SELECT DISTINCT(transaction_currency) FROM indicator_price WHERE counter_currency = %s"""

nearest_price_query = """SELECT price, timestamp FROM indicator_price WHERE transaction_currency = %s AND counter_currency = %s 
                         AND timestamp <= %s ORDER BY timestamp DESC LIMIT 1;"""

def get_signals(signal_type, transaction_currency, start_time, end_time, counter_currency="BTC"):
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    cursor.execute(signal_query, params=(signal_type.value, transaction_currency, counter_currency_id, start_time, end_time))
    signals = []
    for (trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value) in cursor:
        signals.append(Signal(signal_type, trend, horizon, strength_value, strength_max,
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
    connection = mysql.connector.connect(**database_config, pool_name="my_pool", pool_size=5)
    cursor = connection.cursor()
    cursor.execute(price_query, params=(currency, timestamp, counter_currency_id))
    price = cursor.fetchall()
    if cursor.rowcount == 0:
        error_text = "ERROR: No data for value of {} in {} on {}".format(currency, counter_currency, datetime_from_timestamp(timestamp))
        print(error_text)
        connection.close()
        return get_price_nearest_to_timestamp(currency, timestamp, counter_currency)
        #raise NoPriceDataException(error_text)

    assert cursor.rowcount == 1
    connection.close()

    price = price[0][0]
    if normalize:
        return price / 1E8
    else:
        return price


def get_price_nearest_to_timestamp(currency, timestamp, counter_currency, max_delta_seconds = 5):
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config, pool_name="my_pool", pool_size=5)
    cursor = connection.cursor()
    cursor.execute(nearest_price_query, params=(currency, counter_currency_id, timestamp))
    data = cursor.fetchall()
    if len(data) == 0:
        raise NoPriceDataException()

    assert cursor.rowcount == 1
    connection.close()
    price, nearest_timestamp = data[0]
    if abs(timestamp - nearest_timestamp) > max_delta_seconds:
        raise NoPriceDataException()

    return price / 1E8


def convert_value_to_USDT(value, timestamp, transaction_currency):
    if transaction_currency == "USDT":
        return value
    value_BTC_in_USDT = get_price("BTC", timestamp, "USDT")
    value_transaction_currency_in_BTC = get_price(transaction_currency, timestamp, "BTC")
    return value_BTC_in_USDT * value_transaction_currency_in_BTC * value


def get_currencies_trading_against_counter(counter_currency):
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    cursor.execute(trading_against_counter_query, params=(counter_currency_id,))
    data = cursor.fetchall()
    currencies = []
    for currency in data:
        currencies.append(currency[0])
    return currencies






