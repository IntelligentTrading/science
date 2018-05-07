from enum import Enum

import mysql.connector

from config import database_config
from signals import Signal
from utils import datetime_from_timestamp
from signals import SignalType

class NoPriceDataException(Exception):
    pass


class CounterCurrency(Enum):
    BTC = 0
    ETH = 1
    USDT = 2
    XMR = 3


class Horizon(Enum):
    any = None
    short = 0
    medium = 1
    long = 2

class Strength(Enum):
    any = None
    short = 1
    medium = 2
    long = 3

#(BTC, ETH, USDT, XMR) = list(range(4))

signal_query = """ SELECT trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value 
            FROM signal_signal 
            WHERE   signal_signal.signal=%s AND 
                    transaction_currency=%s AND 
                    counter_currency=%s AND
                    timestamp >= %s AND
                    timestamp <= %s AND
                    source = 0
            ORDER BY timestamp;"""

all_signals_query = """ SELECT signal_signal.signal, trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value 
            FROM signal_signal 
            WHERE   transaction_currency=%s AND 
                    counter_currency=%s AND
                    timestamp >= %s AND
                    timestamp <= %s AND
                    source = 0 AND
                    horizon = %s
            ORDER BY timestamp;"""

rsi_signal_query = """ SELECT trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value 
            FROM signal_signal 
            WHERE   signal_signal.signal=%s AND 
                    transaction_currency=%s AND 
                    counter_currency=%s AND
                    timestamp >= %s AND
                    timestamp <= %s AND
                    (rsi_value > %s OR
                    rsi_value < %s) AND
                    source = 0
            ORDER BY timestamp;"""


most_recent_price_query = """SELECT price FROM indicator_price 
                            WHERE timestamp = 
                                (SELECT MAX(timestamp) 
                                FROM indicator_price 
                                WHERE transaction_currency = "%s") 
                            AND transaction_currency = "%s"
                            AND source = 0
                            AND counter_currency = %s;"""

price_query = """SELECT price FROM indicator_price 
                            WHERE transaction_currency = %s
                            AND timestamp = %s
                            AND source = 0
                            AND counter_currency = %s;"""

timestamp_range_query = """SELECT MIN(timestamp), MAX(timestamp) FROM indicator_price WHERE counter_currency = 2 AND source = 0;"""

trading_against_counter_query = """SELECT DISTINCT(transaction_currency) FROM indicator_price WHERE counter_currency = %s AND source = 0"""


trading_against_counter_and_signal_query = """SELECT DISTINCT(transaction_currency) FROM signal_signal 
                                              WHERE counter_currency = %s AND signal_signal.signal = %s AND source = 0"""

nearest_price_query = """SELECT price, timestamp FROM indicator_price WHERE transaction_currency = %s AND counter_currency = %s 
                         AND source = 0 AND timestamp <= %s ORDER BY timestamp DESC LIMIT 1;"""


def get_filtered_signals(signal_type=None, transaction_currency=None, start_time=None, end_time=None, horizon=None,
                         counter_currency=None):
    query = """ SELECT signal_signal.signal, trend, horizon, strength_value, strength_max, price, price_change, 
                timestamp, rsi_value, transaction_currency, counter_currency FROM signal_signal """
    additions = []
    params = []
    if signal_type is not None:
        additions.append("signal_signal.signal=%s")
        params.append(signal_type.value)
    if transaction_currency is not None:
        additions.append("transaction_currency = %s")
        params.append(transaction_currency)
    if start_time is not None:
        additions.append("timestamp >= %s")
        params.append(start_time)
    if end_time is not None:
        additions.append("timestamp <= %s")
        params.append(end_time)
    if horizon is not None:
        additions.append("horizon = %s")
        params.append(horizon.value)
    if counter_currency is not None:
        additions.append("counter_currency = %s")
        params.append(CounterCurrency[counter_currency].value)

    # TODO: support for different sources
    additions.append("source = %s")
    params.append(0)

    if len(additions) > 0:
        query += "WHERE {}".format(" AND ".join(additions))
        params = tuple(params)

    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    cursor.execute(query, params)
    print(query)

    signals = []

    for (signal_type, trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value,
         transaction_currency, counter_currency) in cursor:
        signals.append(Signal(signal_type, trend, horizon, strength_value, strength_max,
                              price/1E8,  price_change, timestamp, rsi_value, transaction_currency,
                              CounterCurrency(counter_currency).name))
    connection.close()
    return signals


def get_signals(signal_type, transaction_currency, start_time, end_time, counter_currency="BTC"):
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    cursor.execute(signal_query, params=(signal_type.value, transaction_currency, counter_currency_id, start_time, end_time))
    signals = []
    for (trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value) in cursor:
        signals.append(Signal(signal_type, trend, horizon, strength_value, strength_max,
                                 price/1E8,  price_change, timestamp, rsi_value, transaction_currency,
                              CounterCurrency[counter_currency]))
    return signals

def get_all_signals(transaction_currency, start_time, end_time, horizon, counter_currency="BTC"):
    counter_currency_id = CounterCurrency[counter_currency].value
    horizon_id = horizon.value
    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    cursor.execute(all_signals_query, params=(transaction_currency, counter_currency_id, start_time, end_time, horizon_id))
    signals = []
    for (signal_type, trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value) in cursor:
        signals.append(Signal(signal_type, trend, horizon, strength_value, strength_max,
                                 price/1E8,  price_change, timestamp, rsi_value, transaction_currency,
                              CounterCurrency[counter_currency]))
    return signals

def get_signals_rsi(transaction_currency, start_time, end_time, rsi_overbought, rsi_oversold, counter_currency="BTC"):
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    cursor.execute(rsi_signal_query, params=("RSI", transaction_currency, counter_currency_id, start_time, end_time, rsi_overbought, rsi_oversold))
    signals = []
    for (trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value) in cursor:
        signals.append(Signal(SignalType.RSI, trend, horizon, strength_value, strength_max,
                                 price/1E8,  price_change, timestamp, rsi_value, transaction_currency,
                              CounterCurrency[counter_currency]))
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
    connection = mysql.connector.connect(**database_config, pool_name="my_pool", pool_size=32)
    cursor = connection.cursor()
    cursor.execute(price_query, params=(currency, timestamp, counter_currency_id))
    price = cursor.fetchall()
    if cursor.rowcount == 0:
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


def get_price_nearest_to_timestamp(currency, timestamp, counter_currency, max_delta_seconds=60*5):
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config, pool_name="my_pool", pool_size=32)
    cursor = connection.cursor()
    cursor.execute(nearest_price_query, params=(currency, counter_currency_id, timestamp))
    data = cursor.fetchall()
    if len(data) == 0:
        error_text = "ERROR: No data for value of {} in {} on {}".format(currency, counter_currency,
                                                                         datetime_from_timestamp(timestamp))
        print(error_text)
        connection.close()
        raise NoPriceDataException()

    assert cursor.rowcount == 1
    price, nearest_timestamp = data[0]
    if abs(timestamp - nearest_timestamp) > max_delta_seconds:
        connection.close()
        raise NoPriceDataException()
    connection.close()
    return price / 1E8


def convert_value_to_USDT(value, timestamp, transaction_currency):
    if value == 0:
        return 0
    if transaction_currency == "USDT": # already in USDT
        return value
    try:
        value_USDT = value * get_price(transaction_currency, timestamp, "USDT") # if trading against USDT
        # print("Found USDT price data for {}".format(transaction_currency))
        return value_USDT
    except:
        # print("Couldn't find USDT price data for {}".format(transaction_currency))
        value_BTC_in_USDT = get_price("BTC", timestamp, "USDT")
        if transaction_currency == "BTC":
            return value * value_BTC_in_USDT

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


def get_currencies_for_signal(counter_currency, signal):
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    cursor.execute(trading_against_counter_and_signal_query , params=(counter_currency_id,signal,))
    data = cursor.fetchall()
    currencies = []
    for currency in data:
        currencies.append(currency[0])
    return currencies



