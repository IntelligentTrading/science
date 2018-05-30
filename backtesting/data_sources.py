from enum import Enum

import mysql.connector

from config import database_config
from signals import Signal
import pandas as pd
from signals import SignalType
from config import output_log


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



price_query = """SELECT price FROM indicator_price 
                            WHERE transaction_currency = %s
                            AND timestamp = %s
                            AND source = %s
                            AND counter_currency = %s;"""

trading_against_counter_query = """SELECT DISTINCT(transaction_currency) FROM indicator_price WHERE counter_currency = %s AND source = 0"""


trading_against_counter_and_signal_query = """SELECT DISTINCT(transaction_currency) FROM signal_signal 
                                              WHERE counter_currency = %s AND signal_signal.signal = %s AND source = 0"""

price_in_range_query_desc = """SELECT price, timestamp FROM indicator_price WHERE transaction_currency = %s AND counter_currency = %s 
                         AND source = %s AND timestamp >= %s AND timestamp <= %s ORDER BY timestamp DESC"""

price_in_range_query_asc = """SELECT price, timestamp FROM indicator_price WHERE transaction_currency = %s AND counter_currency = %s 
                         AND source = %s AND timestamp >= %s AND timestamp <= %s ORDER BY timestamp ASC"""

resampled_price_range_query = """SELECT timestamp, close_price
                                 FROM indicator_priceresampl 
                                 WHERE transaction_currency = %s 
                                 AND counter_currency = %s 
                                 AND source = %s 
                                 AND timestamp >= %s AND timestamp <= %s
                                 AND resample_period = %s
                                 AND close_price is not null"""


def get_resampled_prices_in_range(start_time, end_time, transaction_currency, counter_currency, resample_period, source=0,
                                  normalize=True):
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config, pool_name="my_pool", pool_size=32)
    price_data = pd.read_sql(resampled_price_range_query, con=connection, params=(transaction_currency,
                                                                               counter_currency_id,
                                                                               source,
                                                                               start_time,
                                                                               end_time,
                                                                               resample_period),
                             index_col="timestamp")
    if normalize:
        price_data.loc[:, 'close_price'] /= 1E8
    connection.close()
    return price_data

def get_filtered_signals(signal_type=None, transaction_currency=None, start_time=None, end_time=None, horizon=Horizon.any,
                         counter_currency=None, strength=Strength.any, source=None):
    query = """ SELECT signal_signal.signal, trend, horizon, strength_value, strength_max, price, price_change, 
                timestamp, rsi_value, transaction_currency, counter_currency FROM signal_signal """
    additions = []
    params = []
    if signal_type is not None:
        additions.append("signal_signal.signal=%s")
        params.append(signal_type)
    if transaction_currency is not None:
        additions.append("transaction_currency = %s")
        params.append(transaction_currency)
    if start_time is not None:
        additions.append("timestamp >= %s")
        params.append(start_time)
    if end_time is not None:
        additions.append("timestamp <= %s")
        params.append(end_time)
    if horizon.value is not None:
        additions.append("horizon = %s")
        params.append(horizon.value)
    if counter_currency is not None:
        additions.append("counter_currency = %s")
        params.append(CounterCurrency[counter_currency].value)
    if strength.value is not None:
        additions.append("strength_value = %s")
        params.append(strength.value)
    if source is not None:
        additions.append("source = %s")
        params.append(source)

    if len(additions) > 0:
        query += "WHERE {}".format(" AND ".join(additions))
        params = tuple(params)

    connection = mysql.connector.connect(**database_config)
    cursor = connection.cursor()
    cursor.execute(query, params)

    signals = []

    for (signal_type, trend, horizon, strength_value, strength_max, price, price_change, timestamp, rsi_value,
         transaction_currency, counter_currency) in cursor:
        if len(trend) > 5:   # hacky solution for one instance of bad data
            continue
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



def get_price(currency, timestamp, source, counter_currency="BTC", normalize=True):
    if currency == counter_currency:
        return 1
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config, pool_name="my_pool", pool_size=32)
    cursor = connection.cursor()
    cursor.execute(price_query, params=(currency, timestamp, source, counter_currency_id))

    price = cursor.fetchall()
    if cursor.rowcount == 0:
        connection.close()
        price = get_price_nearest_to_timestamp(currency, timestamp, source, counter_currency)
    else:
        connection.close()
        assert cursor.rowcount == 1
        price = price[0][0]

    if normalize:
        return price / 1E8
    else:
        return price


def get_price_nearest_to_timestamp(currency, timestamp, source, counter_currency, max_delta_seconds_past=60*60,
                                   max_delta_seconds_future=60*5):
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config, pool_name="my_pool", pool_size=32)
    cursor = connection.cursor()
    cursor.execute(price_in_range_query_desc, params=(currency, counter_currency_id, source,
                                                 timestamp - max_delta_seconds_past, timestamp))
    history = cursor.fetchall()
    cursor.execute(price_in_range_query_asc, params=(currency, counter_currency_id, source,
                                                          timestamp, timestamp + max_delta_seconds_future))
    future = cursor.fetchall()
    connection.close()

    if len(history) == 0:
        output_log("Error: no historical price data in {} minutes before timestamp {}...".format(max_delta_seconds_past/60, timestamp))
        if len(future) == 0:
            output_log("No future data found.")
            raise NoPriceDataException()
        else:
            output_log("Returning future price...")

            return future[0][0]
    else:
        output_log("Returning historical price data for timestamp {} (difference of {} minutes)"
              .format(timestamp,(timestamp - history[0][1])/60))
        return history[0][0]

def get_prices_in_range(start_time, end_time, transaction_currency, counter_currency, source):
    counter_currency_id = CounterCurrency[counter_currency].value
    connection = mysql.connector.connect(**database_config, pool_name="my_pool", pool_size=32)
    price_data = pd.read_sql(price_in_range_query_asc, con=connection, params=(transaction_currency,
                                                                               counter_currency_id,
                                                                               source,
                                                                               start_time,
                                                                               end_time),
                             index_col="timestamp")
    connection.close()
    return price_data


def convert_value_to_USDT(value, timestamp, transaction_currency, source):
    if value == 0:
        return 0
    if transaction_currency == "USDT": # already in USDT
        return value
    try:
        value_USDT = value * get_price(transaction_currency, timestamp, source, "USDT") # if trading against USDT
        # print("Found USDT price data for {}".format(transaction_currency))
        return value_USDT
    except:
        # print("Couldn't find USDT price data for {}".format(transaction_currency))
        value_BTC_in_USDT = get_price("BTC", timestamp, source, "USDT")
        if transaction_currency == "BTC":
            return value * value_BTC_in_USDT

        value_transaction_currency_in_BTC = get_price(transaction_currency, timestamp, source, "BTC")
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
    cursor.execute(trading_against_counter_and_signal_query, params=(counter_currency_id,signal,))
    data = cursor.fetchall()
    currencies = []
    for currency in data:
        currencies.append(currency[0])
    return currencies



