import datetime
import logging
import time


def datetime_from_timestamp(timestamp):
    return datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def get_distinct_signal_types(signals):
    return set([x.signal_signature for x in signals])


def time_performance(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end-start:.4f} seconds")
        return result

    return wrapper