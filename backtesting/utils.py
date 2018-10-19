import datetime
import logging
import time
from config import POOL_SIZE
from pathos.multiprocessing import Pool


def datetime_from_timestamp(timestamp):
    return datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def get_distinct_signal_types(signals):
    return set([x.signal_signature if x is not None else '(no signals)' for x in signals])


def time_performance(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end-start:.4f} seconds")
        return result

    return wrapper


def parallel_run(func, param_list, pool_size=POOL_SIZE):
    with Pool(pool_size) as pool:
        results = pool.map(func, param_list)
        pool.close()
        pool.join()
        pool.terminate() # needed for Pathos,
#        pool.restart()   # see https://stackoverflow.com/questions/49888485/pathos-multiprocessings-pool-appears-to-be-nonlocal
    return results

