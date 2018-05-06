import datetime


def datetime_from_timestamp(timestamp):
    return datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def get_distinct_signal_types(signals):
    return set([x.signal_signature for x in signals])
