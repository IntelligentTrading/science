import datetime
MAX_TIME = 10000000000000000


def datetime_from_timestamp(timestamp):
    if timestamp == MAX_TIME:
        return "using all available data, no upper time limit set"
    return datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
