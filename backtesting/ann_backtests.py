from data_sources import postgres_db
from utils import datetime_from_timestamp
import numpy as np
import pandas as pd
from utils import datetime_to_timestamp
from collections import OrderedDict


def analyze_ann_signals(signal_type, start_time, end_time, source, ann_lookahead_window, hit_threshold,
                        normalize=False):
    # get signals
    signals = postgres_db.get_filtered_signals(signal_type=signal_type, start_time=start_time, end_time=end_time, source=source,
                                               normalize=normalize)

    correct_up = 0
    correct_down = 0
    total = 0
    incorrect = 0
    # get resampled prices
    differences = []
    print(f'Retrieved {len(signals)} signals')
    for signal in signals:
        target_time = signal.timestamp + ann_lookahead_window * signal.resample_period * 60
        price_data = postgres_db.get_nearest_resampled_price(target_time,
                                                             signal.transaction_currency, signal.counter_currency,
                                                             signal.resample_period, signal.source, normalize=normalize)

        if price_data.empty:
            continue

        future_price = price_data.iloc[0].close_price
        future_timestamp = price_data.iloc[0].name
        if future_timestamp - target_time > 60*60:
            # missed target time by more than one hour, skip
            continue

        print(f'For price {signal.price} at {datetime_from_timestamp(signal.timestamp)}, '
              f'the nearest resampled price {ann_lookahead_window} periods after is {future_price} '
              f'at {datetime_from_timestamp(future_timestamp)}.')

        differences.append(abs(future_price - signal.price) / signal.price)
        if future_price / signal.price > (1 + hit_threshold):
            correct_up += 1
            print('Correct up')
        elif future_price / signal.price < (1 - hit_threshold):
            correct_down += 1
            print('Correct down')
        else:
            incorrect += 1
            print('Incorrect')

        total += 1

    print(differences)
    print(np.mean(differences))
    print(f'Analyzed {total} signals.')
    print(f'{correct_up} correct_up, {correct_down} correct_down, {incorrect} incorrect')


def count_co_occurring_signals(start_time, end_time, out_filename):
    signal_types = ['RSI', 'RSI_Cumulative', 'ANN_Simple', 'ANN_AnomalyPrc', 'kumo_breakout', 'VBI']
    data = OrderedDict()
    for row_signal in signal_types:
        data[row_signal] = {}
        for col_signal in signal_types:
            data[row_signal][col_signal] = postgres_db.count_signals_ocurring_at_the_same_time(row_signal, col_signal, start_time, end_time)
    df = pd.DataFrame(data)
    df = df.reindex(signal_types)  # so the index is not sorted alphabetically
    writer = pd.ExcelWriter(out_filename)
    df.to_excel(writer, 'Counts')
    writer.save()
    return df


if __name__ == '__main__':
    analyze_ann_signals('ANN_AnomalyPrc', datetime_to_timestamp('2018-10-01 00:00:00'),
                        datetime_to_timestamp('2018-11-01 00:00:00'), None, 4, 0.05, normalize=False)
    count_co_occurring_signals(datetime_to_timestamp('2018-10-01 00:00:00'),
                               datetime_to_timestamp('2018-11-01 00:00:00'),
                               'cooccurring.xlsx')