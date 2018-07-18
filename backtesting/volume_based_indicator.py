import talib
from data_sources import get_resampled_prices_in_range, get_volumes_in_range
from strategies import  Horizon
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from signals import Signal
from strategies import SignalSignatureStrategy
import operator
from orders import OrderType

def build_signals(price_volume_df, percent_change_price, percent_change_volume, transaction_currency, counter_currency,
                  source, resample_period):
    # build signals
    all_buy_signals = []
    first_cross_buy_signals = []
    valid_in_previous_step = False
    for row in price_volume_df.itertuples():
        timestamp = row.Index.timestamp()
        price = row.price
        volume = row.volume
        avg_price = row.average_price
        avg_volume = row.average_volume

        # check whether to generate a buy signal:
        if price > (1 + percent_change_price)*avg_price and volume > (1 + percent_change_volume)*avg_volume:
            signal = Signal("RSI", 1, Horizon.any, 3, 3, price, 0, timestamp, 0, transaction_currency, counter_currency,
                            source, resample_period)
            all_buy_signals.append(signal)
            if not valid_in_previous_step:
                valid_in_previous_step = True
                first_cross_buy_signals.append(signal)
                # print(timestamp, price, avg_price, volume, avg_volume)
        else:
            valid_in_previous_step = False
    return all_buy_signals, first_cross_buy_signals


def build_strategy(price_volume_df, percent_change_price, percent_change_volume, buy_only_on_first_cross=True,
                   sell_strategy=None, **kwargs):
    source = kwargs['source']
    transaction_currency = kwargs['transaction_currency']
    counter_currency = kwargs['counter_currency']
    start_time = kwargs['start_time']
    end_time = kwargs['end_time']
    horizon = kwargs['horizon']
    source = kwargs['source']
    resample_period = kwargs['resample_period']
    all_buy_signals, first_cross_buy_signals = build_signals(price_volume_df, percent_change_price, percent_change_volume,
                                                             transaction_currency, counter_currency, source, resample_period)

    if sell_strategy is None:
        strategy = SignalSignatureStrategy(["rsi_buy_2", "rsi_sell_2", "rsi_buy_3", "rsi_sell_3"],
                                           start_time, end_time, horizon,
                                           counter_currency, transaction_currency, source)
    else:
        strategy = sell_strategy

    rsi_sell_signals = strategy.get_sell_signals()

    if buy_only_on_first_cross:
        buy_signals = first_cross_buy_signals
    else:
        buy_signals = all_buy_signals

    strategy_signals = buy_signals + rsi_sell_signals
    sorted_signals = sorted(strategy_signals, key=operator.attrgetter('timestamp'))

    # burn in hell hacky, but it works :)
    strategy.signals = sorted_signals
    return strategy, all_buy_signals, first_cross_buy_signals


def plot_results(price_volume_df, transaction_currency, counter_currency, all_buy_signals,
                 first_cross_buy_signals, orders, include_all_buy=True, include_first_cross=False, include_orders=True):
    ax = price_volume_df.plot(lw=1, title="{} to {}".format(transaction_currency, counter_currency), figsize=(16, 10),
                              secondary_y=['volume', 'average_volume'])

    if include_all_buy:
        for signal in all_buy_signals:
            timestamp = pd.to_datetime(signal.timestamp, unit="s")
            price = signal.price
            if signal.trend == 1:
                ax.axvline(timestamp, color="lightgreen", lw=1, zorder=-1)
                # circle = plt.Circle((timestamp, price), 10000, color="lightgreen")
                # ax.add_artist(circle)

    if include_first_cross:
        for signal in first_cross_buy_signals:
            timestamp = pd.to_datetime(signal.timestamp, unit="s")
            price = signal.price
            ax.axvline(timestamp, color="pink", lw=1)
            # circle = plt.Circle((timestamp, price), 0.05, color="lightgreen")
            # ax.add_artist(circle)

    if include_orders:
        for order in orders:
            if order.order_type == OrderType.BUY:
                color = "g"
            else:
                color = "r"
            timestamp = pd.to_datetime(order.timestamp, unit="s")
            # timestamp = order.timestamp
            # price = order.unit_price
            ax.axvline(timestamp, color=color, lw=2)
            # circle = plt.Circle((timestamp, price), 0.05, color=color)
            # ax.add_artist(circle)

    # ax.set_yticklabels([])
    #xmin, xmax = ax.get_xlim()
    #period = pd.Period(ordinal=int(xmax))
    # Then convert to pandas timestamp
    #ts = period.to_timestamp() + 60*60*24

    #ax.set_xbound(xmin, pd.Timestamp(ts).to_period())

    ax.autoscale(enable=True, axis='x', tight=True)

    ax.set_ylim(bottom=0)
    plt.show()


# Try out different thresholds for volume and price change
def calculate_profits(price_volume_df, start_cash, start_crypto, sell_strategy=None, **kwargs):
    start_time = kwargs['start_time']
    end_time = kwargs['end_time']
    volume_change_percents = []
    price_change_percents = []
    profits = []

    for percent_change_volume in np.arange(0, 0.10, 0.005):
        for percent_change_price in np.arange(0, 0.10, 0.005):
            strategy, all_buy_signals, first_cross_buy_signals = build_strategy(price_volume_df, percent_change_price,
                                                                                percent_change_volume,
                                                                                buy_only_on_first_cross=True,
                                                                                sell_strategy=sell_strategy,
                                                                                **kwargs)

            orders, _ = strategy.get_orders(start_cash, start_crypto)
            evaluation = strategy.evaluate(start_cash, start_crypto, start_time, end_time, verbose=False)
            volume_change_percents.append(percent_change_volume)
            price_change_percents.append(percent_change_price)
            profits.append(evaluation.get_profit_percent())
    profit_df = pd.DataFrame.from_items(zip(["Volume change percent", "Price change percent", "Profit percent"],
                                            [volume_change_percents, price_change_percents, profits]))
    return profit_df


def plot_profit_df(profit_df, title="", date_padding=0):
    # Plot scatter
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.dist = 11

    x = profit_df["Volume change percent"]
    y = profit_df["Price change percent"]
    z = profit_df["Profit percent"]

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('Volume change threshold')
    ax.set_ylabel('Price change threshold')
    ax.set_zlabel('Profit [%]')

    plt.title(title)
    plt.show()

    # Plot surface
    # Interpolated mesh

    # %matplotlib notebook
    # plt.interactive(True)

    fig3d = plt.figure(1)

    ax = fig3d.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    from scipy.interpolate import griddata
    from matplotlib import cm
    Z = griddata((x, y), z, (X, Y), method='cubic')

    surface_plot = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('Volume change threshold')
    ax.set_ylabel('Price change threshold')
    ax.set_zlabel('Profit [%]')
    ax.dist = 11

    plt.show()


def write_to_excel(df, path):
    writer = pd.ExcelWriter(path)
    df.to_excel(writer, "Results")
    writer.save()

def build_resampled_price_volume_df(start_time, end_time, transaction_currency, counter_currency, resample_period, source,
                                    AVERAGING_PERIOD = 50):
    # Load price and volume data and calculate average prices and volumes
    prices_df = get_resampled_prices_in_range(start_time, end_time, transaction_currency, counter_currency,
                                              resample_period, source,
                                              normalize=True)
    prices_df = prices_df.sort_index()
    prices_df = prices_df.dropna()

    volumes_df = get_volumes_in_range(start_time, end_time, transaction_currency, counter_currency, source)
    volumes_df = volumes_df.dropna()
    # volumes_df.volume /= 1E4 #5 # scaling for visualization

    prices_avg = talib.SMA(np.array(prices_df.close_price, dtype=float), timeperiod=AVERAGING_PERIOD)
    prices_df['average_price'] = pd.Series(prices_avg, index=prices_df.index)

    if not volumes_df.index.is_unique:
        start_len = len(volumes_df)
        volumes_df = volumes_df[~volumes_df.index.duplicated(keep='first')]
        print(" --> reduced size of volume dataframe from {} to {} because of duplicate data.".format(
            start_len, len(volumes_df)
        ))

    volumes_reindexed_df = volumes_df.reindex(prices_df.index, method='nearest')  # TODO find a better way
    volumes_reindexed_df = volumes_reindexed_df[~volumes_reindexed_df.index.duplicated()]
    volumes_reindexed_df = volumes_reindexed_df.sort_index()

    price_volume_df = prices_df.join(volumes_reindexed_df, how='inner')  # to make sure timestamps match
    price_volume_df = price_volume_df.sort_index()
    volumes_avg = talib.SMA(np.array(price_volume_df['volume'], dtype=float), timeperiod=AVERAGING_PERIOD)

    price_volume_df['average_volume'] = pd.Series(volumes_avg, index=price_volume_df.index)

    # Convert indexes to datetime
    prices_df.index = pd.to_datetime(prices_df.index, unit='s', utc=True)
    volumes_df.index = pd.to_datetime(volumes_df.index, unit='s', utc=True)
    price_volume_df.index = pd.to_datetime(price_volume_df.index, unit='s', utc=True)

    # rename columns so the rest of the code is compatible
    price_volume_df = price_volume_df.rename(columns={'close_price': 'price'})

    return price_volume_df


def evaluate_vbi(**kwargs):
    price_volume_df = build_resampled_price_volume_df(kwargs['start_time'],
                                                      kwargs['end_time'],
                                                      kwargs['transaction_currency'],
                                                      kwargs['counter_currency'],
                                                      kwargs['resample_period'],
                                                      kwargs['source'])
    strategy, all_buy_signals, first_cross_buy_signals = build_strategy(price_volume_df, 0.02, 0.02,
                                                                        buy_only_on_first_cross=True,
                                                                        sell_strategy=None, **kwargs)
    orders, _ = strategy.get_orders(kwargs['start_cash'], kwargs['start_crypto'])
    # Backtest the strategy
    return strategy.evaluate(kwargs['start_cash'], kwargs['start_crypto'], kwargs['start_time'], kwargs['end_time'],
                             verbose = kwargs.get('verbose', False))

def backtest_vbi():
    counter_currency = "BTC"
    end_time = 1531699200
    start_time = end_time - 60 * 60 * 24 * 45
    resample_period = 60
    source = 0

    kwargs = {}
    kwargs['source'] = source

    kwargs['counter_currency'] = counter_currency
    kwargs['start_time'] = start_time
    kwargs['end_time'] = end_time
    kwargs['resample_period'] = resample_period

    from backtesting_runs import get_currencies_for_signal, ComparativeEvaluation
    transaction_currencies = get_currencies_for_signal(counter_currency, "RSI_Cumulative")

    strategies = []
    for resample_period in [60, 240]:
        for transaction_currency in transaction_currencies:
            kwargs['transaction_currency'] = transaction_currency
            kwargs['horizon'] = Horizon.short if resample_period == 60 else Horizon.medium
            kwargs['resample_period'] = resample_period
            print ("Processing {}".format(transaction_currency))
            try:
                price_volume_df = build_resampled_price_volume_df(start_time, end_time, transaction_currency,
                                                                  counter_currency, resample_period, source)
                strategy, all_buy_signals, first_cross_buy_signals = build_strategy(price_volume_df, 0.02, 0.02,
                                                                                    buy_only_on_first_cross=True,
                                                                                    sell_strategy=None, **kwargs)
                strategies.append(strategy)
            except Exception as e:
                print("Error: {}".format(str(e)))

    comparison = ComparativeEvaluation(strategy_set=strategies,
                                       start_cash=1, start_crypto=0,
                                       start_time=start_time, end_time=end_time,
                                       output_file="vbi_backtest_2018_07_17.xlsx"
                                       )



if __name__ == "__main__":
    backtest_vbi()
