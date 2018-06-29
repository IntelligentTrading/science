import talib
from data_sources import get_prices_in_range, get_volumes_in_range
from strategies import  Horizon
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from signals import Signal
from strategies import SignalSignatureStrategy
import operator
from orders import OrderType

# Initial settings
transaction_currency = "ETH"
counter_currency = "BTC"
end_time = 1526637600
start_time = end_time - 60 * 60 * 24 * 60
horizon = Horizon.short
resample_period = 60
start_cash = 1
start_crypto = 0
source = 0
strength = 3
history_size = 100
AVERAGING_PERIOD = 50*60

# Load price and volume data and calculate average prices and volumes
prices_df = get_prices_in_range(start_time, end_time, transaction_currency, counter_currency, source)
prices_df.price /= 1E8

volumes_df = get_volumes_in_range(start_time, end_time, transaction_currency, counter_currency, source)
volumes_df.volume /= 1E5 # scaling for visualization

sma_price = talib.SMA(np.array(prices_df.price, dtype=float), timeperiod=AVERAGING_PERIOD)
sma_volume = talib.SMA(np.array(volumes_df.volume, dtype=float), timeperiod=AVERAGING_PERIOD)

prices_df['average_price'] = pd.Series(sma_price, index=prices_df.index)
volumes_df['average_volume'] = pd.Series(sma_volume, index=volumes_df.index)
price_volume_df = prices_df.join(volumes_df)


def build_signals(percent_change_price, percent_change_volume):
    # build signals
    all_buy_signals = []
    first_cross_buy_signals = []
    valid_in_previous_step = False
    for row in price_volume_df.itertuples():
        timestamp = row.Index
        price = row.price
        volume = row.volume
        avg_price = row.average_price
        avg_volume = row.average_volume

        # check whether to generate a buy signal:
        if price > (1+percent_change_price)*avg_price and volume > (1+percent_change_volume)*avg_volume:
            signal = Signal("RSI", 1, Horizon.any, 3, 3, price, 0, timestamp, 0, transaction_currency, counter_currency)
            all_buy_signals.append(signal)
            if not valid_in_previous_step:
                valid_in_previous_step = True
                first_cross_buy_signals.append(signal)
                print(timestamp, price, avg_price, volume, avg_volume)
        else:
            valid_in_previous_step = False
    return all_buy_signals, first_cross_buy_signals

all_buy_signals, first_cross_buy_signals = build_signals(0, 0)

strategy = SignalSignatureStrategy(["rsi_buy_2", "rsi_sell_2", "rsi_buy_3", "rsi_sell_3"],
                                   start_time, end_time, horizon,
                                   counter_currency, transaction_currency, source)
rsi_sell_signals = strategy.get_sell_signals()

#signals.extend(rsi_sell_signals)
first_cross_buy_signals.extend(rsi_sell_signals)
#sorted_signals = sorted(signals, key=operator.attrgetter('timestamp'))
sorted_signals = sorted(first_cross_buy_signals, key=operator.attrgetter('timestamp'))

# burn in hell hacky, but it works :)
strategy.signals = sorted_signals
orders, _ = strategy.get_orders(start_cash, start_crypto)

# Backtest the strategy
print(strategy.evaluate(start_cash, start_crypto, start_time, end_time))

def plot_results(include_all_buy, include_first_cross, include_orders):
    prices_df.index = pd.to_datetime(prices_df.index, unit='s')
    volumes_df.index = pd.to_datetime(volumes_df.index, unit='s')
    price_volume_df.index = pd.to_datetime(price_volume_df.index, unit='s')

    ax = price_volume_df.plot(lw=1, title="ETH to BTC")
    volumes_df.plot(lw=2)

    if include_all_buy:
        for signal in all_buy_signals:
            timestamp = pd.to_datetime(signal.timestamp, unit="s")
            price = signal.price
            if signal.trend == 1:
                #ax.axvline(timestamp, color="lightgreen", lw=1)
                circle = plt.Circle((timestamp, price), 0.1, color="lightgreen")
                ax.add_artist(circle)

    if include_first_cross:
        for signal in first_cross_buy_signals:
            timestamp = pd.to_datetime(signal.timestamp, unit="s")
            price = signal.price
            ax.axvline(timestamp, color="pink", lw=1)
            #circle = plt.Circle((timestamp, price), 0.05, color="lightgreen")
            #ax.add_artist(circle)

    if include_orders:
        for order in orders:
            if order.order_type == OrderType.BUY:
                color = "g"
            else:
                color = "r"
            timestamp = pd.to_datetime(order.timestamp, unit="s")
            #timestamp = order.timestamp
            price = order.unit_price
            print(timestamp,price)
            ax.axvline(timestamp, color=color, lw=2)
            #circle = plt.Circle((timestamp, price), 0.05, color=color)
            #ax.add_artist(circle)

    ax.set_yticklabels([])
    plt.show()

plot_results(True, False, True)

