import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backtesting.orders import Order, OrderType
import pandas as pd

def price(x):
    return '$%1.2f' % x

def plot_orders(ax, orders):
    for order in orders:
        if order.order_type == OrderType.BUY:
            color = "g"
        else:
            color = "r"
        timestamp = pd.to_datetime(order.timestamp, unit="s")
        price = order.unit_price
        circle = plt.Circle((timestamp, price), 0.02, color=color)
        ax.add_artist(circle)


def draw_chart(timestamps, prices, orders):

    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    fig, ax = plt.subplots()
    ax.plot(timestamps, prices)

    #circle1 = plt.Circle((timestamps[100], prices[100]), 0.02, color='r')
    #ax.add_artist(circle1)

    plot_orders(ax, orders)


    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)

    # round to nearest years...
    datemin = np.datetime64(timestamps[0]) #, 'Y')
    datemax = np.datetime64(timestamps[-1]) #, 'Y') + np.timedelta64(1, 'Y')
    ax.set_xlim(datemin, datemax)


    # format the coords message box

    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.format_ydata = price
    ax.grid(True)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()

    plt.show()