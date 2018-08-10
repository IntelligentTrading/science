import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from orders import OrderType
import pandas as pd


class BacktestingChart:

    def __init__(self, trading_df, orders):
        trading_df.index = pd.to_datetime(trading_df.index, unit='s', utc=True)
        self.trading_df = trading_df
        self.orders = orders

    @staticmethod
    def price(x):
        return '$%1.2f' % x

    def plot_orders(self, ax, orders):
        for order in orders:
            if order.order_type == OrderType.BUY:
                color = "g"
            else:
                color = "r"
            timestamp = pd.to_datetime(order.timestamp, unit="s")
            price = order.unit_price
            ax.axvline(timestamp, color=color, lw=1, zorder=-1)
            #circle = plt.Circle((timestamp, price), 2, color=color)
            #ax.add_artist(circle)

    def draw_chart_from_dataframe(self, df, data_column_name):
        timestamps = pd.to_datetime(df.index.values, unit='s')
        data = df.as_matrix(columns=[data_column_name])
        self.draw_price_chart(timestamps, data, None)

    def draw_price_chart(self):
        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator()  # every month
        weeks = mdates.WeekdayLocator()
        days = mdates.DayLocator()  # every day
        daysFmt = mdates.DateFormatter('%m/%d')
        monthsFmt = mdates.DateFormatter('%m')

        orders = self.orders

        ax1 = self.trading_df['close_price'].plot()
        ax2 = self.trading_df['total_value'].plot(secondary_y=True)



        if orders != None:
            self.plot_orders(ax1, orders)

        plt.show()
        return

        # format the ticks
        ax1.xaxis.set_major_locator(years)
        ax1.xaxis.set_minor_locator(days)
        ax1.xaxis.set_minor_formatter(daysFmt)
        plt.setp(ax1.xaxis.get_minorticklabels(), rotation=90)


        datemin = np.datetime64(int(self.trading_df.index.values.min()), 's')
        datemax = np.datetime64(int(self.trading_df.index.values.max()), 's')
        ax1.set_xlim(datemin, datemax)

        # format the coords message box
        ax1.format_xdata = daysFmt
        #ax1.format_ydata = self.price
        ax1.grid(False)

        plt.ylabel("Price", fontsize=14)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

        plt.show()

