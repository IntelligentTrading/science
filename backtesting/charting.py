import pyfolio as pf
from orders import OrderType
import pandas as pd


class BacktestingChart:

    def __init__(self, backtest, benchmark=None):
        self.backtest = backtest
        self.trading_df = backtest.trading_df.copy()
        self.trading_df.index = pd.to_datetime(self.trading_df.index, unit='s', utc=True)
        if benchmark is not None:
            self.benchmark_trading_df = benchmark.trading_df.copy()
            self.benchmark_trading_df.index = pd.to_datetime(self.benchmark_trading_df.index, unit='s', utc=True)
        self.benchmark = benchmark
        self.draw_price_and_cumulative_returns()

    @staticmethod
    def price(x):
        return '$%1.2f' % x

    def draw_returns_tear_sheet(self, save_file=True, out_filename='pyfolio_returns_tear_sheet.png'):
        import matplotlib
        if save_file:
            matplotlib.use('Agg')

        f = pf.create_returns_tear_sheet(returns=self.trading_df['return_relative_to_past_tick'],
                                         return_fig=True)
                                         # benchmark_rets=
                                         #   self.benchmark_trading_df['return_relative_to_past_tick']
                                         #   if self.benchmark is not None else None)
        f.savefig(out_filename)

    def plot_orders(self, ax, orders):
        for order in orders:
            if order.order_type == OrderType.BUY:
                color = "g"
            else:
                color = "r"
            timestamp = pd.to_datetime(order.timestamp, unit="s")
            ax.axvline(timestamp, color=color, lw=1, zorder=-1)

    def draw_price_and_cumulative_returns(self):
        import matplotlib.pyplot as plt

        orders = self.backtest.get_orders()
        ax1 = self.trading_df['close_price'].plot()
        ax2 = self.trading_df['total_value'].plot(secondary_y=True)
        ax1.set_ylabel('Price')
        ax2.set_ylabel('Total value')

        if orders != None:
            self.plot_orders(ax1, orders)

        ax1.format_ydata = self.price
        ax1.grid(False)

        plt.show()

