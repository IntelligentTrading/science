from evaluation import Evaluation
from tick_listener import TickListener
from orders import Order, OrderType
from data_sources import Horizon
from tick_provider_itf_db import TickProviderITFDB
import pandas as pd
import logging
import empyrical
import numpy as np

class TickDrivenBacktester(Evaluation, TickListener):

    def __init__(self, tick_provider, **kwargs):
        super().__init__(**kwargs)
        self.tick_provider = tick_provider
        self.trading_df = pd.DataFrame(columns=['close_price', 'signal', 'order', 'cash', 'crypto', 'total_value'])
        self.run()

    def run(self):
        # register at tick provider
        self.tick_provider.add_listener(self)

        # ingest ticks
        self.tick_provider.run()

        # the provider will call the broadcast_ended() method when no ticks remain

    def process_event(self, price_data, signals_now):
        timestamp = price_data['timestamp']
        price = price_data['close_price'].item()
        decision, order_signal = self.strategy.process_ticker(price_data, signals_now)
        order = None
        if decision == "SELL" and self.crypto > 0:
            order = Order(OrderType.SELL, self.transaction_currency, self.counter_currency,
                          timestamp, self.crypto, price, self.transaction_cost_percent, 0)
            self.orders.append(order)
            self.order_signals.append(order_signal)
            self.execute_order(order)
        elif decision == "BUY" and self.cash > 0:
            order = Order(OrderType.BUY, self.transaction_currency, self.counter_currency,
                          timestamp, self.cash, price, self.transaction_cost_percent, 0)
            self.orders.append(order)
            self.order_signals.append(order_signal)
            self.execute_order(order)

        # compute asset value at this tick, regardless of the signal
        total_value = self.crypto * price + self.cash

        # fill a row in the trading dataframe
        self.trading_df.loc[timestamp] = pd.Series({'close_price': price,
                                                    'cash': self.cash,
                                                    'crypto': self.crypto,
                                                    'total_value': total_value,
                                                    'order': "" if order is None else order.order_type.value,
                                                    'signal': "" if order is None else order_signal.signal_type})

    def broadcast_ended(self):
        self.finalize_backtesting()

        if self.verbose:
            logging.info(self.get_report())
            logging.info(self.trading_df)

    def finalize_backtesting(self):
        # set finishing variable values
        self.end_cash = self.cash
        self.end_crypto = self.crypto
        self.end_price = self.trading_df.tail(1)['close_price'].item()

        # compute returns for stats
        self.trading_df = self._fill_returns(self.trading_df)
        returns = np.array(self.trading_df['return_relative_to_past_tick'])
        self._max_drawdown = empyrical.max_drawdown(np.array(returns))
        self._sharpe_ratio = empyrical.sharpe_ratio(returns)

        # extract only rows that have orders
        orders_df = self.trading_df[self.trading_df['order'] != ""]
        # recalculate returns
        orders_df = self._fill_returns(orders_df)
        # get profits on sell
        orders_sell_df = orders_df[orders_df['order'] == "SELL"]
        self._buy_sell_pair_returns = np.array(orders_sell_df['return_relative_to_past_tick'])
        self._buy_sell_pair_gains = self._buy_sell_pair_returns[np.where(self._buy_sell_pair_returns > 0)]
        self._buy_sell_pair_losses = self._buy_sell_pair_returns[np.where(self._buy_sell_pair_returns < 0)]

        # if no returns, no gains or no losses, stat functions will return nan
        if len(self._buy_sell_pair_returns) == 0:
            self._buy_sell_pair_returns = np.array([np.nan])

        if len(self._buy_sell_pair_returns) == 0:
            self._buy_sell_pair_losses = np.array([np.nan])

        if len(self._buy_sell_pair_returns) == 0:
            self._buy_sell_pair_losses = np.array([np.nan])


    def _fill_returns(self, df):
        df['return_from_initial_investment'] = (df['total_value'] - self.get_start_value()) / self.get_start_value()
        df['return_relative_to_past_tick'] = df['total_value'].diff() / df['total_value'].shift(1)
        return df

    def plot_portfolio(self):
        import matplotlib.pyplot as plt
        self.trading_df['close_price'].plot()
        self.trading_df['total_value'].plot(secondary_y=True)
        plt.show()

    # override to show all our new stats
    def get_report(self, include_order_signals=True):
        Evaluation.get_report(self, include_order_signals)
        logging.info("\nHere are our new stats:\n\n")
        logging.info("Max drawdown: {}".format(self.max_drawdown))
        logging.info("Sharpe ratio: {}".format(self.sharpe_ratio))
        logging.info("Buy-sell pair gains - overall stats")
        logging.info("   min = {}, max = {}, mean = {}, stdev = {}".format(
            self.min_buy_sell_pair_gain,
            self.max_buy_sell_pair_gain,
            self.mean_buy_sell_pair_gain,
            self.std_buy_sell_pair_gain
        ))

        logging.info("Buy-sell pair losses - overall stats")
        logging.info("   min = {}, max = {}, mean = {}, stdev = {}".format(
            self.min_buy_sell_pair_loss,
            self.max_buy_sell_pair_loss,
            self.mean_buy_sell_pair_loss,
            self.std_buy_sell_pair_loss
        ))

        logging.info("Total buy-sell pairs: {}".format(self.num_buy_sell_pairs))
        logging.info("Total profitable trades: {}".format(self.num_profitable_trades))
        logging.info("Percent profitable trades: {}".format(self.percent_profitable_trades))
        logging.info("Percent unprofitable trades: {}".format(self.percent_unprofitable_trades))

    @property
    def max_drawdown(self):
        return self._max_drawdown

    @property
    def sharpe_ratio(self):
        return self._sharpe_ratio

    @property
    def min_buy_sell_pair_gain(self):
        return self._buy_sell_pair_gains.min()

    @property
    def max_buy_sell_pair_gain(self):
        return self._buy_sell_pair_gains.max()

    @property
    def mean_buy_sell_pair_gain(self):
        return self._buy_sell_pair_gains.mean()

    @property
    def std_buy_sell_pair_gain(self):
        return self._buy_sell_pair_gains.std()

    @property
    def min_buy_sell_pair_loss(self):
        return self._buy_sell_pair_losses.min()

    @property
    def max_buy_sell_pair_loss(self):
        return self._buy_sell_pair_losses.max()

    @property
    def mean_buy_sell_pair_loss(self):
        return self._buy_sell_pair_losses.mean()

    @property
    def std_buy_sell_pair_loss(self):
        return self._buy_sell_pair_losses.std()

    @property
    def num_buy_sell_pairs(self):
        return self.num_sells

    @property
    def percent_profitable_trades(self):
        return len(self._buy_sell_pair_gains) / self.num_buy_sell_pairs

    @property
    def percent_unprofitable_trades(self):
        return len(self._buy_sell_pair_losses) / self.num_buy_sell_pairs


if __name__ == '__main__':
    from strategies import RSITickerStrategy
    end_time = 1531699200
    start_time = end_time - 60*60*24*70
    start_cash = 10000000
    start_crypto = 0
    transaction_currency = 'BTC'
    counter_currency = 'USDT'
    strategy = RSITickerStrategy(start_time, end_time, Horizon.short, None)

    # supply ticks from the ITF DB
    tick_provider = TickProviderITFDB(transaction_currency, counter_currency, start_time, end_time)

    # create a new tick based backtester
    evaluation = TickDrivenBacktester(tick_provider=tick_provider,
                                      strategy=strategy,
                                      transaction_currency='BTC',
                                      counter_currency='USDT',
                                      start_cash=start_cash,
                                      start_crypto=start_crypto,
                                      start_time=start_time,
                                      end_time=end_time)


