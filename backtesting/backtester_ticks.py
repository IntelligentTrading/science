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
        self.run()

    def run(self):
        # register at tick provider
        self.tick_provider.add_listener(self)

        # ingest ticks
        self.tick_provider.run()

        # the provider will call the broadcast_ended() method when no ticks remain

    def process_event(self, price_data, signals_now):
        self._current_timestamp = price_data['timestamp']
        self._current_price = price_data['close_price'].item()
        decision, order_signal = self._strategy.process_ticker(price_data, signals_now)
        order = None
        if decision == "SELL" and self._crypto > 0:
            order = Order(OrderType.SELL, self._transaction_currency, self._counter_currency,
                          self._current_timestamp, self._crypto, self._current_price, self._transaction_cost_percent, 0)
            self.orders.append(order)
            self.order_signals.append(order_signal)
            self.execute_order(order)
        elif decision == "BUY" and self._cash > 0:
            order = Order(OrderType.BUY, self._transaction_currency, self._counter_currency,
                          self._current_timestamp, self._cash, self._current_price, self._transaction_cost_percent, 0)
            self.orders.append(order)
            self.order_signals.append(order_signal)
            self.execute_order(order)

        self._current_order = order
        self._current_signal = order_signal

        self._write_to_trading_df()

        """
        # compute asset value at this tick, regardless of the signal
        total_value = self._crypto * self._current_price + self._cash

        # fill a row in the trading dataframe
        self.trading_df.loc[timestamp] = pd.Series({'close_price': self._current_price,
                                                    'cash': self._cash,
                                                    'crypto': self._crypto,
                                                    'total_value': total_value,
                                                    'order': "" if order is None else order.order_type.value,
                                                    'signal': "" if order is None else order_signal.signal_type})
        """

    def broadcast_ended(self):
        self._finalize_backtesting()

        if self._verbose:
            logging.info(self.get_report())
            logging.info(self.trading_df)


    def plot_portfolio(self):
        import matplotlib.pyplot as plt
        self.trading_df['close_price'].plot()
        self.trading_df['total_value'].plot(secondary_y=True)
        plt.show()

    # override to show all our new stats
    def get_report(self, include_order_signals=True):
        logging.info(Evaluation.get_report(self, include_order_signals))
        logging.info("\nHere are our new stats:")
        logging.info("  Max drawdown: {}".format(self.max_drawdown))
        logging.info("  Sharpe ratio: {}".format(self.sharpe_ratio))
        logging.info("  Buy-sell pair gains - overall stats")
        logging.info("     min = {}, max = {}, mean = {}, stdev = {}".format(
            self.min_buy_sell_pair_gain,
            self.max_buy_sell_pair_gain,
            self.mean_buy_sell_pair_gain,
            self.std_buy_sell_pair_gain
        ))

        logging.info("  Buy-sell pair losses - overall stats")
        logging.info("     min = {}, max = {}, mean = {}, stdev = {}".format(
            self.min_buy_sell_pair_loss,
            self.max_buy_sell_pair_loss,
            self.mean_buy_sell_pair_loss,
            self.std_buy_sell_pair_loss
        ))

        logging.info("  Total buy-sell pairs: {}".format(self.num_buy_sell_pairs))
        logging.info("  Total profitable trades: {}".format(self.num_profitable_trades))
        logging.info("  Percent profitable trades: {}".format(self.percent_profitable_trades))
        logging.info("  Percent unprofitable trades: {}".format(self.percent_unprofitable_trades))

    @property
    def end_price(self):
        #if not self.trading_df.empty:
        #    return self.trading_df.tail(1)['close_price'].item()
        #else:
        return Evaluation.end_price.fget(self)


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


