from evaluation import Evaluation
from tick_listener import TickListener
from orders import Order, OrderType
from data_sources import Horizon
from tick_provider_itf_db import TickProviderITFDB
import pandas as pd
import logging

class TickDrivenBacktester(Evaluation, TickListener):

    def __init__(self, tick_provider, **kwargs):
        super().__init__(**kwargs)
        self.tick_provider = tick_provider
        self.trading_df = pd.DataFrame(columns=['close_price', 'signal', 'cash', 'crypto', 'total_value'])
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
                                                    'total_value': total_value})

    def broadcast_ended(self):
        self.end_cash = self.cash
        self.end_crypto = self.crypto
        self.end_price = self.trading_df.tail(1)['close_price'].item()

        if self.verbose:
            logging.info(self.get_report())


    def plot_portfolio(self):
        import matplotlib.pyplot as plt
        self.trading_df['close_price'].plot()
        self.trading_df['total_value'].plot(secondary_y=True)
        plt.show()


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


