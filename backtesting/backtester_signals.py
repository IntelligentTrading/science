from evaluation import Evaluation
from orders import OrderType
import logging
from data_sources import get_price, NoPriceDataException


class SignalDrivenBacktester(Evaluation):

    def __init__(self, strategy, transaction_currency, counter_currency,
                 start_cash, start_crypto, start_time, end_time, source=0,
                 resample_period=60, evaluate_profit_on_last_order=False, verbose=True, time_delay=0):
        super().__init__(strategy, transaction_currency, counter_currency,
                 start_cash, start_crypto, start_time, end_time, source,
                 resample_period, evaluate_profit_on_last_order, verbose)
        self.orders, self.order_signals = self.strategy.get_orders(
            start_cash=self.start_cash,
            start_crypto=self.start_crypto,
            time_delay=0)
        self.run()

    def execute_orders(self, orders):
        for i, order in enumerate(orders):
            if i == 0: # first order
                assert order.order_type == OrderType.BUY
            self.execute_order(order)

        if self.evaluate_profit_on_last_order and self.num_trades > 0:
            end_price = orders[-1].unit_price
        else:
            if self.num_trades == 0:
                logging.warning("No orders were generated by the chosen strategy.")
            try:
                if self.num_trades > 0:
                    end_price = get_price(self.buy_currency, self.end_time, self.source, self.counter_currency)
                    if orders[-1].order_type == OrderType.BUY:
                        delta_cash = self.cash + end_price * self.crypto
                        if delta_cash > self.invested_on_buy:
                            self.num_profitable_trades += 1
                            buy_sell_pair_profit_percent = (delta_cash - self.invested_on_buy) / self.invested_on_buy * 100
                            self.avg_profit_per_trade_pair += buy_sell_pair_profit_percent
                else:
                    end_price = get_price(self.start_crypto_currency, self.end_time, self.source, self.counter_currency)

            except NoPriceDataException:
                logging.error("No price data found")
                end_price = None

        if self.num_sells != 0:
            self.avg_profit_per_trade_pair /= self.num_sells

        end_crypto_currency = self.buy_currency if self.num_trades > 0 else self.start_crypto_currency

        self.end_cash = self.cash
        self.end_crypto = self.crypto
        self.end_price = end_price
        self.end_crypto_currency = end_crypto_currency

    def run(self):
        self.execute_orders(self.orders)