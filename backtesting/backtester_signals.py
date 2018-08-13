from evaluation import Evaluation
from orders import OrderType
import logging
from data_sources import get_price, get_filtered_signals, NoPriceDataException


class SignalDrivenBacktester(Evaluation):

    def __init__(self, strategy, transaction_currency, counter_currency,
                 start_cash, start_crypto, start_time, end_time, source=0,
                 resample_period=60, evaluate_profit_on_last_order=False, verbose=True, time_delay=0):
        """
        :param strategy: Backtested strategy (instance of SignalStrategy).        
        :param transaction_currency: Transaction currency for which we're backtesting.
        :param counter_currency: Counter currency for which we're backtesting.
        :param start_cash: Starting amount of counter_currency.
        :param start_crypto: Starting amount of transaction_currency.
        :param start_time: Beginning of timeframe for which signals are fetched (UTC timestamp).
        :param end_time: End of timeframe for which signals are fetched (UTC timestamp).
        :param source: ITF exchange code.
        :param resample_period: Duration of 1 candle (minutes).
        :param evaluate_profit_on_last_order: Evaluate gains at the time of the last order (vs. at end_time).
        :param verbose: Produce verbose output.
        :param time_delay: Parameter specifying the delay applied when fetching price info (in seconds).
        """


        super().__init__(strategy, transaction_currency, counter_currency,
                 start_cash, start_crypto, start_time, end_time, source,
                 resample_period, evaluate_profit_on_last_order, verbose)
        self.signals = get_filtered_signals(start_time=start_time, end_time=end_time, counter_currency=counter_currency,
                                            transaction_currency=transaction_currency,
                                            source=source)
        self._buy_currency = self._start_crypto_currency = self._transaction_currency
        self.orders, self.order_signals = self._strategy.get_orders(
            signals = self.signals,
            start_cash=self._start_cash,
            start_crypto=self._start_crypto,
            source=self._source,
            time_delay=time_delay)
        self.run()


    def fill_trading_df(self, orders):
        for i, order in enumerate(orders):
            if i == 0: # first order
                assert order.order_type == OrderType.BUY
                self._start_crypto_currency = self._buy_currency = order.transaction_currency

            if order.order_type == OrderType.BUY:
                self._buy_currency = order.transaction_currency
            elif order.order_type == OrderType.SELL:
                assert order.transaction_currency == self._buy_currency
            self.execute_order(order)
            self._current_timestamp = order.timestamp
            self._current_price = order.unit_price
            self._current_order = order
            self._current_signal = self.order_signals[i] if len(self.order_signals) > 0 else None
            self._write_to_trading_df()
        self._end_crypto_currency = self._buy_currency
        self._finalize_backtesting()


    def execute_orders(self, orders):
        for i, order in enumerate(orders):
            if i == 0: # first order
                assert order.order_type == OrderType.BUY
            self.execute_order(order)

        if self._evaluate_profit_on_last_order and self._num_trades > 0:
            end_price = orders[-1].unit_price
        else:
            if self._num_trades == 0:
                logging.warning("No orders were generated by the chosen strategy.")
            try:
                if self._num_trades > 0:
                    end_price = get_price(self._buy_currency, self._end_time, self._source, self._counter_currency)
                    if orders[-1].order_type == OrderType.BUY:
                        delta_cash = self._cash + end_price * self._crypto
                        if delta_cash > self.invested_on_buy:
                            self.num_profitable_trades += 1
                            buy_sell_pair_profit_percent = (delta_cash - self.invested_on_buy) / self.invested_on_buy * 100
                            self.avg_profit_per_trade_pair += buy_sell_pair_profit_percent
                else:
                    end_price = get_price(self.start_crypto_currency, self._end_time, self._source, self._counter_currency)

            except NoPriceDataException:
                logging.error("No price data found")
                end_price = None

        if self._num_sells != 0:
            self.avg_profit_per_trade_pair /= self._num_sells

        self._end_crypto_currency = self._buy_currency if self._num_trades > 0 else self._transaction_currency

        self._finalize_backtesting()

    def run(self):
        self.fill_trading_df(self.orders)
        #self.execute_orders(self.orders)