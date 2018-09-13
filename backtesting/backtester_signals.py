from evaluation import Evaluation
from orders import OrderType
import logging
from data_sources import get_price, get_filtered_signals, NoPriceDataException
from trader import AlternatingBuySellTrading

class SignalDrivenBacktester(Evaluation):

    def __init__(self, signals=None, **kwargs):
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
        :param slippage: Parameter specifying the slippage percentage, applied in the direction of the trade.
        :param signals: A predefined list of signals passed into the strategy. If not supplied, the signals will be
                        pulled from the database.
        """
        super().__init__(**kwargs)
        if signals is None:
            self.signals = get_filtered_signals(
                start_time=self._start_time,
                end_time=self._end_time,
                counter_currency=self._counter_currency,
                transaction_currency=self._transaction_currency,
                source=self._source,
                resample_period=self._resample_period
            )
        else:
            self.signals = signals

        self._buy_currency = self._start_crypto_currency = self._transaction_currency

        trading_simulator = AlternatingBuySellTrading(strategy=self._strategy,
                                                      signals=self.signals,
                                                      start_cash=self._start_cash,
                                                      start_crypto=self._start_crypto,
                                                      source=self._source,
                                                      time_delay=self._time_delay,
                                                      slippage=self._slippage,
                                                      endless_budget=False)

        self.orders = trading_simulator.orders
        self.order_signals = trading_simulator.order_signals
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
            self._write_trading_df_row()
        self._end_crypto_currency = self._buy_currency
        self._finalize_backtesting()


    def run(self):
        self.fill_trading_df(self.orders)
        #self.execute_orders(self.orders)