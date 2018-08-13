from evaluation import Evaluation
from tick_listener import TickListener
from orders import Order, OrderType
from tick_provider_itf_db import TickProviderITFDB


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


    def broadcast_ended(self):
        self._finalize_backtesting()

    @property
    def end_price(self):
        if not self.trading_df.empty:
            return self.trading_df.tail(1)['close_price'].item()
        else:
            return Evaluation.end_price.fget(self)


if __name__ == '__main__':
    from strategies import SignalSignatureStrategy, TickerStrategy, TickerBuyAndHold
    # from ticker_strategies import TickerStrategy, TickerBuyAndHold
    end_time = 1531699200
    start_time = end_time - 60*60*24*70
    start_cash = 10000000
    start_crypto = 0
    source = 0
    resample_period = 60
    transaction_currency = 'BTC'
    counter_currency = 'USDT'
    rsi_strategy = SignalSignatureStrategy(
        source,
        ['rsi_buy_2', 'rsi_sell_2', 'rsi_buy_1', 'rsi_sell_1', 'rsi_buy_3', 'rsi_sell_3']
    )

    benchmark_transaction_currency, benchmark_counter_currency = "BTC", "USDT"
    benchmark_tick_provider = TickProviderITFDB(benchmark_transaction_currency, benchmark_counter_currency, start_time,
                                                end_time)
    benchmark_strategy = TickerBuyAndHold(start_time, end_time)
    benchmark = TickDrivenBacktester(
        tick_provider=benchmark_tick_provider,
        strategy=benchmark_strategy,
        transaction_currency=benchmark_transaction_currency,
        counter_currency=benchmark_counter_currency,
        start_cash=start_cash,
        start_crypto=start_crypto,
        start_time=start_time,
        end_time=end_time,
        source=source,
        resample_period=60,
        verbose=False
    )

    strategy = TickerStrategy(rsi_strategy)
    #strategy = TickerBuyAndHold(start_time, end_time)

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
                                      end_time=end_time,
                                      benchmark_backtest=benchmark)




