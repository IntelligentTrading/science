from evaluation import Evaluation
from tick_listener import TickListener
from tick_provider_itf_db import TickProviderITFDB
from config import INF_CASH, INF_CRYPTO
from strategies import BuyAndHoldTimebasedStrategy
from order_generator import OrderGenerator
from utils import datetime_from_timestamp


class TickDrivenBacktester(Evaluation, TickListener):

    def __init__(self, tick_provider, **kwargs):
        super().__init__(**kwargs)
        self.tick_provider = tick_provider
        # reevaluate infinite bank
        self._reevaluate_inf_bank()

        self.run()

    def run(self):
        # register at tick provider
        self.tick_provider.add_listener(self)
        # ingest ticks
        self.tick_provider.run()

        # the provider will call the broadcast_ended() method when no ticks remain

    def process_event(self, price_data, signals_now):
        if price_data.Index < self._start_time or price_data.Index > self._end_time:
            return

        self._current_timestamp = price_data.Index
        self._current_price = price_data.close_price

        decision = self._strategy.get_decision(self._current_timestamp, self._current_price, signals_now)
        order = self._order_generator.generate_order(decision)
        if order is not None:
            self.orders.append(order)
            self.order_signals.append(decision.signal)
            self.execute_order(order)

        self._current_order = order
        self._current_signal = decision.signal

        self._write_trading_df_row()

    def broadcast_ended(self):
        self._end_crypto_currency = self._transaction_currency
        self._finalize_backtesting()

    @property
    def end_price(self):
        if not self.trading_df.empty:
            return self.trading_df.tail(1)['close_price'].item()
        else:
            return Evaluation.end_price.fget(self)

    @staticmethod
    def build_benchmark(transaction_currency, counter_currency, start_cash, start_crypto, start_time, end_time,
                        source, tick_provider=None, time_delay=0, slippage=0):
        if tick_provider is None:
            tick_provider = TickProviderITFDB(transaction_currency,
                                                    counter_currency,
                                                    start_time,
                                                    end_time)
        benchmark_strategy = BuyAndHoldTimebasedStrategy(start_time, end_time, transaction_currency, counter_currency,
                                                         source=0)
        benchmark_order_generator = OrderGenerator.ALTERNATING

        benchmark = TickDrivenBacktester(
                tick_provider=tick_provider,
                strategy=benchmark_strategy,
                transaction_currency=transaction_currency,
                counter_currency=counter_currency,
                start_cash=start_cash,
                start_crypto=start_crypto,
                start_time=start_time,
                end_time=end_time,
                source=source,
                resample_period=None,
                verbose=False,
                time_delay=time_delay,
                slippage=slippage,
                order_generator=benchmark_order_generator
            )
        return benchmark


if __name__ == '__main__':
    from strategies import SignalSignatureStrategy, TickerWrapperStrategy

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
        ['rsi_buy_2', 'rsi_sell_2', 'rsi_buy_1', 'rsi_sell_1', 'rsi_buy_3', 'rsi_sell_3']
    )

    from order_generator import OrderGenerator


    benchmark = None
    build_benchmark = True

    from strategies import BuyAndHoldTimebasedStrategy
    if build_benchmark:
        benchmark_transaction_currency, benchmark_counter_currency = "BTC", "USDT"
        benchmark_tick_provider = TickProviderITFDB(benchmark_transaction_currency, benchmark_counter_currency, start_time,
                                                    end_time)
        benchmark_strategy = BuyAndHoldTimebasedStrategy(start_time, end_time, benchmark_transaction_currency, benchmark_counter_currency,
                                                         source=0)

        benchmark = TickDrivenBacktester(
            tick_provider=benchmark_tick_provider,
            strategy=benchmark_strategy,
            transaction_currency=benchmark_transaction_currency,
            counter_currency=benchmark_counter_currency,
            start_cash=INF_CASH, #tart_cash,
            start_crypto=INF_CRYPTO, #start_crypto,
            start_time=start_time,
            end_time=end_time,
            source=source,
            resample_period=60,
            verbose=True,
            time_delay=0,
            slippage=0,
            order_generator=OrderGenerator.ALTERNATING
        )

    strategy = TickerWrapperStrategy(rsi_strategy)
    #strategy = TickerBuyAndHold(start_time, end_time)

    # supply ticks from the ITF DB
    tick_provider = TickProviderITFDB(transaction_currency, counter_currency, start_time, end_time)

    from config import INF_CASH, INF_CRYPTO
    # create a new tick based backtester
    evaluation = TickDrivenBacktester(tick_provider=tick_provider,
                                      strategy=rsi_strategy,
                                      transaction_currency='BTC',
                                      counter_currency='USDT',
                                      start_cash=INF_CASH,
                                      start_crypto=INF_CRYPTO,
                                      start_time=start_time,
                                      end_time=end_time,
                                      benchmark_backtest=benchmark,
                                      time_delay=0,
                                      slippage=0,
                                      order_generator=OrderGenerator.POSITION_BASED
                                      )

    #evaluation.to_excel("test.xlsx")
    evaluation.plot_cumulative_returns()




