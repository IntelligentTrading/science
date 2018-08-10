from orders import *
from data_sources import *
from signals import *
import operator
import random
from backtester_signals import SignalDrivenBacktester

from signals import SignalType
from abc import ABC, abstractmethod

class Strategy(ABC):
    def __init__(self, source):
        self.source = source

    def get_orders(self, signals, start_cash, start_crypto, transaction_cost_percent=0.0025, time_delay=0):
        orders = []
        order_signals = []
        cash = start_cash
        crypto = start_crypto
        buy_currency = None
        for i, signal in enumerate(signals):
            if not self.belongs_to_this_strategy(signal):
                continue

            if self.indicates_sell(signal) and crypto > 0 and signal.transaction_currency == buy_currency:
                price = self.fetch_delayed_price(signal, time_delay)
                order = Order(OrderType.SELL, signal.transaction_currency, signal.counter_currency,
                              signal.timestamp, crypto, price, transaction_cost_percent, time_delay, signal.price)
                orders.append(order)
                order_signals.append(signal)
                delta_crypto, delta_cash = order.execute()
                cash = cash + delta_cash
                crypto = crypto + delta_crypto
                assert crypto == 0

            elif self.indicates_buy(signal) and cash > 0:
                price = self.fetch_delayed_price(signal, time_delay)
                buy_currency = signal.transaction_currency
                order = Order(OrderType.BUY, signal.transaction_currency, signal.counter_currency,
                              signal.timestamp, cash, price, transaction_cost_percent, time_delay, signal.price)
                orders.append(order)
                order_signals.append(signal)
                delta_crypto, delta_cash = order.execute()
                cash = cash + delta_cash
                crypto = crypto + delta_crypto
                assert cash == 0

        return orders, order_signals

    def fetch_delayed_price(self, signal, time_delay):
        if time_delay != 0:
            return get_price(signal.transaction_currency, signal.timestamp + time_delay, self.source, signal.counter_currency)
        else:
           return signal.price

    def indicates_buy(self, signal):
        return int(float(signal.trend)) == 1

    def indicates_sell(self, signal):
        return int(float(signal.trend)) == -1

    @abstractmethod
    def belongs_to_this_strategy(self, signal):
        pass

    @abstractmethod
    def get_short_summary(self):
        pass

    def get_buy_signals(self):
        buy_signals = []
        for signal in self.signals:
            if self.belongs_to_this_strategy(signal) and self.indicates_buy(signal):
                buy_signals.append(signal)
        return buy_signals

    def get_sell_signals(self):
        sell_signals = []
        for signal in self.signals:
            if self.belongs_to_this_strategy(signal) and self.indicates_sell(signal):
                sell_signals.append(signal)
        return sell_signals

    def get_signal_report(self):
        output = []
        for signal in self.signals:
            if self.belongs_to_this_strategy(signal):
                output.append("{} {}".format("BUY" if self.indicates_buy(signal) else "SELL", str(signal)))
        return "\n".join(output)

    def evaluate(self, transaction_currency, counter_currency, start_cash, start_crypto, start_time, end_time,
                 resample_period, evaluate_profit_on_last_order=False, verbose=True, time_delay=0):
        return SignalDrivenBacktester(
            strategy=self,
            transaction_currency=transaction_currency,
            counter_currency=counter_currency,
            start_cash=start_cash,
            start_crypto=start_crypto,
            start_time=start_time,
            end_time=end_time,
            source=self.source,
            resample_period=resample_period,
            evaluate_profit_on_last_order=evaluate_profit_on_last_order,
            verbose=verbose,
            time_delay=time_delay
        )

    # TODO: obsolete, clean up
    @staticmethod
    def generate_strategy(signal_type, transaction_currency, counter_currency, start_time, end_time, horizon=Horizon.any,
                          strength=Strength.any, rsi_overbought=None, rsi_oversold=None):
        signals = get_signals(signal_type, transaction_currency, start_time, end_time, counter_currency)
        if signal_type == SignalType.RSI:
            signals = get_signals_rsi(transaction_currency, start_time, end_time, rsi_overbought, rsi_oversold, counter_currency)
            strategy = SimpleRSIStrategy(signals, rsi_overbought, rsi_oversold, horizon)
        elif signal_type in (SignalType.kumo_breakout, SignalType.SMA, SignalType.EMA, SignalType.RSI_Cumulative):
            strategy = SimpleTrendBasedStrategy(signals, signal_type, horizon, strength)
        return strategy


class RSITickerStrategy(Strategy):

    def process_ticker(self, price_data, signals):
        for i, signal in enumerate(signals):
            if not self.belongs_to_this_strategy(signal):
                continue
            if self.indicates_sell(signal):
                return "SELL", signal
            elif self.indicates_buy(signal):
                return "BUY", signal

        return None, None

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == 'RSI'


class TickerStrategy(Strategy):
    '''
    A wrapper class that converts all existing strategies to ticker-based.
    '''

    def __init__(self, strategy):
        self._strategy = strategy

    def process_ticker(self, price_data, signals):
        for i, signal in enumerate(signals):
            if not self._strategy.belongs_to_this_strategy(signal):
                continue
            if self._strategy.indicates_sell(signal):
                return "SELL", signal
            elif self._strategy.indicates_buy(signal):
                return "BUY", signal
        return None, None


class SignalSignatureStrategy(Strategy):
    def __init__(self, source, signal_set):
        super().__init__(source)
        self.signal_set = signal_set

    def belongs_to_this_strategy(self, signal):
        return signal.signal_signature in self.signal_set

    def __str__(self):
        output = []
        output.append("Strategy: a simple signal set-based strategy")
        output.append("  description: trading according to signal set {}".format(str(self.signal_set)))
        return "\n".join(output)

    def get_short_summary(self):
        return "Signal-set based strategy, trading according to signal set {}".format(str(self.signal_set))


class SimpleRSIStrategy(Strategy):
    def __init__(self, source, overbought_threshold=80, oversold_threshold=25, signal_type="RSI"):
        super().__init__(source)
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.signal_type = signal_type

    def indicates_sell(self, signal):
        return signal.rsi_value >= self.overbought_threshold

    def indicates_buy(self, signal):
        return signal.rsi_value <= self.oversold_threshold

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == self.signal_type

    def __str__(self):
        output = []
        output.append("Strategy: a simple {}-based strategy".format(self.signal_type))
        output.append("  description: selling when rsi_value >= overbought_threshold, buying when rsi_value <= oversold threshold ")
        output.append("Strategy settings:")
        output.append("  overbought_threshold = {}".format(self.overbought_threshold))
        output.append("  oversold_threshold = {}".format(self.oversold_threshold))
        return "\n".join(output)

    def get_short_summary(self):
        return "{} based strategy, overbought = {}, oversold = {}".format(self.signal_type,
                                                                          self.overbought_threshold,
                                                                          self.oversold_threshold)


class SimpleTrendBasedStrategy(Strategy):
    def __init__(self, source, signal_type):
        super().__init__(source)
        self.signal_type = signal_type

    def __str__(self):
        output = []
        output.append("Strategy: trend-based strategy ({})".format(self.signal_type))
        output.append(
            "  description: selling when trend = -1, buying when trend = 1 ")
        output.append("Strategy settings:")
        output.append("  horizon = {}".format(self.horizon.name))
        return "\n".join(output)

    def get_short_summary(self):
        return "Trend-based strategy, signal: {}, strength: {}, horizon: {}".format(self.signal_type,
                                                                                    self.strength.value,
                                                                                    self.horizon)

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == self.signal_type


class BuyOnFirstSignalAndHoldStrategy(Strategy):
    def __init__(self, source, strategy):
        super().__init__(source)
        self.strategy = strategy

    def get_orders(self, start_cash, start_crypto, time_delay=0):
        orders, order_signals = self.strategy.get_orders(start_cash=start_cash,
                                                         start_crypto=start_crypto, time_delay=time_delay)
        filtered_orders = []
        filtered_signals = []
        for i, order in enumerate(orders):
            if order.order_type == OrderType.BUY:
                filtered_orders.append(order)
                filtered_signals.append(order_signals[i])
                break
        return filtered_orders, filtered_signals

    def get_short_summary(self):
        return "Buy first & hold: {}".format(self.strategy.get_short_summary())

    def get_signal_report(self):
        return self.strategy.get_signal_report()


class BuyAndHoldTimebasedStrategy(Strategy):
    def __init__(self, start_time, end_time, transaction_currency, counter_currency, source, transaction_cost_percent=0.0025):
        super().__init__(source)
        self.start_time = start_time
        self.end_time = end_time
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.transaction_cost_percent = transaction_cost_percent

    def get_orders(self, start_cash, start_crypto, time_delay=0, signals=None):
        orders = []
        start_price = get_price(self.transaction_currency, self.start_time, self.source, self.counter_currency)
        end_price = get_price(self.transaction_currency, self.end_time, self.source, self.counter_currency)
        order = Order(OrderType.BUY, self.transaction_currency, self.counter_currency,
                      self.start_time, start_cash, start_price, self.transaction_cost_percent)
        orders.append(order)
        delta_crypto, delta_cash = order.execute()
        order = Order(OrderType.SELL, self.transaction_currency, self.counter_currency,
                      self.end_time, delta_crypto, end_price, self.transaction_cost_percent)
        orders.append(order)
        return orders, []

    def get_short_summary(self):
        return "Buy & hold"

    def get_signal_report(self):
        return self.strategy.get_signal_report()


class RandomTradingStrategy(SimpleTrendBasedStrategy):

    def __init__(self, max_num_signals, start_time, end_time, transaction_currency, counter_currency, source=0):
        super().__init__(source)
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.max_num_signals = max_num_signals
        self.signals = self.build_signals()
        self.signal_type = "Random"

    def build_signals(self):
        num_signals = random.randint(1, self.max_num_signals)
        prices = get_prices_in_range(self.start_time, self.end_time, self.transaction_currency, self.counter_currency,
                                     self.source)
        selected_prices = prices.sample(num_signals)
        selected_prices = selected_prices.sort_index()
        signals = []

        for timestamp, row in selected_prices.iterrows():
            price = row["price"]
            buy = random.random() < 0.5
            signal = Signal("SMA", 1 if buy else -1, Horizon.any, 3, 3,
                 price/1E8, 0, timestamp, None, self.transaction_currency, self.counter_currency)
            signals.append(signal)

        return signals

    def belongs_to_this_strategy(self, signal):
        return True



class MultiSignalStrategy(Strategy):

    def __init__(self, buying_strategies, selling_strategies, horizon):
        self.buying_strategies = buying_strategies
        self.selling_strategies = selling_strategies
        self.buy_signals = []
        self.sell_signals = []
        self.horizon = horizon

        for buying_strategy in buying_strategies:
            self.buy_signals.extend(buying_strategy.get_buy_signals())

        for selling_strategy in selling_strategies:
            self.sell_signals.extend(selling_strategy.get_sell_signals())

        unsorted_signals = list(self.buy_signals)
        unsorted_signals.extend(self.sell_signals)
        self.signals = sorted(unsorted_signals, key=operator.attrgetter('timestamp'))

    def indicates_sell(self, signal):
        return signal in self.sell_signals

    def indicates_buy(self, signal):
        return signal in self.buy_signals

    def belongs_to_this_strategy(self, signal):
        for buying in self.buying_strategies:
            if buying.belongs_to_this_strategy(signal):
                return True
        for selling in self.selling_strategies:
            if selling.belongs_to_this_strategy(signal):
                return True
        return False

    def __str__(self):
        output = []
        output.append("Strategy: multi-signal strategy")
        output.append("  Buying: ")
        for buying_strategy in self.buying_strategies:
            output.append("  {} ".format(buying_strategy.get_short_summary()))
        output.append("  Selling: ")
        for selling_strategy in self.selling_strategies:
            output.append("  {} ".format(selling_strategy.get_short_summary()))
        return "\n".join(output)

    def get_short_summary(self):
        return "Multi-signal, buying on ({}), selling on ({})".format(
            ", ".join(x.get_short_summary() for x in self.buying_strategies),
            ", ".join(x.get_short_summary() for x in self.selling_strategies))


