from orders import *
from data_sources import SignalType
import operator

class Strategy:
    def get_orders(self, start_cash, start_crypto):
        pass

    def get_short_summary(self):
        pass

    @staticmethod
    def filter_based_on_horizon(signals, horizon):
        output = []
        for signal in signals:
            if signal.horizon == horizon:
                output.append(signal)
        return output


class SignalBasedStrategy(Strategy):
    def __init__(self, signals, horizon):
        if horizon is not None:
            self.signals = Strategy.filter_based_on_horizon(signals, horizon)
        else:
            self.signals = signals
        self.horizon = horizon

    def get_orders(self, start_cash, start_crypto):
        orders = []
        order_signals = []
        cash = start_cash
        crypto = start_crypto
        for signal in self.signals:
            if self.indicates_sell(signal) and crypto > 0:
                orders.append(Order(OrderType.SELL, signal.transaction_currency, signal.counter_currency,
                                    signal.timestamp, crypto, signal.price))
                order_signals.append(signal)
                cash += crypto * signal.price
                crypto = 0
            elif self.indicates_buy(signal) and cash > 0:
                orders.append(Order(OrderType.BUY, signal.transaction_currency, signal.counter_currency,
                                    signal.timestamp, cash / signal.price, signal.price))
                order_signals.append(signal)
                crypto += cash / signal.price
                cash = 0
        return orders, order_signals

    def get_buy_signals(self):
        buy_signals = []
        for signal in self.signals:
            if self.indicates_buy(signal):
                buy_signals.append(signal)
        return buy_signals

    def get_sell_signals(self):
        sell_signals = []
        for signal in self.signals:
            if self.indicates_sell(signal):
                sell_signals.append(signal)
        return sell_signals

    def indicates_sell(self, signal):
        pass

    def indicates_buy(self, signal):
        pass

    def get_signal_report(self):
        output = []
        for signal in self.signals:
            output.append("{} {}".format("BUY" if self.indicates_buy(signal) else "SELL", str(signal)))
        return "\n".join(output)


class SimpleRSIStrategy(SignalBasedStrategy):
    def __init__(self, signals, overbought_threshold=80, oversold_threshold=25, horizon=None):
        SignalBasedStrategy.__init__(self, signals, horizon)
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold

    def indicates_sell(self, signal):
        if signal.rsi_value >= self.overbought_threshold:
            return True
        else:
            return False

    def indicates_buy(self, signal):
        if signal.rsi_value <= self.oversold_threshold:
            return True
        else:
            return False

    def __str__(self):
        output = []
        output.append("Strategy: a simple RSI-based strategy")
        output.append("  description: selling when rsi_value >= overbought_threshold, buying when rsi_value <= oversold threshold ")
        output.append("Strategy settings:")
        output.append("  overbought_threshold = {}".format(self.overbought_threshold))
        output.append("  oversold_threshold = {}".format(self.oversold_threshold))
        output.append("  horizon = {}".format(self.horizon if self.horizon is not None else "all"))

        return "\n".join(output)

    def get_short_summary(self):
        return "RSI, overbought = {}, oversold = {}, horizon = {}".format(self.overbought_threshold,
                                                                            self.oversold_threshold,
                                                                            "all" if self.horizon is None
                                                                            else self.horizon)


class SimpleTrendBasedStrategy(SignalBasedStrategy):
    def __init__(self, signals, signal_type, horizon = None):
        SignalBasedStrategy.__init__(self, signals, horizon)
        self.signal_type = signal_type

    def indicates_sell(self, signal):
        if signal.trend == "-1":
            return True
        else:
            return False

    def indicates_buy(self, signal):
        if signal.trend == "1":
            return True
        else:
            return False

    def __str__(self):
        output = []
        output.append("Strategy: a simple {}-based strategy".format(self.signal_type.value))
        output.append(
            "  description: selling when trend = -1, buying when trend = 1 ")
        output.append("  horizon = {}".format(self.horizon if self.horizon is not None else "all"))
        return "\n".join(output)

    def get_short_summary(self):
        return "{} trend-based, horizon = {}".format(self.signal_type.value, "all" if self.horizon is None
                                                                            else self.horizon)


class BuyAndHoldStrategy(Strategy):
    def __init__(self, strategy):
        self.strategy = strategy

    def get_orders(self, start_cash, start_crypto):
        orders, order_signals = self.strategy.get_orders(start_cash, start_crypto)
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


class MultiSignalStrategy(SignalBasedStrategy):

    def __init__(self, buying_strategies, selling_strategies):
        self.buying_strategies = buying_strategies
        self.selling_strategies = selling_strategies
        self.buy_signals = []
        self.sell_signals = []

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
            ",".join(x.get_short_summary() for x in self.buying_strategies),
            ",".join(x.get_short_summary() for x in self.selling_strategies))


