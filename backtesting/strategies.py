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


class OneSignalStrategy(Strategy):
    def __init__(self, signals, signal_type, horizon):
        if horizon is not None:
            self.signals = Strategy.filter_based_on_horizon(signals, horizon)
        else:
            self.signals = signals
        self.signal_type = signal_type
        self.horizon = horizon

    def get_orders(self, start_cash, start_crypto):
        orders = []
        cash = start_cash
        crypto = start_crypto
        for signal in self.signals:
            if self.indicates_sell(signal) and crypto > 0:
                orders.append(Order(OrderType.SELL, signal.transaction_currency, signal.counter_currency,
                                    signal.timestamp, crypto, signal.price))
                cash += crypto * signal.price
                crypto = 0
            elif self.indicates_buy(signal) and cash > 0:
                orders.append(Order(OrderType.BUY, signal.transaction_currency, signal.counter_currency,
                                    signal.timestamp, cash / signal.price, signal.price))
                crypto += cash / signal.price
                cash = 0
        return orders

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


class SimpleRSIStrategy(OneSignalStrategy):
    def __init__(self, signals, overbought_threshold=80, oversold_threshold=25, horizon=None):
        OneSignalStrategy.__init__(self, signals, SignalType.RSI, horizon)
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


class SimpleTrendBasedStrategy(OneSignalStrategy):
    def __init__(self, signals, signal_type, horizon = None):
        OneSignalStrategy.__init__(self, signals, signal_type, horizon)

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
        orders = self.strategy.get_orders(start_cash, start_crypto)
        filtered = []
        for order in orders:
            if order.order_type == OrderType.BUY:
                filtered.append(order)
                break
        return filtered

    def get_short_summary(self):
        return "Buy first & hold: {}".format(self.strategy.get_short_summary())


class MultiSignalStrategy(OneSignalStrategy):

    def __init__(self, buying_strategies, selling_strategies):
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
