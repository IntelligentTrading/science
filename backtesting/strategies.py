from orders import *
from data_sources import SignalType


class Strategy:
    def get_orders(self, start_cash, start_crypto):
        pass

    def get_short_summary(self):
        pass


class OneSignalStrategy(Strategy):
    def __init__(self, signals, signal_type):
        self.signals = signals
        self.signal_type = signal_type

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

    def indicates_sell(self, signal):
        pass

    def indicates_buy(self, signal):
        pass



class SimpleRSIStrategy(OneSignalStrategy):
    def __init__(self, signals, overbought_threshold=80, oversold_threshold=25):
        OneSignalStrategy.__init__(self, signals, SignalType.RSI)
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
        return "\n".join(output)

    def get_short_summary(self):
        return "RSI, overbought = {}, oversold = {}".format(self.overbought_threshold, self.oversold_threshold)


class SimpleTrendBasedStrategy(OneSignalStrategy):
    def __init__(self, signals, signal_type):
        OneSignalStrategy.__init__(self, signals, signal_type)

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
        return "\n".join(output)

    def get_short_summary(self):
        return "{} trend-based".format(self.signal_type.value)


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


