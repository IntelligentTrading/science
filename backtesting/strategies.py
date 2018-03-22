from orders import *
from data_sources import SignalType


class OneSignalStrategy:
    def __init__(self, signals):
        self.signals = signals

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
        OneSignalStrategy.__init__(self, signals)
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
        return("\n".join(output))


class SimpleTrendBasedStrategy(OneSignalStrategy):
    def __init__(self, signals):
        OneSignalStrategy.__init__(self, signals)

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


class RSIBuyAndHoldStrategy(SimpleRSIStrategy):
    def __init__(self, rsi_signals, overbought_threshold=80, oversold_threshold=25):
        SimpleRSIStrategy.__init__(self, rsi_signals, overbought_threshold, oversold_threshold)

    def get_orders(self, start_cash, start_crypto):
        orders = SimpleRSIStrategy.get_orders(self, start_cash, start_crypto)
        filtered = []
        for order in orders:
            if order.order_type == OrderType.BUY:
                filtered.append(order)
                break
        return filtered


