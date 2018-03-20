from orders import *


class Strategy:
    def get_orders(self, start_cash, start_crypto):
        pass


class SimpleRSIStrategy(Strategy):
    def __init__(self, rsi_signals, overbought_threshold=80, oversold_threshold=25):
        self.rsi_signals = rsi_signals
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold

    def get_orders(self, start_cash, start_crypto):
        orders = []
        cash = start_cash
        crypto = start_crypto
        for signal in self.rsi_signals:
            if signal.rsi_value >= self.overbought_threshold and crypto > 0:
                orders.append(Order(OrderType.SELL, signal.transaction_currency, signal.counter_currency,
                                    signal.timestamp, crypto, signal.price))
                cash += crypto*signal.price
                crypto = 0
            elif signal.rsi_value <= self.oversold_threshold and cash > 0:
                orders.append(Order(OrderType.BUY, signal.transaction_currency, signal.counter_currency,
                                    signal.timestamp, cash/signal.price, signal.price))
                crypto += cash / signal.price
                cash = 0
        return orders

    def __str__(self):
        output = []
        output.append("Strategy: a simple RSI-based strategy")
        output.append("  description: selling when rsi_value >= overbought_threshold, buying when rsi_value <= oversold threshold ")
        output.append("Strategy settings:")
        output.append("  overbought_threshold = {}".format(self.overbought_threshold))
        output.append("  oversold_threshold = {}".format(self.oversold_threshold))
        return("\n".join(output))
