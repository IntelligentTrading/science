class Signal:
    def __init__(self, signal_type, trend, horizon, strength_value, strength_max,
                 price, price_change, timestamp, rsi_value, transaction_currency, counter_currency):
        self.signal_type = signal_type
        self.trend = trend
        self.horizon = horizon
        self.strength_value = strength_value
        self.strength_max = strength_max
        self.price = price
        self.price_change = price_change
        self.timestamp = timestamp
        self.rsi_value = rsi_value
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency

    def __str__(self):
        return ("{} {} {} {} {} {} {} {} {}".format(
            self.transaction_currency,
            self.trend, self.horizon, self.strength_value, self.strength_max,
            self.price, self.price_change, self.timestamp, self.rsi_value))









