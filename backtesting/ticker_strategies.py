from strategies import Strategy

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


    def belongs_to_this_strategy(self, signal):
        return self._strategy.belongs_to_this_strategy(signal)

    def get_short_summary(self):
        return self._strategy.get_short_summary()


class TickerBuyAndHold(TickerStrategy):
    '''
    Ticker-based buy & hold strategy.
    '''

    def __init__(self, start_time, end_time):
        self._start_time = start_time
        self._end_time = end_time
        self._bought = False
        self._sold = False

    def process_ticker(self, price_data, signals):
        timestamp = price_data['timestamp']
        if timestamp >= self._start_time and timestamp <= self._end_time and not self._bought:
            if abs(timestamp - self._start_time) > 120:
                logging.warning("Buy and hold BUY: ticker more than 2 mins after start time ({:.2f} mins)!"
                                .format(abs(timestamp - self._start_time)/60))
            self._bought = True
            return "BUY", None
        elif timestamp >= self._end_time and not self._sold:
            logging.warning("Buy and hold SELL: ticker more than 2 mins after end time ({:.2f} mins)!"
                            .format(abs(timestamp - self._end_time) / 60))
            self._sold = True
            return "SELL", None
        else:
            return None, None

    def get_short_summary(self):
        return "Ticker-based buy and hold strategy"
