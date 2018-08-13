import random
from orders import *
from data_sources import *
from signals import *
from backtester_signals import SignalDrivenBacktester
from abc import ABC, abstractmethod
from config import transaction_cost_percents


class Strategy(ABC):
    """
    Base class for trading strategies.
    """

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


class SignalStrategy(Strategy):
    """
    Base class for trading strategies that build orders based on a given list of signals.
    """

    def get_orders(self, signals, start_cash, start_crypto, source, time_delay=0):
        """
        Produces a list of buy-sell orders based on input signals.
        :param signals: A list of input signals.
        :param start_cash: Starting amount of counter_currency (counter_currency is read from first signal).
        :param start_crypto: Starting amount of transaction_currency (transaction_currency is read from first signal).
        :param source: ITF exchange code.
        :param time_delay: Parameter specifying the delay applied when fetching price info (in seconds).
        :return: A list of orders produced by the strategy.
        """
        orders = []
        order_signals = []
        cash = start_cash
        crypto = start_crypto
        buy_currency = None
        for i, signal in enumerate(signals):
            if not self.belongs_to_this_strategy(signal):
                continue

            if self.indicates_sell(signal) and crypto > 0 and signal.transaction_currency == buy_currency:
                price = fetch_delayed_price(signal, source, time_delay)
                order = Order(OrderType.SELL, signal.transaction_currency, signal.counter_currency,
                              signal.timestamp, crypto, price, transaction_cost_percents[source], time_delay, signal.price)
                orders.append(order)
                order_signals.append(signal)
                delta_crypto, delta_cash = order.execute()
                cash = cash + delta_cash
                crypto = crypto + delta_crypto
                assert crypto == 0

            elif self.indicates_buy(signal) and cash > 0:
                price = fetch_delayed_price(signal, source, time_delay)
                buy_currency = signal.transaction_currency
                order = Order(OrderType.BUY, signal.transaction_currency, signal.counter_currency,
                              signal.timestamp, cash, price, transaction_cost_percents[source], time_delay, signal.price)
                orders.append(order)
                order_signals.append(signal)
                delta_crypto, delta_cash = order.execute()
                cash = cash + delta_cash
                crypto = crypto + delta_crypto
                assert cash == 0

        return orders, order_signals

    def evaluate(self, transaction_currency, counter_currency, start_cash, start_crypto, start_time, end_time,
                 source, resample_period, evaluate_profit_on_last_order=False, verbose=True, time_delay=0):
        """
        Builds a signal-based backtester using this strategy.
        See full documentation in SignalDrivenBacktester.
        :return: A SignalDrivenBacktester object.
        """
        return SignalDrivenBacktester(
            strategy=self,
            transaction_currency=transaction_currency,
            counter_currency=counter_currency,
            start_cash=start_cash,
            start_crypto=start_crypto,
            start_time=start_time,
            end_time=end_time,
            source=source,
            resample_period=resample_period,
            evaluate_profit_on_last_order=evaluate_profit_on_last_order,
            verbose=verbose,
            time_delay=time_delay
        )

    # TODO: test and clean this
    @staticmethod
    def get_buy_signals(signals):
        return [signal for signal in signals
                if SignalStrategy.belongs_to_this_strategy(signal) and SignalStrategy.indicates_buy(signal)]

    def get_sell_signals(signals):
        return [signal for signal in signals
                if SignalStrategy.belongs_to_this_strategy(signal) and SignalStrategy.indicates_sell(signal)]



class SignalSignatureStrategy(SignalStrategy):
    """
    A strategy based on ITF signal signatures (see a list of signatures in the field ALL_SIGNALS).
    """
    def __init__(self, signal_set):
        """
        Builds the signal signature strategy.
        :param signal_set: A list of ITF signal signatures to be used in the strategy.
        """
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


class SimpleRSIStrategy(SignalStrategy):
    """
    A strategy based on RSI thresholds.
    """
    def __init__(self, overbought_threshold=80, oversold_threshold=25, signal_type="RSI"):
        """
        Builds the RSI strategy.
        :param overbought_threshold: The RSI overbought threshold.
        :param oversold_threshold: The RSI oversold threshold.
        :param signal_type: signal type: Set to "RSI" or "RSI_Cumulative".
        """
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


class SimpleTrendBasedStrategy(SignalStrategy):
    """
    A strategy that decides on buys and sells based on signal trend information.
    """
    def __init__(self, signal_type):
        """
        Builds the trend-based strategy.
        :param signal_type: The underlying signal type to use with the strategy (e.g. "SMA", "RSI", "ANN").
        """
        self.signal_type = signal_type

    def __str__(self):
        output = []
        output.append("Strategy: trend-based strategy ({})".format(self.signal_type))
        output.append(
            "  description: selling when trend = -1, buying when trend = 1 ")
        return "\n".join(output)

    def get_short_summary(self):
        return "Trend-based strategy, signal: {}, strength: {}, horizon: {}".format(self.signal_type,
                                                                                    self.strength.value,
                                                                                    self.horizon)

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == self.signal_type


class BuyOnFirstSignalAndHoldStrategy(SignalStrategy):
    """
    A wrapper class for SignalStrategies. Buys on first signal, then holds.
    """
    def __init__(self, strategy):
        """
        Builds the strategy.
        :param strategy: The wrapped SignalStrategy.
        """
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

class BuyAndHoldTimebasedStrategy(SignalStrategy):
    """
    Standard buy & hold strategy (signal-driven).
    """
    def __init__(self, start_time, end_time, transaction_currency, counter_currency):
        """
        Builds the buy and hold strategy.
        :param start_time: When to buy transaction_currency.
        :param end_time: When to sell transaction_currency.
        :param transaction_currency: Transaction currency.
        :param counter_currency: Counter currency.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency

    def get_orders(self, signals, start_cash, start_crypto, source, time_delay=0):
        transaction_cost_percent = transaction_cost_percents[source]
        orders = []
        start_price = get_price(self.transaction_currency, self.start_time, source, self.counter_currency)
        end_price = get_price(self.transaction_currency, self.end_time, source, self.counter_currency)
        order = Order(OrderType.BUY, self.transaction_currency, self.counter_currency,
                      self.start_time, start_cash, start_price, transaction_cost_percent)
        orders.append(order)
        delta_crypto, delta_cash = order.execute()
        order = Order(OrderType.SELL, self.transaction_currency, self.counter_currency,
                      self.end_time, delta_crypto, end_price, transaction_cost_percent)
        orders.append(order)
        return orders, []

    def get_short_summary(self):
        return "Buy & hold"

    def get_signal_report(self):
        return self.strategy.get_signal_report()

    def belongs_to_this_strategy(self, signal):
        return False


class StrategyDecision:
    """
    Strategy decisions used by TickerStrategies.
    """
    BUY = "BUY"
    SELL = "SELL"
    IGNORE = None


class TickerStrategy(Strategy):
    """
    A wrapper class that converts signal-based strategies to ticker-based.
    Ticker-based strategies are used by TickDrivenBacktester.
    """

    @abstractmethod
    def process_ticker(self, price_data, signals):
        """

        :param price_data: Pandas row with OHLC data and timestamp.
        :param signals: ITF signals co-ocurring with price tick.
        :return: StrategyDecision.BUY or StrategyDecision.SELL or StrategyDecision.IGNORE
        """
        pass


class TickerWrapperStrategy(TickerStrategy):

    def __init__(self, strategy):
        """
        Builds a ticker-driven strategy out of a regular SignalStrategy.
        :param strategy: The wrapped SignalStrategy.
        """
        self._strategy = strategy

    def process_ticker(self, price_data, signals):
        for i, signal in enumerate(signals):
            if not self._strategy.belongs_to_this_strategy(signal):
                continue
            if self._strategy.indicates_sell(signal):
                return StrategyDecision.SELL, signal
            elif self._strategy.indicates_buy(signal):
                return StrategyDecision.BUY, signal
        return StrategyDecision.IGNORE, None

    def belongs_to_this_strategy(self, signal):
        return self._strategy.belongs_to_this_strategy(signal)

    def get_short_summary(self):
        return self._strategy.get_short_summary()


class TickerBuyAndHold(TickerWrapperStrategy):
    """
    Ticker-based buy & hold strategy (used as baseline for benchmark stats calculation).
    """

    def __init__(self, start_time, end_time):
        """
        Builds the ticker-based but
        :param start_time:
        :param end_time:
        """
        self._start_time = start_time
        self._end_time = end_time
        self._bought = False
        self._sold = False

    def process_ticker(self, price_data, signals):
        """
        Process incoming price data.
        :param price_data: Pandas row with OHLC information and timestamp. Assumed to be pre-filtered so this method
        always receives only one transaction and counter currency
        :param signals: ITF signals, ignored.
        :return: StrategyDecision.BUY or StrategyDecision.SELL or StrategyDecision.IGNORE
        """
        timestamp = price_data['timestamp']
        if timestamp >= self._start_time and timestamp <= self._end_time and not self._bought:
            if abs(timestamp - self._start_time) > 120:
                logging.warning("Buy and hold BUY: ticker more than 2 mins after start time ({:.2f} mins)!"
                                .format(abs(timestamp - self._start_time)/60))
            self._bought = True
            return StrategyDecision.BUY, None
        elif timestamp >= self._end_time and not self._sold:
            logging.warning("Buy and hold SELL: ticker more than 2 mins after end time ({:.2f} mins)!"
                            .format(abs(timestamp - self._end_time) / 60))
            self._sold = True
            return StrategyDecision.SELL, None
        else:
            return StrategyDecision.IGNORE, None

    def get_short_summary(self):
        return "Ticker-based buy and hold strategy"


# TODO: obsolete, clean
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
