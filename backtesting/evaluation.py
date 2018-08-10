from data_sources import *
from orders import *
from utils import *
import logging
from config import transaction_cost_percents
logging.getLogger().setLevel(logging.INFO)
from abc import ABC, abstractmethod


class Evaluation(ABC):
    def __init__(self, strategy, transaction_currency, counter_currency,
                 start_cash, start_crypto, start_time, end_time, source=0,
                 resample_period=60, evaluate_profit_on_last_order=False, verbose=True):
        self._strategy = strategy
        self._transaction_currency = transaction_currency
        self._counter_currency = counter_currency
        self._start_cash = start_cash
        self._start_crypto = start_crypto
        self._start_time = start_time
        self._end_time = end_time
        self._source = source
        self._resample_period = resample_period
        self._evaluate_profit_on_last_order = evaluate_profit_on_last_order
        self._transaction_cost_percent = transaction_cost_percents[source]
        self._verbose = verbose

        # Init backtesting variables
        self._cash = start_cash
        self._crypto = start_crypto

        # TODO: this goes out
        self.num_trades = 0
        self.num_profitable_trades = 0
        self.invested_on_buy = 0
        self.avg_profit_per_trade_pair = 0
        self.num_sells = 0
        self.orders = []
        self.order_signals = []

    @abstractmethod
    def run(self):
        pass


    @property
    def start_value_usdt(self):
        try:
            start_value_USDT = convert_value_to_USDT(self._start_cash, self._start_time,
                                                     self._counter_currency, self._source)
            if self._start_crypto > 0 and self._transaction_currency is not None:
                start_value_USDT += convert_value_to_USDT(self._start_crypto, self._start_time,
                                                          self.start_crypto_currency, self._source)
            return start_value_USDT
        except NoPriceDataException:
            return None

    @property
    def end_value_usdt(self):
        try:
            end_value_USDT = convert_value_to_USDT(self.end_cash, self._end_time, self._counter_currency, self._source) + \
                             convert_value_to_USDT(self.end_crypto, self._end_time, self._end_crypto_currency, self._source)
            return end_value_USDT
        except NoPriceDataException:
            return None

    @property
    def profit_usdt(self):
        if self.start_value_usdt is None or self.end_value_usdt is None:
            return None
        else:
            return self.end_value_usdt - self.start_value_usdt

    @property
    def profit_percent_usdt(self):
        if self.profit_usdt is None or self.start_value_usdt is None:
            return None
        else:
            return self.profit_usdt / self.start_value_usdt * 100

    @property
    def start_value(self):
        try:
            return self._start_cash + \
                   (self._start_crypto * get_price(
                   self.start_crypto_currency,
                   self._start_time,
                   self._source,
                   self._counter_currency) if self._start_crypto > 0 else 0)
                   # because more often than not we start with 0 crypto and at the "beginning of time"
        except NoPriceDataException:
            return None

    @property
    def end_cash(self):
        return self._end_cash

    @property
    def end_crypto(self):
        return self._end_crypto

    @property
    def end_value(self):
        try:
            return self.end_cash + self.end_price * self.end_crypto
        except:
            return None

    @property
    def profit(self):
        if self.end_value is None or self.start_value is None:
            return None
        else:
            return self.end_value - self.start_value

    @property
    def profit_percent(self):
        if self.profit is None or self.start_value is None:
            return None
        else:
            return self.profit/self.start_value*100

    #@property
    #def num_trades(self):
    #    return len(self.orders)

    #@property
    #def end_price(self):
    #    return self.trading_df.tail(1)['close_price'].item()

    def get_orders(self):
        return self._orders

    def _format_price_dependent_value(self, value):
        if value is None:
            return float('nan')
        else:
            return value

    def get_report(self, include_order_signals=True):
        output = []
        output.append(str(self._strategy))

        # output.append(self.strategy.get_signal_report())
        output.append("--")

        output.append("\n* Order execution log *\n")
        output.append("Start balance: cash = {} {}, crypto = {} {}".format(self._start_cash, self._counter_currency,
                                                                           self._start_crypto, self.start_crypto_currency
                                                                           if self._start_crypto != 0 else ""))

        output.append("Start time: {}\n--".format(datetime_from_timestamp(self._start_time)))
        output.append("--")

        for i, order in enumerate(self.orders):
            output.append(str(order))
            if include_order_signals and len(self.order_signals) == len(self.orders): # for buy & hold we don't have signals
                output.append("   signal: {}".format(self.order_signals[i]))

        output.append("End time: {}".format(datetime_from_timestamp(self._end_time)))
        output.append("\nSummary")
        output.append("--")
        output.append("Number of trades: {}".format(self.num_trades))
        output.append("End cash: {0:.2f} {1}".format(self.end_cash, self._counter_currency))
        output.append("End crypto: {0:.6f} {1}".format(self.end_crypto, self._transaction_currency))

        sign = "+" if self.profit != None and self.profit >= 0 else ""
        output.append("Total value invested: {} {}".format(self._format_price_dependent_value(self.start_value),
                                                           self._counter_currency))
        output.append(
            "Total value after investment: {0:.2f} {1} ({2}{3:.2f}%)".format(self._format_price_dependent_value(self.end_value),
                                                                             self._counter_currency,
                                                                             sign,
                                                                             self._format_price_dependent_value(self.profit_percent)))
        output.append("Profit: {0:.2f} {1}".format(self._format_price_dependent_value(self.profit), self._counter_currency))

        if self._counter_currency != "USDT":
            sign = "+" if self.profit_usdt is not None and self.profit_usdt >= 0 else ""
            output.append("Total value invested: {:.2f} {} (conversion on {})".format(
                self._format_price_dependent_value(self.start_value_usdt),
                "USDT",
                datetime_from_timestamp(self._start_time)))
            output.append(
                    "Total value after investment: {0:.2f} {1} ({2}{3:.2f}%) (conversion on {4})".format(
                        self._format_price_dependent_value(self.end_value_usdt), "USDT", sign,
                        self._format_price_dependent_value(self.profit_percent_usdt),
                        datetime_from_timestamp(self._end_time)))
            output.append("Profit: {0:.2f} {1}".format(self._format_price_dependent_value(self.profit_usdt),
                                                       "USDT"))

        return "\n".join(output)

    def get_short_summary(self):
        return ("{} \t Invested: {} {}, {} {}\t After investment: {:.2f} {}, {:.2f} {} \t Profit: {}{:.2f}%".format(
            self._strategy.get_short_summary(),
            self._start_cash, self._counter_currency, self._start_crypto, self.start_crypto_currency,
            self.end_cash, self._counter_currency, self.end_crypto, self.end_crypto_currency,
            "+" if self.profit_percent is not None and self.profit_percent >= 0 else "",
            self._format_price_dependent_value(self.profit_percent)))

    def execute_order(self, order):
        delta_crypto, delta_cash = order.execute()
        self._cash += delta_cash
        self._crypto += delta_crypto
        self.num_trades += 1
        if order.order_type == OrderType.BUY:
            self.invested_on_buy = -delta_cash
            self._buy_currency = order.transaction_currency
        else:
            # the currency we're selling must match the bought currency
            assert order.transaction_currency == self._buy_currency
            self.num_sells += 1
            buy_sell_pair_profit_percent = (delta_cash - self.invested_on_buy) / self.invested_on_buy * 100
            self.avg_profit_per_trade_pair += buy_sell_pair_profit_percent
            if buy_sell_pair_profit_percent > 0:
                self.num_profitable_trades += 1


    def to_dictionary(self):
        dictionary = vars(self).copy()
        # remove trailing underscores
        tmp = {(k[1:] if k.startswith("_") else k): dictionary[k] for k in dictionary.keys()}
        dictionary = tmp
        del dictionary["orders"]
        dictionary["strategy"] = dictionary["strategy"].get_short_summary()
        dictionary["utilized_signals"] = ", ".join(get_distinct_signal_types(self.order_signals))
        dictionary["start_time"] = datetime_from_timestamp(dictionary["start_time"])
        dictionary["end_time"] = datetime_from_timestamp(dictionary["end_time"])

        dictionary["transaction_currency"] = self._end_crypto_currency
        if "horizon" not in vars(self._strategy):
            dictionary["horizon"] = "N/A"
        else:
            dictionary["horizon"] = self._strategy.horizon.name

        if self._end_price == None:
            dictionary["profit"] = "N/A"
            dictionary["profit_percent"] = "N/A"
            dictionary["profit_USDT"] = "N/A"
            dictionary["profit_percent_USDT"] = "N/A"
        else:
            try:
                dictionary["profit"] = self.profit
                dictionary["profit_percent"] = self.profit_percent
                dictionary["profit_USDT"] = self.profit_usdt
                dictionary["profit_percent_USDT"] = self.profit_percent_usdt
            except NoPriceDataException:
                logging.error("No price data!")
                dictionary["profit"] = "N/A"
                dictionary["profit_percent"] = "N/A"
                dictionary["profit_USDT"] = "N/A"
                dictionary["profit_percent_USDT"] = "N/A"
        return dictionary


if __name__ == '__main__':
    from strategies import RSITickerStrategy
    end_time = 1531699200
    start_time = end_time - 60*60*24*7
    start_cash = 10000000
    start_crypto = 0
    strategy = RSITickerStrategy(start_time, end_time, Horizon.short, None)
    evaluation = Evaluation(strategy, 'BTC', 'USDT', start_cash, start_crypto, start_time, end_time)
    #evaluation.simulate_events()
