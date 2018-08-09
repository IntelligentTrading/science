from data_sources import *
from orders import *
from utils import *
import logging
from config import transaction_cost_percents
logging.getLogger().setLevel(logging.INFO)


class Evaluation:
    def __init__(self, strategy, transaction_currency, counter_currency,
                 start_cash, start_crypto, start_time, end_time, source=0,
                 resample_period=60, evaluate_profit_on_last_order=False, verbose=True):
        self.strategy = strategy
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.start_time = start_time
        self.end_time = end_time
        self.source = source
        self.resample_period = resample_period
        self.evaluate_profit_on_last_order = evaluate_profit_on_last_order
        self.transaction_cost_percent = transaction_cost_percents[source]
        self.verbose = verbose

        # Init backtesting variables
        self.cash = start_cash
        self.crypto = start_crypto
        self.num_trades = 0
        self.num_profitable_trades = 0
        self.invested_on_buy = 0
        self.avg_profit_per_trade_pair = 0
        self.num_sells = 0

    def get_start_value_USDT(self):
        try:
            start_value_USDT = convert_value_to_USDT(self.start_cash, self.start_time,
                                                     self.counter_currency, self.source)
            if self.start_crypto > 0 and self.transaction_currency is not None:
                start_value_USDT += convert_value_to_USDT(self.start_crypto, self.start_time,
                                                          self.start_crypto_currency, self.source)
            return start_value_USDT
        except NoPriceDataException:
            return None

    def get_end_value_USDT(self):
        try:
            end_value_USDT = convert_value_to_USDT(self.end_cash, self.end_time, self.counter_currency, self.source) + \
                             convert_value_to_USDT(self.end_crypto, self.end_time, self.end_crypto_currency, self.source)
            return end_value_USDT
        except NoPriceDataException:
            return None

    def get_profit_USDT(self):
        end_value = self.get_end_value_USDT()
        start_value = self.get_start_value_USDT()

        if start_value is None or end_value is None:
            return None
        else:
            return end_value - start_value

    def get_profit_percent_USDT(self):
        profit = self.get_profit_USDT()
        start_value = self.get_start_value_USDT()

        if profit is None or start_value is None:
            return None
        else:
            return profit / start_value * 100

    def get_start_value(self):
        try:
            return self.start_cash + \
               (self.start_crypto * get_price(
                   self.start_crypto_currency,
                   self.start_time,
                   self.source,
                   self.counter_currency) if self.start_crypto > 0 else 0)
                   # because more often than not we start with 0 crypto and at the "beginning of time"
        except NoPriceDataException:
            return None

    def get_end_cash(self):
        return self.end_cash

    def get_end_crypto(self):
        return self.end_crypto

    def get_end_value(self):
        try:
            return self.end_cash + self.end_price * self.end_crypto
        except:
            return None

    def get_profit_value(self):
        start_value = self.get_start_value()
        end_value = self.get_end_value()
        if end_value is None or start_value is None:
            return None
        else:
            return end_value - start_value

    def get_profit_percent(self):
        profit = self.get_profit_value()
        start_value = self.get_start_value()
        if profit is None or start_value is None:
            return None
        else:
            return profit/start_value*100

    #@property
    #def num_trades(self):
    #    return len(self.orders)

    #@property
    #def end_price(self):
    #    return self.trading_df.tail(1)['close_price'].item()

    def get_orders(self):
        return self.orders

    def format_price_dependent_value(self, value):
        if value is None:
            return float('nan')
        else:
            return value

    def get_report(self, include_order_signals=True):
        output = []
        output.append(str(self.strategy))

        # output.append(self.strategy.get_signal_report())
        output.append("--")

        output.append("\n* Order execution log *\n")
        output.append("Start balance: cash = {} {}, crypto = {} {}".format(self.start_cash, self.counter_currency,
                                                                           self.start_crypto, self.start_crypto_currency
                                                                           if self.start_crypto != 0 else ""))

        output.append("Start time: {}\n--".format(datetime_from_timestamp(self.start_time)))
        output.append("--")

        for i, order in enumerate(self.orders):
            output.append(str(order))
            if include_order_signals and len(self.order_signals) == len(self.orders): # for buy & hold we don't have signals
                output.append("   signal: {}".format(self.order_signals[i]))

        output.append("End time: {}".format(datetime_from_timestamp(self.end_time)))
        output.append("\nSummary")
        output.append("--")
        output.append("Number of trades: {}".format(self.num_trades))
        output.append("End cash: {0:.2f} {1}".format(self.end_cash, self.counter_currency))
        output.append("End crypto: {0:.6f} {1}".format(self.end_crypto, self.transaction_currency))

        sign = "+" if self.get_profit_value() != None and self.get_profit_value() >= 0 else ""
        output.append("Total value invested: {} {}".format(self.format_price_dependent_value(self.get_start_value()),
                                                           self.counter_currency))
        output.append(
            "Total value after investment: {0:.2f} {1} ({2}{3:.2f}%)".format(self.format_price_dependent_value(self.get_end_value()),
                                                                             self.counter_currency,
                                                                             sign,
                                                                             self.format_price_dependent_value(self.get_profit_percent())))
        output.append("Profit: {0:.2f} {1}".format(self.format_price_dependent_value(self.get_profit_value()), self.counter_currency))

        if self.counter_currency != "USDT":
            sign = "+" if self.get_profit_USDT() is not None and self.get_profit_USDT() >= 0 else ""
            output.append("Total value invested: {:.2f} {} (conversion on {})".format(
                self.format_price_dependent_value(self.get_start_value_USDT()),
                "USDT",
                datetime_from_timestamp(self.start_time)))
            output.append(
                    "Total value after investment: {0:.2f} {1} ({2}{3:.2f}%) (conversion on {4})".format(
                        self.format_price_dependent_value(self.get_end_value_USDT()), "USDT", sign,
                        self.format_price_dependent_value(self.get_profit_percent_USDT()),
                        datetime_from_timestamp(self.end_time)))
            output.append("Profit: {0:.2f} {1}".format(self.format_price_dependent_value(self.get_profit_USDT()),
                                                       "USDT"))

        return "\n".join(output)

    def get_short_summary(self):
        return ("{} \t Invested: {} {}, {} {}\t After investment: {:.2f} {}, {:.2f} {} \t Profit: {}{:.2f}%".format(
            self.strategy.get_short_summary(),
            self.start_cash, self.counter_currency, self.start_crypto, self.start_crypto_currency,
            self.end_cash, self.counter_currency, self.end_crypto, self.end_crypto_currency,
            "+" if self.get_profit_percent() is not None and self.get_profit_percent() >= 0 else "",
            self.format_price_dependent_value(self.get_profit_percent())))

    def execute_order(self, order):
        delta_crypto, delta_cash = order.execute()
        self.cash += delta_cash
        self.crypto += delta_crypto
        self.num_trades += 1
        if order.order_type == OrderType.BUY:
            self.invested_on_buy = -delta_cash
            self.buy_currency = order.transaction_currency
        else:
            # the currency we're selling must match the bought currency
            assert order.transaction_currency == self.buy_currency
            self.num_sells += 1
            buy_sell_pair_profit_percent = (delta_cash - self.invested_on_buy) / self.invested_on_buy * 100
            self.avg_profit_per_trade_pair += buy_sell_pair_profit_percent
            if buy_sell_pair_profit_percent > 0:
                self.num_profitable_trades += 1


    def execute_orders(self, orders):

        for i, order in enumerate(orders):
            if i == 0: # first order
                assert order.order_type == OrderType.BUY
            self.execute_order(order)

        if self.evaluate_profit_on_last_order and self.num_trades > 0:
            end_price = orders[-1].unit_price
        else:
            if self.num_trades == 0:
                logging.warning("No orders were generated by the chosen strategy.")
            try:
                if self.num_trades > 0:
                    end_price = get_price(self.buy_currency, self.end_time, self.source, self.counter_currency)
                    if orders[-1].order_type == OrderType.BUY:
                        delta_cash = self.cash + end_price * self.crypto
                        if delta_cash > self.invested_on_buy:
                            self.num_profitable_trades += 1
                            buy_sell_pair_profit_percent = (delta_cash - self.invested_on_buy) / self.invested_on_buy * 100
                            self.avg_profit_per_trade_pair += buy_sell_pair_profit_percent
                else:
                    end_price = get_price(self.start_crypto_currency, self.end_time, self.source, self.counter_currency)

            except NoPriceDataException:
                logging.error("No price data found")
                end_price = None

        if self.num_sells != 0:
            self.avg_profit_per_trade_pair /= self.num_sells

        end_crypto_currency = self.buy_currency if self.num_trades > 0 else self.start_crypto_currency

        self.end_cash = self.cash
        self.end_crypto = self.crypto
        self.end_price = end_price
        self.num_profitable_trades = self.num_profitable_trades
        self.avg_profit_per_trade_pair = self.avg_profit_per_trade_pair
        self.num_sells = self.num_sells
        self.end_crypto_currency = end_crypto_currency


    def to_dictionary(self):
        dictionary = vars(self).copy()
        del dictionary["orders"]
        dictionary["strategy"] = dictionary["strategy"].get_short_summary()
        dictionary["utilized_signals"] = ", ".join(get_distinct_signal_types(self.order_signals))
        dictionary["start_time"] = datetime_from_timestamp(dictionary["start_time"])
        dictionary["end_time"] = datetime_from_timestamp(dictionary["end_time"])

        dictionary["transaction_currency"] = self.end_crypto_currency
        if "horizon" not in vars(self.strategy):
            dictionary["horizon"] = "N/A"
        else:
            dictionary["horizon"] = self.strategy.horizon.name

        if self.end_price == None:
            dictionary["profit"] = "N/A"
            dictionary["profit_percent"] = "N/A"
            dictionary["profit_USDT"] = "N/A"
            dictionary["profit_percent_USDT"] = "N/A"
        else:
            try:
                dictionary["profit"] = self.get_profit_value()
                dictionary["profit_percent"] = self.get_profit_percent()
                dictionary["profit_USDT"] = self.get_profit_USDT()
                dictionary["profit_percent_USDT"] = self.get_profit_percent_USDT()
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
