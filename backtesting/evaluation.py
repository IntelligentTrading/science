from utils import datetime_from_timestamp
from data_sources import *


class Evaluation:
    def __init__(self, strategy, transaction_currency, counter_currency,
                 start_cash, start_crypto, start_time, end_time, evaluate_profit_on_last_order=False, verbose = True):
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.start_time = start_time
        self.end_time = end_time
        self.strategy = strategy
        self.evaluate_profit_on_last_order = evaluate_profit_on_last_order
        self.orders = strategy.get_orders(start_cash, start_crypto)
        self.num_trades, self.end_cash, self.end_crypto, self.end_price = self.execute_orders(self.orders)
        if verbose:
            print("\n".join(self.get_report()))


    def get_start_value_USDT(self):
        start_value_USDT = convert_value_to_USDT(self.start_cash, self.start_time, self.counter_currency) + \
                           convert_value_to_USDT(self.start_crypto, self.start_time, self.transaction_currency)
        return start_value_USDT

    def get_end_value_USDT(self):
        end_value_USDT = convert_value_to_USDT(self.end_cash, self.end_time, self.counter_currency) + \
                         convert_value_to_USDT(self.end_price * self.end_crypto, self.end_time, self.counter_currency)
        return end_value_USDT

    def get_profit_USDT(self):
        return self.get_end_value_USDT() - self.get_start_value_USDT()

    def get_profit_percent_USDT(self):
        return self.get_profit_USDT() / self.get_start_value_USDT() * 100

    def get_start_value(self):
        return self.start_cash + \
               (self.start_crypto * get_price(
                   self.transaction_currency,
                   self.start_time,
                   self.counter_currency) if self.start_crypto > 0 else 0)
                   # because more often than not we start with 0 crypto and at the "beginning of time"

    def get_end_value(self):
        return self.end_cash + self.end_price * self.end_crypto

    def get_profit_value(self):
        return self.get_end_value() - self.get_start_value()

    def get_profit_percent(self):
        return self.get_profit_value()/self.get_start_value()*100

    def get_report(self):
        output = []
        output.append(str(self.strategy))
        output.append("\n* Order execution log *\n")
        output.append("Start balance: cash = {} {}, crypto = {} {}".format(self.start_cash, self.counter_currency,
                                                                           self.start_crypto, self.transaction_currency))

        output.append("Start time: {}\n--".format(datetime_from_timestamp(self.start_time)))
        output.append("--")

        for order in self.orders:
            output.append(str(order))

        output.append("End time: {}".format(datetime_from_timestamp(self.end_time)))
        output.append("\nSummary")
        output.append("--")
        output.append("Number of trades: {}".format(self.num_trades))
        output.append("End cash: {0:.2f} {1}".format(self.end_cash, self.counter_currency))
        output.append("End crypto: {0:.6f} {1}".format(self.end_crypto, self.transaction_currency))

        sign = "+" if self.get_profit_value() >= 0 else ""
        output.append("Total value invested: {} {}".format(self.get_start_value(), self.counter_currency))
        output.append(
            "Total value after investment: {0:.2f} {1} ({2}{3:.2f}%)".format(self.get_end_value(), self.counter_currency,
                                                                             sign, self.get_profit_percent()))
        output.append("Profit: {0:.2f} {1}".format(self.get_profit_value(), self.counter_currency))

        if self.counter_currency != "USDT":
            try:
                sign = "+" if self.get_profit_USDT() >= 0 else ""
                output.append("Total value invested: {:.2f} {} (conversion on {})".format(self.get_start_value_USDT(),
                                                                                          "USDT",
                                                                                          datetime_from_timestamp(
                                                                                              self.start_time)))
                output.append(
                    "Total value after investment: {0:.2f} {1} ({2}{3:.2f}%) (conversion on {4})".format(
                        self.get_end_value_USDT(), "USDT", sign,
                        self.get_profit_percent_USDT(), datetime_from_timestamp(self.end_time)))
                output.append("Profit: {0:.2f} {1}".format(self.get_profit_USDT(), "USDT"))
            except NoPriceDataException:
                output.append("[ -- WARNING: USDT price information not available -- ]")

        return output

    def execute_orders(self, orders):
        cash = self.start_cash
        crypto = self.start_crypto
        num_trades = 0

        for order in orders:
            assert order.transaction_currency == self.transaction_currency
            delta_crypto, delta_cash = order.execute()
            cash += delta_cash
            crypto += delta_crypto
            num_trades += 1

        if self.evaluate_profit_on_last_order and self.num_trades > 0:
            end_price = orders[-1].unit_price
        else:
            if num_trades == 0:
                print("WARNING: no orders were generated by the chosen strategy.")
            end_price = get_price(self.transaction_currency, self.end_time, self.counter_currency)

        end_cash = cash
        end_crypto = crypto

        return num_trades, end_cash, end_crypto, end_price



