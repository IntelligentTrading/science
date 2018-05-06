#from comparative_evaluation import ComparativeEvaluationOneSignal
from data_sources import *
from strategies import *
from utils import *

ordered_columns = ["strategy", "transaction_currency", "counter_currency", "start_cash", "start_crypto", "end_cash",
                   "end_crypto", "num_trades", "profit", "profit_percent", "profit_USDT", "profit_percent_USDT",
                   "buy_and_hold_profit", "buy_and_hold_profit_percent", "buy_and_hold_profit_USDT",
                   "buy_and_hold_profit_percent_USDT", "end_price", "start_time", "end_time",
                   "evaluate_profit_on_last_order", "horizon"]

ordered_columns_condensed = ["strategy", "utilized_signals", "transaction_currency", "counter_currency", "num_trades",
                             "profit_percent", "profit_percent_USDT",
                             "buy_and_hold_profit_percent",
                             "buy_and_hold_profit_percent_USDT", "start_time", "end_time",
                             "evaluate_profit_on_last_order", "horizon",  "num_profitable_trades",
                                "avg_profit_per_trade_pair", "num_sells"]

ordered_columns_no_baseline_condensed = ["strategy", "transaction_currency", "counter_currency", "num_trades",
                             "profit_percent", "profit_percent_USDT", "start_time", "end_time",
                             "evaluate_profit_on_last_order", "horizon"]


class Evaluation:
    def __init__(self, strategy, start_crypto_currency, counter_currency,
                 start_cash, start_crypto, start_time, end_time, evaluate_profit_on_last_order=False, verbose=True):
        self.start_crypto_currency = start_crypto_currency
        self.counter_currency = counter_currency
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.start_time = start_time
        self.end_time = end_time
        self.strategy = strategy
        self.evaluate_profit_on_last_order = evaluate_profit_on_last_order
        self.orders, self.order_signals = strategy.get_orders(start_cash, start_crypto)
        self.execute_orders(self.orders)
        if verbose:
            print(self.get_report())
        if self.end_price is None:
            raise NoPriceDataException()

    def get_start_value_USDT(self):
        try:
            start_value_USDT = convert_value_to_USDT(self.start_cash, self.start_time, self.counter_currency)
            if self.start_crypto > 0 and self.start_crypto_currency is not None:
                start_value_USDT += convert_value_to_USDT(self.start_crypto, self.start_time, self.start_crypto_currency)
            return start_value_USDT
        except NoPriceDataException:
            return None

    def get_end_value_USDT(self):
        try:
            end_value_USDT = convert_value_to_USDT(self.end_cash, self.end_time, self.counter_currency) + \
                             convert_value_to_USDT(self.end_crypto, self.end_time, self.end_crypto_currency)
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
                   self.transaction_currency,
                   self.start_time,
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
        except NoPriceDataException:
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

    def get_num_trades(self):
        return self.num_trades

    def get_orders(self):
        return self.orders

    def format_price_dependent_value(self, value):
        if value is None:
            return "N/A"
        else:
            return value

    def get_report(self, include_order_signals=True):
        output = []
        output.append(str(self.strategy))

        output.append(self.strategy.get_signal_report())
        output.append("--")

        output.append("\n* Order execution log *\n")
        output.append("Start balance: cash = {} {}, crypto = {} {}".format(self.start_cash, self.counter_currency,
                                                                           self.start_crypto, self.start_crypto_currency
                                                                           if self.start_crypto != 0 else ""))

        output.append("Start time: {}\n--".format(datetime_from_timestamp(self.start_time)))
        output.append("--")

        for i, order in enumerate(self.orders):
            output.append(str(order))
            if include_order_signals:
                output.append("   signal: {}".format(self.order_signals[i]))

        output.append("End time: {}".format(datetime_from_timestamp(self.end_time)))
        output.append("\nSummary")
        output.append("--")
        output.append("Number of trades: {}".format(self.num_trades))
        output.append("End cash: {0:.2f} {1}".format(self.end_cash, self.counter_currency))
        output.append("End crypto: {0:.6f} {1}".format(self.end_crypto, self.end_crypto_currency))

        sign = "+" if self.get_profit_value() >= 0 else ""
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

    def execute_orders(self, orders):
        cash = self.start_cash
        crypto = self.start_crypto
        num_trades = 0
        num_profitable_trades = 0
        invested_on_buy = 0
        avg_profit_per_trade_pair = 0
        num_sells = 0

        for i, order in enumerate(orders):
            if i == 0: # first order
                assert order.order_type == OrderType.BUY
                buy_currency = order.transaction_currency

            delta_crypto, delta_cash = order.execute()
            cash += delta_cash
            crypto += delta_crypto
            num_trades += 1
            if order.order_type == OrderType.BUY:
                invested_on_buy = -delta_cash
                buy_currency = order.transaction_currency
            else:
                # the currency we're selling must match the bought currency
                assert order.transaction_currency == buy_currency
                num_sells += 1
                buy_sell_pair_profit_percent = (delta_cash - invested_on_buy) / invested_on_buy * 100
                avg_profit_per_trade_pair += buy_sell_pair_profit_percent
                if buy_sell_pair_profit_percent > 0:
                    num_profitable_trades += 1

        if self.evaluate_profit_on_last_order and num_trades > 0:
            end_price = orders[-1].unit_price
        else:
            if num_trades == 0:
                print("WARNING: no orders were generated by the chosen strategy.")
            try:
                if num_trades > 0:
                    end_price = get_price(buy_currency, self.end_time, self.counter_currency)
                    if orders[-1].order_type == OrderType.BUY:
                        delta_cash = cash + end_price * crypto
                        if delta_cash > invested_on_buy:
                            num_profitable_trades += 1
                            buy_sell_pair_profit_percent = (delta_cash - invested_on_buy) / invested_on_buy * 100
                            avg_profit_per_trade_pair += buy_sell_pair_profit_percent
                else:
                    end_price = get_price(self.start_crypto_currency, self.end_time, self.counter_currency)

            except NoPriceDataException:
                print("Error: no price data found")
                end_price = None

        if num_sells != 0:
            avg_profit_per_trade_pair /= num_sells

        end_crypto_currency = buy_currency if num_trades > 0 else self.start_crypto_currency

        self.num_trades = num_trades
        self.end_cash = cash
        self.end_crypto = crypto
        self.end_price = end_price
        self.num_profitable_trades = num_profitable_trades
        self.avg_profit_per_trade_pair = avg_profit_per_trade_pair
        self.num_sells = num_sells
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
                print("Warning: no price data")
                dictionary["profit"] = "N/A"
                dictionary["profit_percent"] = "N/A"
                dictionary["profit_USDT"] = "N/A"
                dictionary["profit_percent_USDT"] = "N/A"
        return dictionary


if __name__ == "__main__":
    start, end = get_timestamp_range()
    start = 1518523200 # first instance of RSI_Cumulative signal
    end = 1524355233.882 # April 23
    counter_currency = "BTC"
    transaction_currencies = get_currencies_for_signal(counter_currency, "RSI_Cumulative")
    currency_pairs = []
    for transaction_currency in transaction_currencies:
        currency_pairs.append((transaction_currency, counter_currency))


    #eval = ComparativeEvaluationMultiSignal(
    #    signal_types=(SignalType.RSI, SignalType.SMA, SignalType.kumo_breakout, # SignalType.EMA,
    #                  SignalType.RSI_Cumulative),
    #    currency_pairs=currency_pairs,
    #    start_cash=1000, start_crypto=0,
    #    start_time=start, end_time=end,
    #    output_file="test.xlsx",
    #    horizons=(Horizon.any, Horizon.short, Horizon.medium, Horizon.long),
    #    rsi_overbought_values=[70], rsi_oversold_values=[30],
    #    sma_strengths=(Strength.any,))
    #eval.summary_stats("stats_full.xlsx", "stats_profit.xlsx")

    #currency_pairs = (("DGB", "BTC"),)

    #ComparativeEvaluationOneSignal(signal_types=(SignalType.RSI,
    #                                             SignalType.RSI_Cumulative),
    #                               currency_pairs=currency_pairs,
    #                               start_cash=1, start_crypto=0,
    #                               start_time=start, end_time=end,
    #                               output_file="output5.xlsx",
    #                               horizons=(Horizon.any,),
    #                               rsi_overbought_values=[75], rsi_oversold_values=[25])

    #exit(0)
    end = 1525445779.6664
    start = end - 60*60*24*5
    ComparativeEvaluationOneSignal(signal_types=(SignalType.RSI,
                                                 SignalType.RSI_Cumulative, SignalType.kumo_breakout),
                                   currency_pairs=currency_pairs,
                                   start_cash=1, start_crypto=0,
                                   start_time=start, end_time=end,
                                   output_file="output_comparative_v4.xlsx",
                                   horizons=(Horizon.short, Horizon.medium, Horizon.long),
                                   rsi_overbought_values=[70, 75, 80], rsi_oversold_values=[20, 25, 30])
