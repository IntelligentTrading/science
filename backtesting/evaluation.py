from data_sources import *
from strategies import *
import pandas as pd

ordered_columns = ["strategy", "transaction_currency", "counter_currency", "start_cash", "start_crypto", "end_cash",
                   "end_crypto", "num_trades", "profit", "profit_percent", "profit_USDT", "profit_percent_USDT",
                   "buy_and_hold_profit", "buy_and_hold_profit_percent", "buy_and_hold_profit_USDT",
                   "buy_and_hold_profit_percent_USDT", "end_price", "start_time", "end_time",
                   "evaluate_profit_on_last_order"]

class Evaluation:
    def __init__(self, strategy, transaction_currency, counter_currency,
                 start_cash, start_crypto, start_time, end_time, evaluate_profit_on_last_order=False, verbose=True):
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

    def get_end_cash(self):
        return self.end_cash

    def get_end_crypto(self):
        return self.end_crypto

    def get_end_value(self):
        return self.end_cash + self.end_price * self.end_crypto

    def get_profit_value(self):
        return self.get_end_value() - self.get_start_value()

    def get_profit_percent(self):
        return self.get_profit_value()/self.get_start_value()*100

    def get_num_trades(self):
        return self.num_trades

    def get_orders(self):
        return self.orders

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

    def get_short_summary(self):
        return ("{} \t Invested: {} {}, {} {}\t After investment: {:.2f} {}, {:.2f} {} \t Profit: {}{:.2f}%".format(
            self.strategy.get_short_summary(),
            self.start_cash, self.counter_currency, self.start_crypto, self.transaction_currency,
            self.end_cash, self.counter_currency, self.end_crypto, self.transaction_currency,
            "+" if self.get_profit_percent() >= 0 else "", self.get_profit_percent()))

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

        if self.evaluate_profit_on_last_order and num_trades > 0:
            end_price = orders[-1].unit_price
        else:
            if num_trades == 0:
                print("WARNING: no orders were generated by the chosen strategy.")
            try:
                end_price = get_price(self.transaction_currency, self.end_time, self.counter_currency)
            except NoPriceDataException:
                end_price = -1 # TODO fix this logic

        end_cash = cash
        end_crypto = crypto
        return num_trades, end_cash, end_crypto, end_price

    def to_dictionary(self):
        dictionary = vars(self).copy()
        del dictionary["orders"]
        dictionary["strategy"] = dictionary["strategy"].get_short_summary()

        dictionary["profit"] = self.get_profit_value()
        dictionary["profit_percent"] = self.get_profit_percent()

        try:
            dictionary["profit_USDT"] = self.get_profit_USDT()
            dictionary["profit_percent_USDT"] = self.get_profit_percent_USDT()
        except NoPriceDataException:
            dictionary["profit_USDT"] = "N/A"
            dictionary["profit_percent_USDT"] = "N/A"

        return dictionary


class ComparativeEvaluation:

    def __init__(self, signal_types, currency_pairs, start_cash, start_crypto, start_time, end_time,
                 rsi_overbought_values=None, rsi_oversold_values=None):
        output = None
        for (transaction_currency, counter_currency) in currency_pairs:
            dataframe = self.perform_evaluations(signal_types, transaction_currency, counter_currency, start_cash, start_crypto,
                                     start_time, end_time, rsi_overbought_values, rsi_oversold_values)
            if output is None:
                output = dataframe
            else:
                output = output.append(dataframe)

        writer = pd.ExcelWriter('output.xlsx')
        output.to_excel(writer, 'Sheet1')
        writer.save()


    def generate_evaluation(self, signal_type, transaction_currency, counter_currency, start_cash, start_crypto,
                            start_time, end_time, evaluate_profit_on_last_order=True, rsi_overbought=None, rsi_oversold=None):
        signals = get_signals(signal_type, transaction_currency, start_time, end_time, counter_currency)
        if signal_type == SignalType.RSI:
            strategy = SimpleRSIStrategy(signals, rsi_overbought, rsi_oversold)
        elif signal_type in (SignalType.kumo_breakout, SignalType.SMA, SignalType.EMA):
            strategy = SimpleTrendBasedStrategy(signals, signal_type)
        baseline = BuyAndHoldStrategy(strategy)
        baseline_evaluation = Evaluation(baseline, transaction_currency, counter_currency, start_cash,
                                         start_crypto, start_time, end_time, evaluate_profit_on_last_order, False)
        evaluation = Evaluation(strategy, transaction_currency, counter_currency, start_cash,
                                start_crypto, start_time, end_time, evaluate_profit_on_last_order, False)
        return evaluation, baseline_evaluation

    def get_pandas_row_dict(self, evaluation, baseline):
        evaluation_dict = evaluation.to_dictionary()
        baseline_dict = baseline.to_dictionary()
        evaluation_dict["buy_and_hold_profit"] = baseline_dict["profit"]
        evaluation_dict["buy_and_hold_profit_percent"] = baseline_dict["profit_percent"]
        evaluation_dict["buy_and_hold_profit_USDT"] = baseline_dict["profit_USDT"]
        evaluation_dict["buy_and_hold_profit_percent_USDT"] = baseline_dict["profit_percent_USDT"]
        return evaluation_dict

    def perform_evaluations(self, signal_types, transaction_currency, counter_currency, start_cash, start_crypto,
                            start_time, end_time, rsi_overbought_values, rsi_oversold_values, evaluate_profit_on_last_order=False):
        evaluations = []
        evaluation_dicts = []

        for signal_type in signal_types:
            if signal_type == SignalType.RSI:
                for overbought_threshold in rsi_overbought_values:
                    for oversold_threshold in rsi_oversold_values:
                        evaluation, baseline = self.generate_evaluation(signal_type, transaction_currency,
                                                                        counter_currency, start_cash, start_crypto,
                                                                        start_time, end_time,
                                                                        evaluate_profit_on_last_order,
                                                                        overbought_threshold, oversold_threshold)
                        evaluations.append((evaluation, baseline))
                        evaluation_dicts.append(self.get_pandas_row_dict(evaluation, baseline))
                        print("Evaluated {}".format(evaluation.get_short_summary()))

            else:
                evaluation, baseline = self.generate_evaluation(signal_type, transaction_currency, counter_currency,
                                                                start_cash, start_crypto, start_time, end_time,
                                                                evaluate_profit_on_last_order)
                evaluations.append((evaluation, baseline))
                evaluation_dicts.append(self.get_pandas_row_dict(evaluation, baseline))
                print("Evaluated {}".format(evaluation.get_short_summary()))

        for evaluation, baseline in evaluations:
            print(evaluation.get_short_summary())
            print(baseline.get_short_summary())

        dataframe = pd.DataFrame(evaluation_dicts)
        dataframe = dataframe[ordered_columns]

        return dataframe


if __name__ == "__main__":
    start, end = get_timestamp_range()
    counter_currency = "BTC"
    transaction_currencies = get_currencies_trading_against_counter(counter_currency)
    currency_pairs = []
    for transaction_currency in transaction_currencies:
        currency_pairs.append((transaction_currency, counter_currency))

    eval = ComparativeEvaluation(signal_types=(SignalType.RSI, SignalType.SMA, SignalType.kumo_breakout, SignalType.EMA),
                                 currency_pairs=currency_pairs,
                                 start_cash=1000, start_crypto=0,
                                 start_time=start, end_time=end,
                                 rsi_overbought_values=[70, 75, 80], rsi_oversold_values=[20, 25, 30])
