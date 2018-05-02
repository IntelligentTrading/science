import itertools

import pandas as pd

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
        self.num_trades, self.end_cash, self.end_crypto, self.end_price,  self.num_profitable_trades, \
        self.avg_profit_per_trade_pair, self.num_sells, self.end_crypto_currency = self.execute_orders(self.orders)
        if verbose:
            print(self.get_report())

    def get_start_value_USDT(self):
        start_value_USDT = convert_value_to_USDT(self.start_cash, self.start_time, self.counter_currency)
        if self.start_crypto > 0 and self.start_crypto_currency is not None:
            start_value_USDT += convert_value_to_USDT(self.start_crypto, self.start_time, self.start_crypto_currency)
        return start_value_USDT

    def get_end_value_USDT(self):
        end_value_USDT = convert_value_to_USDT(self.end_cash, self.end_time, self.counter_currency) + \
                         convert_value_to_USDT(self.end_crypto, self.end_time, self.end_crypto_currency)
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

        return "\n".join(output)

    def get_short_summary(self):
        return ("{} \t Invested: {} {}, {} {}\t After investment: {:.2f} {}, {:.2f} {} \t Profit: {}{:.2f}%".format(
            self.strategy.get_short_summary(),
            self.start_cash, self.counter_currency, self.start_crypto, self.start_crypto_currency,
            self.end_cash, self.counter_currency, self.end_crypto, self.end_crypto_currency,
            "+" if self.get_profit_percent() >= 0 else "", self.get_profit_percent()))

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

        if num_sells != 0:
            avg_profit_per_trade_pair /= num_sells

        if self.evaluate_profit_on_last_order and num_trades > 0:
            end_price = orders[-1].unit_price
        else:
            if num_trades == 0:
                print("WARNING: no orders were generated by the chosen strategy.")
            try:
                if num_trades > 0:
                    end_price = get_price(buy_currency, self.end_time, self.counter_currency)
                else:
                    end_price = get_price(self.start_crypto_currency, self.end_time, self.counter_currency)

            except NoPriceDataException:
                print("WARNING: no data found for end price.")
                end_price = -1

        end_cash = cash
        end_crypto = crypto
        end_crypto_currency = buy_currency if num_trades > 0 else self.start_crypto_currency
        return num_trades, end_cash, end_crypto, end_price, num_profitable_trades, \
               avg_profit_per_trade_pair, num_sells, end_crypto_currency

    def to_dictionary(self):
        dictionary = vars(self).copy()
        del dictionary["orders"]
        dictionary["strategy"] = dictionary["strategy"].get_short_summary()
        dictionary["utilized_signals"] = ", ".join(get_distinct_signal_types(self.order_signals))
        dictionary["start_time"] = datetime_from_timestamp(dictionary["start_time"])
        dictionary["end_time"] = datetime_from_timestamp(dictionary["end_time"])
        dictionary["profit"] = self.get_profit_value()
        dictionary["profit_percent"] = self.get_profit_percent()
        dictionary["transaction_currency"] = self.end_crypto_currency
        if "horizon" not in vars(self.strategy):
            dictionary["horizon"] = "N/A"
        else:
            dictionary["horizon"] = self.strategy.horizon.name

        try:
            dictionary["profit_USDT"] = self.get_profit_USDT()
            dictionary["profit_percent_USDT"] = self.get_profit_percent_USDT()
        except NoPriceDataException:
            dictionary["profit_USDT"] = "N/A"
            dictionary["profit_percent_USDT"] = "N/A"
        return dictionary


class ComparativeEvaluationOneSignal:

    def __init__(self, signal_types, currency_pairs, start_cash, start_crypto, start_time, end_time, output_file, horizons=(None,),
                 rsi_overbought_values=None, rsi_oversold_values=None):
        output = None
        for (transaction_currency, counter_currency) in currency_pairs:
            dataframe = self.generate_and_perform_evaluations(signal_types, transaction_currency, counter_currency, start_cash, start_crypto,
                                                              start_time, end_time, horizons, rsi_overbought_values, rsi_oversold_values)
            if output is None:
                output = dataframe
            else:
                output = output.append(dataframe)

        output = output[ordered_columns_condensed]
        output = output[output.num_trades != 0]   # remove empty trades
        writer = pd.ExcelWriter(output_file)
        output.to_excel(writer, 'Results')
        writer.save()

    def generate_strategy_and_evaluation(self, signal_type, transaction_currency, counter_currency, start_cash, start_crypto,
                                         start_time, end_time, horizon=Horizon.any, evaluate_profit_on_last_order=True,
                                         rsi_overbought=None, rsi_oversold=None, strength=Strength.any):

        strategy = Strategy.generate_strategy(signal_type, transaction_currency, counter_currency, start_time, end_time, horizon,
                                     strength, rsi_overbought, rsi_oversold)
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

    def generate_and_perform_evaluations(self, signal_types, transaction_currency, counter_currency, start_cash, start_crypto,
                                         start_time, end_time, horizons, rsi_overbought_values, rsi_oversold_values,
                                         evaluate_profit_on_last_order=False):
        evaluations = []
        evaluation_dicts = []

        for horizon in horizons:
            for signal_type in signal_types:
                if signal_type == SignalType.RSI:
                    for overbought_threshold in rsi_overbought_values:
                        for oversold_threshold in rsi_oversold_values:
                            evaluation, baseline = self.generate_strategy_and_evaluation(signal_type, transaction_currency,
                                                                                         counter_currency, start_cash, start_crypto,
                                                                                         start_time, end_time, horizon,
                                                                                         evaluate_profit_on_last_order,
                                                                                         overbought_threshold, oversold_threshold)
                            evaluations.append((evaluation, baseline))
                            evaluation_dicts.append(self.get_pandas_row_dict(evaluation, baseline))
                            print("Evaluated {}".format(evaluation.get_short_summary()))

                else:
                    evaluation, baseline = self.generate_strategy_and_evaluation(signal_type, transaction_currency, counter_currency,
                                                                                 start_cash, start_crypto, start_time, end_time,
                                                                                 horizon, evaluate_profit_on_last_order)
                    evaluations.append((evaluation, baseline))
                    evaluation_dicts.append(self.get_pandas_row_dict(evaluation, baseline))
                    print("Evaluated {}".format(evaluation.get_short_summary()))

        for evaluation, baseline in evaluations:
            print(evaluation.get_short_summary())
            print(baseline.get_short_summary())
            if evaluation.get_profit_percent() != 0:
                tmp = open("output.txt", "w")
                tmp.write(evaluation.get_report())
                tmp.close()

        dataframe = pd.DataFrame(evaluation_dicts)

        return dataframe


class ComparativeEvaluationMultiSignal:

    def __init__(self, signal_types, currency_pairs, start_cash, start_crypto, start_time, end_time, output_file, horizons=(None,),
                 rsi_overbought_values=None, rsi_oversold_values=None, sma_strengths=None):
        output = None
        for (transaction_currency, counter_currency) in currency_pairs:
            dataframe = self.generate_and_perform_evaluations(signal_types, transaction_currency, counter_currency,
                                                              start_cash, start_crypto, start_time, end_time, horizons,
                                                              rsi_overbought_values, rsi_oversold_values,
                                                              sma_strengths)
            if output is None:
                output = dataframe
            else:
                output = output.append(dataframe)

        output = output[output.num_trades != 0]  # remove empty trades

        output = output.sort_values(by=['profit_percent'], ascending=False)
        output.reset_index(inplace=True, drop=True)
        self.results = output
        best_eval = output.iloc[0]["evaluation_object"]
        print("BEST:")
        print(best_eval.get_report(include_order_signals=True))
        # bf = open("best.txt", "w")
        # bf.write(best_eval.get_report(include_order_signals=True))
        # bf.close()
        output = output[ordered_columns_condensed]

        writer = pd.ExcelWriter(output_file)
        output.to_excel(writer, 'Results')
        writer.save()

    def summary_stats(self, out_file_full, out_file_profit):
        writer = pd.ExcelWriter(out_file_full)
        tmp = self.results[ordered_columns_condensed]
        tmp.describe(include='all').to_excel(writer, "test")
        writer.save()

        by_strat = self.results.groupby(["strategy"])
        strategies = by_strat.groups.keys()
        for strat in strategies:
            print("{}: {}".format(strat, len(by_strat.groups[strat])))
        writer = pd.ExcelWriter(out_file_profit)
        by_strat["profit_percent"].describe(include="all").to_excel(writer, 'Results')
        writer.save()

    def get_pandas_row_dict(self, evaluation, baseline):
        evaluation_dict = evaluation.to_dictionary()
        baseline_dict = baseline.to_dictionary()
        evaluation_dict["buy_and_hold_profit"] = baseline_dict["profit"]
        evaluation_dict["buy_and_hold_profit_percent"] = baseline_dict["profit_percent"]
        evaluation_dict["buy_and_hold_profit_USDT"] = baseline_dict["profit_USDT"]
        evaluation_dict["buy_and_hold_profit_percent_USDT"] = baseline_dict["profit_percent_USDT"]
        evaluation_dict["evaluation_object"] = evaluation
        return evaluation_dict

    def generate_and_perform_evaluations(self, signal_types, transaction_currency, counter_currency, start_cash, start_crypto,
                                         start_time, end_time, horizons, rsi_overbought_values, rsi_oversold_values,
                                         sma_strengths=None, evaluate_profit_on_last_order=False):

        evaluation_dicts = []

        for horizon in horizons:
            print(transaction_currency, horizon.name)
            strategies = []
            for signal_type in signal_types:
                if signal_type == SignalType.RSI:
                    for overbought_threshold in rsi_overbought_values:
                        for oversold_threshold in rsi_oversold_values:
                            strategy = Strategy.generate_strategy(signal_type, transaction_currency, counter_currency,
                                                         start_time, end_time, horizon, None, overbought_threshold,
                                                         oversold_threshold)

                            strategies.append(strategy)
                elif signal_type == SignalType.SMA:
                    for strength in sma_strengths:
                        strategy = Strategy.generate_strategy(signal_type, transaction_currency, counter_currency,
                                                              start_time, end_time, horizon, strength)

                        strategies.append(strategy)

                else:
                    strategy = Strategy.generate_strategy(signal_type, transaction_currency, counter_currency,
                                                 start_time, end_time, horizon)
                    strategies.append(strategy)

            combinations = []
            for i in range(1, len(strategies) + 1):
                sample = [list(x) for x in itertools.combinations(strategies, i)]
                combinations.extend(sample)
            buy_sell_pairs = itertools.product(combinations, repeat=2)
            for buy, sell in buy_sell_pairs:
                if len(buy) == 0 or len(sell) == 0:
                    continue
                multi_strat = MultiSignalStrategy(buy, sell, horizon)
                buy_and_hold = BuyAndHoldStrategy(multi_strat)
                evaluation = Evaluation(multi_strat, transaction_currency, counter_currency, start_cash, start_crypto,
                                        start_time, end_time, evaluate_profit_on_last_order, False)
                baseline = Evaluation(buy_and_hold, transaction_currency, counter_currency, start_cash, start_crypto,
                                        start_time, end_time, evaluate_profit_on_last_order, False)
                dict = self.get_pandas_row_dict(evaluation, baseline)
                if len(dict["utilized_signals"].split(",")) == 1:
                    continue
                evaluation_dicts.append(dict)

        dataframe = pd.DataFrame(evaluation_dicts)
        return dataframe


if __name__ == "__main__":
    start, end = get_timestamp_range()
    counter_currency = "BTC"
    transaction_currencies = get_currencies_trading_against_counter(counter_currency)
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

    ComparativeEvaluationOneSignal(signal_types=(SignalType.RSI,
                                                 SignalType.RSI_Cumulative),
                                   currency_pairs=currency_pairs,
                                   start_cash=1, start_crypto=0,
                                   start_time=start, end_time=end,
                                   output_file="output2.xlsx",
                                   horizons=(Horizon.any, Horizon.short, Horizon.medium, Horizon.long),
                                   rsi_overbought_values=[70, 75, 80], rsi_oversold_values=[20, 25, 30])
