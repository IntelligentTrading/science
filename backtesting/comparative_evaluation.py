import itertools

import pandas as pd

from data_sources import *
from evaluation import ordered_columns_condensed, Evaluation
from signals import SignalType, ALL_SIGNALS
from strategies import Strategy, BuyAndHoldStrategyTimebased, MultiSignalStrategy, BuyAndHoldStrategy, SignalTypedStrategy, SimpleRSIStrategy
from enum import Enum

class SignalCombinationMode(Enum):
    ANY = 0
    SAME_TYPE = 1
    SAME_TYPE_AND_STRENGTH = 2


class StrategyEvaluationSetBuilder:

    @staticmethod
    def build_from_signal_set(buy_signals, sell_signals, num_buy, num_sell, signal_combination_mode,
                              horizons, start_time, end_time, currency_pairs):
        strategies = []

        buy_combinations = []
        for i in range(1, num_buy+1):
            sample = [list(x) for x in itertools.combinations(buy_signals, i)]
            buy_combinations.extend(sample)

        sell_combinations = []
        for i in range(1, num_sell+1):
            sample = [list(x) for x in itertools.combinations(sell_signals, i)]
            sell_combinations.extend(sample)

        buy_sell_pairs = itertools.product(buy_combinations, sell_combinations)

        for buy, sell in buy_sell_pairs:
            if not StrategyEvaluationSetBuilder.check_signal_combination_mode(buy, sell, signal_combination_mode):
                continue
            signal_set = list(buy)
            signal_set.extend(list(sell))
            if len(signal_set) == 0:
                continue

            for horizon in horizons:
                for transaction_currency, counter_currency in currency_pairs:
                    strategy = SignalTypedStrategy(signal_set, start_time, end_time, horizon, counter_currency,
                                               transaction_currency)
                    strategies.append(strategy)

        return strategies

    @staticmethod
    def check_signal_combination_mode(buy_signal_set, sell_signal_set, signal_combination_mode):
        if signal_combination_mode == SignalCombinationMode.ANY:
            return True
        buy_types = set([ALL_SIGNALS[x].signal for x in buy_signal_set])
        sell_types = set([ALL_SIGNALS[x].signal for x in sell_signal_set])
        if signal_combination_mode == SignalCombinationMode.SAME_TYPE:
            return buy_types == sell_types and len(buy_types) == 1
        elif signal_combination_mode == SignalCombinationMode.SAME_TYPE_AND_STRENGTH:
            buy_strengths = set([ALL_SIGNALS[x].strength for x in buy_signal_set])
            sell_strengths = set([ALL_SIGNALS[x].strength for x in sell_signal_set])

            return buy_types == sell_types and len(buy_types) == 1 and \
                   buy_strengths == sell_strengths and len(buy_strengths) == 1

    @staticmethod
    def build_from_rsi_thresholds(signal_type, overbought_thresholds, oversold_thresholds,
                                  horizons, start_time, end_time, currency_pairs):

        strategies = []
        for overbought in overbought_thresholds:
            for oversold in oversold_thresholds:
                for transaction_currency, counter_currency in currency_pairs:
                    for horizon in horizons:
                        strategy = SimpleRSIStrategy(start_time, end_time, horizon, counter_currency,
                                                     overbought, oversold, transaction_currency)
                        strategies.append(strategy)
        return strategies




class ComparativeEvaluation:

    def __init__(self, strategy_set, start_cash, start_crypto, start_time, end_time, output_file):

        self.strategy_set = strategy_set
        self.start_time = start_time
        self.end_time = end_time
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.evaluate_profit_on_last_order = False
        self.buy_first_and_hold = False
        self.output_file = output_file

        self.build_dataframe(strategy_set, output_file)

    def build_dataframe(self, strategy_set, output_file):
        evaluation_dicts = []
        for strategy in strategy_set:
            dict = self.evaluate(strategy)
            evaluation_dicts.append(dict)

        output = pd.DataFrame(evaluation_dicts)
        #output = output[output.num_trades != 0]  # remove empty trades
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

    def evaluate(self, strategy):
        source = 0
        transaction_currency = strategy.transaction_currency
        counter_currency = strategy.counter_currency
        horizon = strategy.horizon

        baseline = BuyAndHoldStrategyTimebased(self.start_time, self.end_time, transaction_currency,
                                               counter_currency, source, horizon)
        baseline_evaluation = Evaluation(baseline, transaction_currency, counter_currency, self.start_cash,
                                         self.start_crypto, self.start_time, self.end_time,
                                         self.evaluate_profit_on_last_order, verbose=False)
        evaluation = Evaluation(strategy, transaction_currency, counter_currency, self.start_cash,
                                self.start_crypto, self.start_time, self.end_time,
                                self.evaluate_profit_on_last_order, verbose=False)
        return self.get_pandas_row_dict(evaluation, baseline_evaluation)

    def get_pandas_row_dict(self, evaluation, baseline):
        evaluation_dict = evaluation.to_dictionary()
        baseline_dict = baseline.to_dictionary()
        evaluation_dict["evaluation_object"] = evaluation
        evaluation_dict["buy_and_hold_profit"] = baseline_dict["profit"]
        evaluation_dict["buy_and_hold_profit_percent"] = baseline_dict["profit_percent"]
        evaluation_dict["buy_and_hold_profit_USDT"] = baseline_dict["profit_USDT"]
        evaluation_dict["buy_and_hold_profit_percent_USDT"] = baseline_dict["profit_percent_USDT"]
        return evaluation_dict




if __name__ == "__main__":
    end = 1525445779.6664
    start = end - 60*60*24*5
    counter_currency = "BTC"
    transaction_currencies = get_currencies_for_signal(counter_currency, "RSI_Cumulative")
    currency_pairs = []
    for transaction_currency in transaction_currencies:
        currency_pairs.append((transaction_currency, counter_currency))

    start = 1518523200  # first instance of RSI_Cumulative signal
    end = 1524355233.882  # April 23

    currency_pairs = [("AMP","BTC"),]
    buy_signals = ['rsi_buy_3', 'rsi_buy_2', 'rsi_cumulat_buy_2', 'rsi_cumulat_buy_3']
    sell_signals = ['rsi_sell_3', 'rsi_sell_2', 'rsi_cumulat_sell_2', 'rsi_cumulat_sell_3']
    num_buy = 2
    num_sell = 2
    signal_combination_mode = SignalCombinationMode.SAME_TYPE
    currency_pairs = currency_pairs
    start_cash = 1
    start_crypto = 0
    start_time = start
    end_time = end
    output_file = "output new cumulative.xlsx",
    horizons = (Horizon.short, Horizon.medium, Horizon.long)


    strategies = StrategyEvaluationSetBuilder.build_from_signal_set(buy_signals, sell_signals, num_buy, num_sell, signal_combination_mode,
                          horizons, start_time, end_time, currency_pairs)

    strategies = StrategyEvaluationSetBuilder.build_from_rsi_thresholds("RSI", [70,75,80],[20,25,30],horizons, start_time,
                                                                        end_time, currency_pairs)

    ComparativeEvaluation(strategy_set=strategies,
                          start_cash=1, start_crypto=0,
                          start_time=start, end_time=end,
                          output_file="output new cumulative refactored rsi.xlsx"
                          )
    exit(0)

    ComparativeEvaluation(buy_signals=['rsi_buy_3', 'rsi_buy_2','rsi_cumulat_buy_2', 'rsi_cumulat_buy_3'],
                          sell_signals=['rsi_sell_3', 'rsi_sell_2', 'rsi_cumulat_sell_2', 'rsi_cumulat_sell_3'],
                          num_buy=2,
                          num_sell=2,
                          signal_combination_mode=SignalCombinationMode.SAME_TYPE,
                          currency_pairs=currency_pairs,
                          start_cash=1, start_crypto=0,
                          start_time=start, end_time=end,
                          output_file="output new cumulative.xlsx",
                          horizons=(Horizon.short, Horizon.medium, Horizon.long))
