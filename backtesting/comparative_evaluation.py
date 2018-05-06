import itertools

import pandas as pd

from data_sources import *
from evaluation import ordered_columns_condensed, Evaluation
from signals import SignalType, ALL_SIGNALS
from strategies import Strategy, BuyAndHoldStrategyTimebased, MultiSignalStrategy, BuyAndHoldStrategy, SignalTypedStrategy


class ComparativeEvaluation:

    def __init__(self, buy_signals, sell_signals, num_buy, num_sell, force_same_signal_type,
                 currency_pairs, start_cash, start_crypto, start_time, end_time, output_file, horizons):

        self.buy_signals = buy_signals
        self.sell_signals = sell_signals
        self.num_buy = num_buy
        self.num_sell = num_sell
        self.force_same_signal_type = force_same_signal_type
        self.start_time = start_time
        self.end_time = end_time
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.evaluate_profit_on_last_order = True
        self.buy_first_and_hold = False
        self.output_file = output_file

        self.build_dataframe(currency_pairs, horizons, output_file)

    def build_dataframe(self, currency_pairs, horizons, output_file):
        evaluation_dicts = []
        for (transaction_currency, counter_currency) in currency_pairs:
            for horizon in horizons:
                for dict in self.generate_evaluations(transaction_currency, counter_currency, horizon):
                    evaluation_dicts.append(dict)

        output = pd.DataFrame(evaluation_dicts)
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

    def evaluate(self, signal_set, transaction_currency, counter_currency, horizon):
        source = 0

        signals = get_all_signals(transaction_currency, self.start_time, self.end_time, horizon, counter_currency)
        strategy = SignalTypedStrategy(signal_set, signals)
        baseline = BuyAndHoldStrategyTimebased(self.start_time, self.end_time, transaction_currency,
                                               counter_currency, source, horizon)
        baseline_evaluation = Evaluation(baseline, transaction_currency, counter_currency, self.start_cash,
                                         self.start_crypto, self.start_time, self.end_time,
                                         self.evaluate_profit_on_last_order, verbose=False)
        evaluation = Evaluation(strategy, transaction_currency, counter_currency, self.start_cash,
                                self.start_crypto, self.start_time, self.end_time,
                                self.evaluate_profit_on_last_order, verbose=False)
        return evaluation, baseline_evaluation

    def get_pandas_row_dict(self, evaluation, baseline):
        evaluation_dict = evaluation.to_dictionary()
        baseline_dict = baseline.to_dictionary()
        evaluation_dict["evaluation_object"] = evaluation
        evaluation_dict["buy_and_hold_profit"] = baseline_dict["profit"]
        evaluation_dict["buy_and_hold_profit_percent"] = baseline_dict["profit_percent"]
        evaluation_dict["buy_and_hold_profit_USDT"] = baseline_dict["profit_USDT"]
        evaluation_dict["buy_and_hold_profit_percent_USDT"] = baseline_dict["profit_percent_USDT"]
        return evaluation_dict


    def check_same_type(self, buy_signal_set, sell_signal_set):
        buy_types = set([ALL_SIGNALS[x].signal for x in buy_signal_set])
        sell_types = set([ALL_SIGNALS[x].signal for x in sell_signal_set])
        return buy_types == sell_types and len(buy_types) == 1


    def generate_evaluations(self, transaction_currency, counter_currency, horizon):
        buy_combinations = []
        for i in range(1, self.num_buy+1):
            sample = [list(x) for x in itertools.combinations(self.buy_signals, i)]
            buy_combinations.extend(sample)

        sell_combinations = []
        for i in range(1, self.num_sell+1):
            sample = [list(x) for x in itertools.combinations(self.sell_signals, i)]
            sell_combinations.extend(sample)

        buy_sell_pairs = itertools.product(buy_combinations, sell_combinations)

        for buy, sell in buy_sell_pairs:
            if self.force_same_signal_type and not self.check_same_type(buy, sell):
                continue
            signal_set = list(buy)
            signal_set.extend(list(sell))
            if len(signal_set) == 0:
                continue
            try:
                evaluation, baseline = self.evaluate(signal_set, transaction_currency, counter_currency, horizon)
            except NoPriceDataException:
                continue
            dict = self.get_pandas_row_dict(evaluation, baseline)
            yield dict





if __name__ == "__main__":
    end = 1525445779.6664
    start = end - 60*60*24*5
    counter_currency = "BTC"
    transaction_currencies = get_currencies_for_signal(counter_currency, "RSI_Cumulative")
    currency_pairs = []
    for transaction_currency in transaction_currencies:
        currency_pairs.append((transaction_currency, counter_currency))

    currency_pairs = [("OMG","BTC"),]
    ComparativeEvaluation(buy_signals = ['rsi_buy_3', 'rsi_buy_2', 'rsi_buy_1', 'rsi_cumulat_buy_3', 'rsi_cumulat_buy_2'],
                          sell_signals = ['rsi_sell_3', 'rsi_sell_2', 'rsi_sell_1', 'rsi_cumulat_sell_3', 'rsi_cumulat_sell_2'],
                          num_buy=1,
                          num_sell=1,
                          force_same_signal_type=True,
                          currency_pairs=currency_pairs,
                          start_cash=1, start_crypto=0,
                          start_time=start, end_time=end,
                          output_file="output_new 3.xlsx",
                          horizons=(Horizon.short, Horizon.medium, Horizon.long))
