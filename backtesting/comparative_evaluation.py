import itertools

import pandas as pd

from data_sources import *
from evaluation import ordered_columns_condensed, Evaluation
from signals import SignalType, ALL_SIGNALS
from strategies import Strategy, BuyAndHoldStrategyTimebased, MultiSignalStrategy, BuyAndHoldStrategy, SignalTypedStrategy


class ComparativeEvaluation:

    def __init__(self, buy_signals, sell_signals, currency_pairs, start_cash,
                 start_crypto, start_time, end_time, output_file, horizons):

        output = None
        evaluate_profit_on_last_order = True

        for (transaction_currency, counter_currency) in currency_pairs:
            dataframe = self.perform_evaluations(buy_signals, sell_signals, horizons, evaluate_profit_on_last_order,
                                                 transaction_currency=transaction_currency,
                                                 counter_currency=counter_currency,
                                                 start_cash=start_cash, start_crypto=start_crypto,
                                                 start_time=start_time, end_time=end_time,
                                                 )
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


    def generate_strategy_and_evaluation(self, signal_set, transaction_currency, counter_currency, start_cash, start_crypto,
                                         start_time, end_time, horizon=Horizon.any, evaluate_profit_on_last_order=True):

        signals = get_all_signals(transaction_currency, start_time, end_time, horizon, counter_currency)
        strategy = SignalTypedStrategy(signal_set, signals)
        baseline = BuyAndHoldStrategyTimebased(start_time, end_time, transaction_currency, counter_currency,
                                               0, horizon)
        baseline_evaluation = Evaluation(baseline, transaction_currency, counter_currency, start_cash,
                                         start_crypto, start_time, end_time, evaluate_profit_on_last_order, False)
        evaluation = Evaluation(strategy, transaction_currency, counter_currency, start_cash,
                                start_crypto, start_time, end_time, evaluate_profit_on_last_order, False)
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
        sell_types = set([ALL_SIGNALS[x].signal for x in buy_signal_set])
        return buy_types == sell_types and len(buy_types) == 1


    def generate_evaluations(self, force_same_type, buy_signals, sell_signals, num_signals_for_buy, num_signals_for_sell, horizon,**params):
        buy_combinations = []
        for i in range(1, num_signals_for_buy+1):
            sample = [list(x) for x in itertools.combinations(buy_signals, i)]
            buy_combinations.extend(sample)

        sell_combinations = []
        for i in range(1, num_signals_for_sell+1):
            sample = [list(x) for x in itertools.combinations(sell_signals, i)]
            sell_combinations.extend(sample)

        buy_sell_pairs = itertools.product(buy_combinations, sell_combinations)

        for buy, sell in buy_sell_pairs:
            if force_same_type and not self.check_same_type(buy, sell):
                continue
            signal_set = list(buy)
            signal_set.extend(list(sell))
            if len(signal_set) == 0:
                continue
            try:
                evaluation, baseline = self.generate_strategy_and_evaluation(signal_set, **params)
            except NoPriceDataException:
                continue
            dict = self.get_pandas_row_dict(evaluation, baseline)
            yield dict


    def perform_evaluations(self, buy_signals, sell_signals, horizons, evaluate_profit_on_last_order, **params):
        evaluations = []
        evaluation_dicts = []

        for horizon in horizons:
            for dict in self.generate_evaluations(True, buy_signals, sell_signals, 2, 1, horizon, **params):
                evaluation_dicts.append(dict)

        dataframe = pd.DataFrame(evaluation_dicts)
        return dataframe


    def generate_and_perform_evaluations(self, buy_signals, sell_signals, horizons, **params):
        evaluations = []
        evaluation_dicts = []

        for horizon in horizons:
            for buy_signal in buy_signals:
                for sell_signal in sell_signals:
                    buy_signature = ALL_SIGNALS[buy_signal]
                    sell_signature = ALL_SIGNALS[sell_signal]

                    if force_same_type and buy_signature.signal != sell_signature.signal:
                        continue
                    signal_set = [buy_signal, sell_signal]
                    try:
                        evaluation, baseline = self.generate_strategy_and_evaluation(signal_set, transaction_currency,
                                                                                        counter_currency, start_cash, start_crypto,
                                                                                        start_time, end_time, horizon,
                                                                                        evaluate_profit_on_last_order,
                                                                                        )
                        evaluations.append((evaluation, baseline))
                        evaluation_dicts.append(self.get_pandas_row_dict(evaluation, baseline))
                        print("Evaluated {}".format(evaluation.get_short_summary()))
                    except NoPriceDataException:
                        print("No price data found for currency, continuing...")
                        continue

        for evaluation, baseline in evaluations:
            print(evaluation.get_report())
            print(baseline.get_short_summary())
            if evaluation.get_profit_percent() != 0:
                tmp = open("output.txt", "w")
                tmp.write(evaluation.get_report())
                tmp.close()

        dataframe = pd.DataFrame(evaluation_dicts)
        return dataframe




if __name__ == "__main__":
    end = 1525445779.6664
    start = end - 60*60*24*5
    counter_currency = "BTC"
    transaction_currencies = get_currencies_for_signal(counter_currency, "RSI_Cumulative")
    currency_pairs = []
    for transaction_currency in transaction_currencies:
        currency_pairs.append((transaction_currency, counter_currency))

    ComparativeEvaluation(buy_signals = ['rsi_buy_3', 'rsi_buy_2', 'ichi_kumo_up'],
                          sell_signals = ['rsi_sell_3','rsi_sell_2', 'ichi_kumo_down'],
                          currency_pairs=currency_pairs,
                          start_cash=1, start_crypto=0,
                          start_time=start, end_time=end,
                          output_file="output_new 2.xlsx",
                          horizons=(Horizon.short, Horizon.medium, Horizon.long),
                          rsi_overbought_values=[70, 75, 80], rsi_oversold_values=[20, 25, 30])
