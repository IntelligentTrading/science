import itertools

from data_sources import *
from backtester_signals import SignalDrivenBacktester
from config import backtesting_report_columns
from signals import ALL_SIGNALS
from strategies import BuyAndHoldTimebasedStrategy, SignalSignatureStrategy, SimpleRSIStrategy
from enum import Enum

class SignalCombinationMode(Enum):
    ANY = 0
    SAME_TYPE = 1
    SAME_TYPE_AND_STRENGTH = 2


class StrategyEvaluationSetBuilder:

    @staticmethod
    def build_from_signal_set(buy_signals, sell_signals, num_buy, num_sell, signal_combination_mode):
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

            strategy = SignalSignatureStrategy(signal_set)
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
                strategy = SimpleRSIStrategy(overbought, oversold, signal_type)
                strategies.append(strategy)
        return strategies


class ComparativeEvaluation:

    def __init__(self, strategy_set, currency_pairs, resample_periods, source,
                 start_cash, start_crypto, start_time, end_time, output_file, time_delay=0):

        self.strategy_set = strategy_set
        self.currency_pairs = currency_pairs
        self.resample_periods = resample_periods
        self.source = source
        self.start_time = start_time
        self.end_time = end_time
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.evaluate_profit_on_last_order = False
        self.buy_first_and_hold = False
        self.output_file = output_file
        self.time_delay = time_delay

        self.build_dataframe(strategy_set, output_file)

    def build_dataframe(self, strategy_set, output_file):
        evaluation_dicts = []
        for transaction_currency, counter_currency in self.currency_pairs:
            for resample_period in self.resample_periods:
                for strategy in strategy_set:
                    try:
                        dict = self.evaluate(strategy, transaction_currency, counter_currency, resample_period)
                        evaluation_dicts.append(dict)
                    except NoPriceDataException:
                        continue

        output = pd.DataFrame(evaluation_dicts)
        if len(output) == 0:
            print("WARNING: No strategies evaluated.")
            return
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
        output = output[backtesting_report_columns]
        self.dataframe = output
        writer = pd.ExcelWriter(output_file)
        output.to_excel(writer, 'Results')
        writer.save()

    def evaluate(self, strategy, transaction_currency, counter_currency, resample_period):
        print("Evaluating strategy...")

        params = {}
        params['start_time'] = self.start_time
        params['end_time'] = self.end_time
        params['transaction_currency'] = transaction_currency
        params['counter_currency'] = counter_currency
        params['resample_period'] = resample_period
        params['start_cash'] = self.start_cash
        params['start_crypto'] = self.start_crypto
        params['evaluate_profit_on_last_order'] = self.evaluate_profit_on_last_order
        params['verbose'] = False
        params['source'] = self.source



        baseline = BuyAndHoldTimebasedStrategy(self.start_time, self.end_time, transaction_currency,
                                               counter_currency)
        """
        baseline_evaluation = SignalDrivenBacktester(strategy=baseline,
                                                     transaction_currency=transaction_currency,
                                                     counter_currency=counter_currency,
                                                     start_cash=self.start_cash,
                                                     start_crypto=self.start_crypto,
                                                     start_time=self.start_time,
                                                     end_time=self.end_time,
                                                     evaluate_profit_on_last_order=self.evaluate_profit_on_last_order,
                                                     verbose=False)
        evaluation = SignalDrivenBacktester(strategy=strategy,
                                            transaction_currency=transaction_currency,
                                            counter_currency=counter_currency,
                                            start_cash=self.start_cash,
                                            start_crypto=self.start_crypto,
                                            start_time=self.start_time,
                                            end_time=self.end_time,
                                            evaluate_profit_on_last_order=self.evaluate_profit_on_last_order,
                                            verbose=False,
                                            time_delay=self.time_delay)
        """
        baseline_evaluation = SignalDrivenBacktester(strategy=baseline, **params)
        evaluation = SignalDrivenBacktester(strategy=strategy, **params)

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


    def write_comparative_summary(self, summary_path):
        writer = pd.ExcelWriter(summary_path)
        summary = self.results
        # remove empty trades
        summary = summary[summary.num_trades != 0]
        summary.groupby(["strategy", "horizon"]).describe().to_excel(writer, 'Results')
        writer.save()

