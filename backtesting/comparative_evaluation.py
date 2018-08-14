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
    """
    Builds a set of strategies for comparative backtest by combining signals.
    """

    @staticmethod
    def build_from_signal_set(buy_signals, sell_signals, num_buy, num_sell, signal_combination_mode):
        """
        Builds a set of strategies based on a set of buy signals and a set of sell signals.
        :param buy_signals: A list of signal signatures to be used for buying (see ALL_SIGNALS).
        :param sell_signals: A list of signal signatures to be used for selling (see ALL_SIGNALS).
        :param num_buy: Number of different signals a strategy should use to buy.
        :param num_sell: Number of different signals a strategy should use to sell.
        :param signal_combination_mode: Indicates how to build strategies.
               Options: SignalCombinationMode.ANY: the signal type of buy and sell signals need not match (e.g. RSI buy, SMA sell)
                        SignalCombinationMode.SAME_TYPE: the signal type of buy and sell signals must match.
                        SignalCombinationMode.SAME_TYPE_AND_STRENGTH: the signal type and strength of buy and sell signals must match.
        :return: A list of all SignalSignature strategies satisfying the building constraints.
        """
        strategies = []

        # create all possible strategies
        buy_combinations = []
        for i in range(1, num_buy+1):
            sample = [list(x) for x in itertools.combinations(buy_signals, i)]
            buy_combinations.extend(sample)

        sell_combinations = []
        for i in range(1, num_sell+1):
            sample = [list(x) for x in itertools.combinations(sell_signals, i)]
            sell_combinations.extend(sample)

        buy_sell_pairs = itertools.product(buy_combinations, sell_combinations)

        # filter out those not satisfying the combination criteria
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
        """
        Checks if a particular set of buy signals and set signals satisfies the combination criterion.
        :param buy_signal_set: A set of buy signal signatures.
        :param sell_signal_set: A set of sell signal signatures.
        :param signal_combination_mode: Instance of SignalCombinationMode.
        :return: True if the combination of buy and sell signals is valid according to
        """
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
    def build_from_rsi_thresholds(signal_type, overbought_thresholds, oversold_thresholds):
        """
        Builds a set of strategies as a Cartesian product of lists of overbought and oversold thresholds.
        :param signal_type: "RSI" or "RSI_Cumulative".
        :param overbought_thresholds: A list of overbought thresholds.
        :param oversold_thresholds: A list of oversold thresholds.
        :return: A list of strategies.
        """

        strategies = []
        for overbought in overbought_thresholds:
            for oversold in oversold_thresholds:
                strategy = SimpleRSIStrategy(overbought, oversold, signal_type)
                strategies.append(strategy)
        return strategies


class ComparativeEvaluation:
    """
    Comparatively backtests a set of strategies on given currency pairs, resample periods and exchanges.
    """

    def __init__(self, strategy_set, currency_pairs, resample_periods, sources,
                 start_cash, start_crypto, start_time, end_time, output_file, time_delay=0):

        self.strategy_set = strategy_set
        self.currency_pairs = currency_pairs
        self.resample_periods = resample_periods
        self.sources = sources
        self.start_time = start_time
        self.end_time = end_time
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.evaluate_profit_on_last_order = False
        self.buy_first_and_hold = False
        self.output_file = output_file
        self.time_delay = time_delay

        results_df = self.build_dataframe(output_file)
        writer = pd.ExcelWriter(output_file)
        results_df.to_excel(writer, 'Results')
        writer.save()

    def build_dataframe(self):
        evaluation_dicts = []
        for (transaction_currency, counter_currency), resample_period, source, strategy in \
                itertools.product(self.currency_pairs, self.resample_periods, self.sources, self.strategy_set):
            try:
                evaluation_dicts.append(
                    self.evaluate(strategy, transaction_currency, counter_currency, resample_period, source))
            except NoPriceDataException:
                continue

        output = pd.DataFrame(evaluation_dicts)
        if len(output) == 0:
            logging.warning("No strategies evaluated.")
            return

        # sort results by profit
        output = output.sort_values(by=['profit_percent'], ascending=False)

        # drop index
        output.reset_index(inplace=True, drop=True)

        # save full results
        self.results = output

        # find the backtest of the best performing strategy
        best_strat_backtest = output.iloc[0]["evaluation_object"]
        logging.info("Best performing strategy:")
        logging.info(best_strat_backtest.get_report(include_order_signals=True))

        # filter so that only report columns remain
        output = output[backtesting_report_columns]
        return output

    def evaluate(self, strategy, transaction_currency, counter_currency, resample_period, source):
        logging.info("Evaluating strategy {}...", strategy.get_short_summary())

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
        params['source'] = source

        baseline = BuyAndHoldTimebasedStrategy(self.start_time, self.end_time, transaction_currency, counter_currency)
        baseline_evaluation = SignalDrivenBacktester(strategy=baseline, **params)
        evaluation = SignalDrivenBacktester(strategy=strategy, **params)

        return self._create_row_dict(evaluation, baseline_evaluation)

    def _create_row_dict(self, evaluation, baseline):
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

