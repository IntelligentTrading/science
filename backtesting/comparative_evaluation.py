import itertools
import numpy as np
from data_sources import *
from backtester_signals import SignalDrivenBacktester
from config import backtesting_report_columns
from signals import ALL_SIGNALS
from strategies import BuyAndHoldTimebasedStrategy, SignalSignatureStrategy, SimpleRSIStrategy
from enum import Enum
from config import COINMARKETCAP_TOP_20_ALTS

import pandas.io.formats.excel

pandas.io.formats.excel.header_style = None


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
    Uses SignalDrivenBacktester.
    """

    def __init__(self, strategy_set, counter_currencies, resample_periods, sources,
                 start_cash, start_crypto, start_time, end_time, output_file=None, time_delay=0, debug=False):

        self.strategy_set = sorted(strategy_set)
        self.counter_currencies = counter_currencies
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

        self._run_backtests(debug)
        self._report = ComparativeReportBuilder(self.backtests, self.baselines)
        best_backtest = self._report.get_best_performing_backtest()
        logging.info(best_backtest.get_report())
        if output_file is not None:
            self._report.write_summary(output_file)

    def _run_backtests(self, debug):
        self.backtests = []
        self.baselines = []
        transaction_currencies_cache = {}

        for strategy, counter_currency, resample_period, source in \
                itertools.product(self.strategy_set, self.counter_currencies, self.resample_periods, self.sources):

            # performance optimization, ensure we get each list of transaction_currencies only once
            # per counter and source
            if (counter_currency, source) not in transaction_currencies_cache:
                transaction_currencies_cache[(counter_currency, source)] \
                    = get_currencies_trading_against_counter(counter_currency, source)
            transaction_currencies = transaction_currencies_cache[(counter_currency, source)]

            for transaction_currency in transaction_currencies:
                try:
                    backtest, baseline = self._evaluate(strategy, transaction_currency, counter_currency, resample_period, source)
                    self.backtests.append(backtest)
                    self.baselines.append(baseline)
                    if debug:
                        break
                except NoPriceDataException:
                    continue


    def _evaluate(self, strategy, transaction_currency, counter_currency, resample_period, source):
        logging.info("Evaluating strategy...")

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

        return evaluation, baseline_evaluation

    @property
    def report(self):
        return self._report


class ComparativeReportBuilder:
    """
    Based on a set of backtests and corresponding baseline backtests, builds a comparative report of their performance.
    """

    def __init__(self, backtests, baselines):
        """
        Initializes and builds the report dataframe.
        :param backtests: A collection of backtest objects.
        :param baselines: A collection of baselines corresponding to backtests.
        """
        self.backtests = backtests
        self.baselines = baselines
        self._build_dataframe()

    def _build_dataframe(self):
        evaluation_dicts = [self._create_row_dict(backtest, baseline) for backtest, baseline in zip(self.backtests, self.baselines)]
        output = pd.DataFrame(evaluation_dicts)
        if len(output) == 0:
            logging.warning("No strategies evaluated.")
            return
        # sort results by profit
        # output = output.sort_values(by=['profit_percent'], ascending=False)

        # drop index
        output.reset_index(inplace=True, drop=True)

        # reformat sources to human readable format
        output.source.replace([0, 1, 2], ['Poloniex', 'Bittrex', 'Binance'], inplace=True)

        # save full results
        self.results_df = output


    def _create_row_dict(self, evaluation, baseline):
        evaluation_dict = evaluation.to_dictionary()
        baseline_dict = baseline.to_dictionary()
        evaluation_dict["evaluation_object"] = evaluation
        evaluation_dict["buy_and_hold_profit"] = baseline_dict["profit"]
        evaluation_dict["buy_and_hold_profit_percent"] = baseline_dict["profit_percent"]
        evaluation_dict["buy_and_hold_profit_USDT"] = baseline_dict["profit_USDT"]
        evaluation_dict["buy_and_hold_profit_percent_USDT"] = baseline_dict["profit_percent_USDT"]
        return evaluation_dict

    def get_best_performing_backtest(self):
        return self.results_df[self.results_df.num_trades > 0].iloc[0]["evaluation_object"]

    def write_summary(self, output_file):
        writer = pd.ExcelWriter(output_file)
        # filter so that only report columns remain
        self.results_df[backtesting_report_columns]. \
            sort_values(by=['profit_percent'], ascending=False).to_excel(writer, 'Results')
        writer.save()

    def write_comparative_summary(self, summary_path):
        writer = pd.ExcelWriter(summary_path)
        summary = self.results_df.copy()
        # remove empty trades
        summary = summary[summary.num_trades != 0]
        summary.groupby(["strategy", "resample_period"]).describe().to_excel(writer, 'Results')
        writer.save()

    def _filter_df_based_on_currency_pairs(self, df, currency_pairs_to_keep):
        return df[df[["transaction_currency", "counter_currency"]].apply(tuple, 1).isin(currency_pairs_to_keep)]

    def _get_description_stat_values(self, desc_df, field_name, row_name):
        mean = desc_df[[(field_name, 'mean')]].loc[row_name][0]
        std = desc_df[[(field_name, 'std')]].loc[row_name][0]
        min = desc_df[[(field_name, 'min')]].loc[row_name][0]
        max = desc_df[[(field_name, 'max')]].loc[row_name][0]
        return mean, std, min, max

    def _describe_and_write(self, filtered_df, writer, sheet_prefix):

        if filtered_df.empty:
            return

        # group by strategy, evaluate
        by_strategy_df = filtered_df[["source", "strategy", "resample_period", "profit_percent",
                                      "buy_and_hold_profit_percent", "num_trades"]]\
            .groupby(["source", "strategy", "resample_period"], sort=False).describe(percentiles=[])

        # add outperformance stat
        outperforms = pd.Series(by_strategy_df[('profit_percent', 'mean')]) > \
                      pd.Series(by_strategy_df[('buy_and_hold_profit_percent', 'mean')])

        by_strategy_df["outperforms"] = outperforms

        # reorder columns and write
        by_strategy_df[["profit_percent", "buy_and_hold_profit_percent", "num_trades", "outperforms"]]\
            .to_excel(writer, f'{sheet_prefix} by strategy', header=False, startrow=4)


        # format
        workbook = writer.book
        percent_format = workbook.add_format({'num_format': '0.00\%'})
        gray_percent_format = workbook.add_format({'num_format': '0.00\%', 'pattern': 1, 'bg_color' : '#D9D9D9'})

        number_format = workbook.add_format({'num_format': '0.00'})
        bold_red_percent = workbook.add_format({'bold': True, 'font_color':'red', 'num_format': '0.00\%'})
        bold_red_number = workbook.add_format({'bold': True, 'font_color': 'red', 'num_format': '0.00'})
        large_bold_red = workbook.add_format({'bold': True, 'font_color': 'red', 'font_size': 16, 'num_format': '0.00%',
                                              'border': 1, 'pattern': 1, 'bg_color': '#BFBFBF', 'align': 'center',
                                              'valign': 'vcenter'})

        gray_bold_red_percent = workbook.add_format({'bold': True, 'font_color': 'red', 'num_format': '0.00\%',
                                                     'pattern': 1, 'bg_color' : '#D9D9D9'})



        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'align': 'center',
            'valign': 'vcenter',
            'bg_color': '#808080',
            'font_color': '#FFFFFF',
            'border': 1,
            'pattern': 1
        })

        aux_header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'align': 'center',
            'valign': 'vcenter',
            'bg_color': '#BFBFBF',
            'font_color': '#FFFFFF',
            'border': 1,
            'pattern': 1
        })

        mid_bold = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'align': 'center',
            'valign': 'vcenter',
        })

        top_bold = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'align': 'center',
            'valign': 'top'
        })

        # add outperformance percent at the top
        sheet = writer.sheets[f'{sheet_prefix} by strategy']
        sheet.write('V4', np.mean(outperforms), large_bold_red)

        # apply this to all header cells and reinit header
        sheet.merge_range('D3:I3', 'Profit percent', header_format)
        sheet.merge_range('J3:O3', 'Buy and hold profit percent', header_format)
        sheet.merge_range('P3:U3', 'Number of trades', header_format)
        sheet.write('V3', 'Outperforms', header_format)

        sheet.write_row('D4', ['number of tests (coin, strategy)',
                               'mean', 'std', 'min', 'median', 'max'] * 3, aux_header_format)
        sheet.write_row('A5', ['exchange',
                               'strategy', 'resample period (minutes)'], header_format)

        sheet.set_column('E:I', None, percent_format)
        sheet.set_column('L:O', None, gray_percent_format)
        sheet.set_column('Q:U', None, number_format)
        sheet.set_column('E:E', None, bold_red_percent)
        sheet.set_column('K:K', None, gray_bold_red_percent)
        sheet.set_column('V:V', 15, gray_percent_format)
        sheet.set_column('B:C', None, mid_bold)
        sheet.set_column('A:A', None, top_bold)

        #sheet.set_column('')
        sheet.set_column('Q:Q', None, bold_red_number)


        #sheet.set_row(0, None, header_format)

        # Write the column headers with the defined format.
        # for col_num, value in enumerate(by_strategy_df.columns.values):
        #    sheet.write(0, col_num + 1, value[0], header_format)
        #    sheet.write(1, col_num + 1, value[1], header_format)

        #writer.cur_sheet.write()


        # group by coin, evaluate
        by_coin_df = filtered_df[["source", "transaction_currency", "profit_percent",
                                  "buy_and_hold_profit_percent", "num_trades"]]\
            .groupby(["source", "transaction_currency"]).describe(percentiles=[])
        # reorder columns and write
        # by_coin_df[["profit_percent", "buy_and_hold_profit_percent", "num_trades"]].to_excel(writer, f'{sheet_prefix} by coin')

        g = by_coin_df.groupby(level=0, group_keys=False)
        by_coin_sorted_df = g.apply(lambda x: x.sort_values([('profit_percent', 'mean')], ascending=False))
        by_coin_sorted_df[["profit_percent", "buy_and_hold_profit_percent", "num_trades"]].to_excel(writer, f'{sheet_prefix} sorted by coin')

        sheet = writer.sheets[f'{sheet_prefix} sorted by coin']
        sheet.set_column('D:H', None, percent_format)
        sheet.set_column('J:N', None, percent_format)
        sheet.set_column('P:T', None, number_format)

    def all_coins_report(self, report_path, currency_pairs_to_keep=None, group_strategy_variants=True):
        df = self.results_df.copy(deep=True)

        # remove strategies without any trades - very important for averaging!
        df = df[df.num_trades > 0]

        if group_strategy_variants:
            # clean up and structure strategy names
            df.loc[df['strategy'].str.contains('rsi_buy'), 'strategy'] = 'RSI variant'
            df.loc[df['strategy'].str.contains('rsi_cumulat'), 'strategy'] = 'RSI cumulative variant'
            df.loc[df['strategy'].str.contains('ann_'), 'strategy'] = 'ANN'
            df.loc[df['strategy'].str.contains('ichi_'), 'strategy'] = 'Ichimoku'

        writer = pd.ExcelWriter(report_path)

        if currency_pairs_to_keep is not None:
            df = self._filter_df_based_on_currency_pairs(currency_pairs_to_keep)

        self._describe_and_write(df, writer, "All coins")
        top_20_alts = zip(COINMARKETCAP_TOP_20_ALTS, ["BTC"] * len(COINMARKETCAP_TOP_20_ALTS))
        top20 = self._filter_df_based_on_currency_pairs(df, top_20_alts)
        self._describe_and_write(top20, writer, "Top 20 alts")

        # top 10 best performing coins
        # top 10 worst performing coins
        df[backtesting_report_columns].to_excel(writer, "Original data")

        writer.save()
        writer.close()






