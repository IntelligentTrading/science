from comparative_evaluation import *
from strategies import RandomTradingStrategy, ANNAnomalyStrategy
import numpy as np
import datetime
from backtesting_helpers import find_num_cumulative_outperforms
from data_sources import get_currencies_trading_against_counter
from utils import time_performance

@time_performance
def best_performing_signals_of_the_period(start_time=None, end_time=None, additional_strategies=[],
                                          best_performing_filename=None, full_report_filename=None,
                                          group_strategy_variants=False):
    if start_time is None or end_time is None:
        start_time = datetime.datetime(2018, 10, 1, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        end_time = datetime.datetime(2018, 11, 1, 23, 59, tzinfo=datetime.timezone.utc).timestamp()

    if best_performing_filename is None:
        best_performing_filename = f"best_performing_{datetime.datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d')}.xlsx"

    if full_report_filename is None:
        full_report_filename = f"full_report_{datetime.datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d')}.xlsx"

    ann_rsi_strategies, basic_strategies, vbi_strategies, ann_anomaly_strategies = build_itf_baseline_strategies()
    strategies = basic_strategies + vbi_strategies + ann_rsi_strategies + ann_anomaly_strategies + additional_strategies

    comparison = ComparativeEvaluation(strategy_set=ann_anomaly_strategies, start_cash=1, start_crypto=0, start_time=start_time,
                                       end_time=end_time, resample_periods=[60, 240, 1440], counter_currencies=["BTC"],
                                       sources=[0, 1, 2], output_file=best_performing_filename, debug=False, parallelize=False)

    comparison.report.all_coins_report(full_report_filename, group_strategy_variants=group_strategy_variants)


def build_itf_baseline_strategies():
    basic_strategies = StrategyEvaluationSetBuilder.build_from_signal_set(
        buy_signals=['rsi_buy_3', 'rsi_buy_2', 'rsi_cumulat_buy_2', 'rsi_cumulat_buy_3', 'ichi_kumo_up',
                     'ann_simple_bull'],
        sell_signals=['rsi_sell_3', 'rsi_sell_2', 'rsi_cumulat_sell_2', 'rsi_cumulat_sell_3', 'ichi_kumo_down',
                      'ann_simple_bear'],
        num_buy=2,
        num_sell=2,
        signal_combination_mode=SignalCombinationMode.SAME_TYPE)
    vbi_strategies = StrategyEvaluationSetBuilder.build_from_signal_set(
        buy_signals=['vbi_buy'],
        sell_signals=['rsi_sell_1', 'rsi_sell_2', 'rsi_sell_3'],
        num_buy=1,
        num_sell=1,
        signal_combination_mode=SignalCombinationMode.ANY
    )
    ann_rsi_strategies = StrategyEvaluationSetBuilder.build_from_signal_set(
        buy_signals=['rsi_buy_1', 'rsi_buy_2', 'rsi_buy_3'],
        sell_signals=["ann_simple_bear"],
        num_buy=1,
        num_sell=1,
        signal_combination_mode=SignalCombinationMode.ANY
    )

    # ANN anomaly strategies
    comparative_signals = ['RSI', 'RSI_Cumulative', 'ANN_Simple', 'VBI', 'kumo_breakout']
    candle_periods = [0, 1, 3, 5]
    ann_anomaly_strategies = [ANNAnomalyStrategy(confirmation_signal, max_delta_period)
                              for confirmation_signal, max_delta_period in itertools.product(comparative_signals, candle_periods)]

    return ann_rsi_strategies, basic_strategies, vbi_strategies, ann_anomaly_strategies


def in_depth_signal_comparison(out_path, additional_strategies = []):
    ann_rsi_strategies, basic_strategies, vbi_strategies, ann_anomaly_strategies = build_itf_baseline_strategies()
    strategies = basic_strategies + vbi_strategies + ann_rsi_strategies + ann_anomaly_strategies + additional_strategies

    periods = {
        'Mar 2018': ('2018/03/01 00:00:00 UTC', '2018/03/31 23:59:59 UTC'),
        'Apr 2018': ('2018/04/01 00:00:00 UTC', '2018/04/30 23:59:59 UTC'),
        'May 2018': ('2018/05/01 00:00:00 UTC', '2018/05/31 23:59:59 UTC'),
        'Jun 2018': ('2018/06/01 00:00:00 UTC', '2018/06/30 23:59:59 UTC'),
        'Jul 2018': ('2018/07/01 00:00:00 UTC', '2018/07/31 23:59:59 UTC'),
        'Aug 2018': ('2018/08/01 00:00:00 UTC', '2018/08/31 23:59:59 UTC'),
        'Sep 2018': ('2018/09/01 00:00:00 UTC', '2018/09/30 23:59:59 UTC'),
        'Oct 2018': ('2018/10/01 00:00:00 UTC', '2018/10/31 23:59:59 UTC'),
        'Q1 2018': ('2018/01/01 00:00:00 UTC', '2018/03/31 23:59:59 UTC'),
        'Q2 2018': ('2018/04/01 00:00:00 UTC', '2018/06/30 23:59:59 UTC'),
        '678 2018': ('2018/06/01 00:00:00 UTC', '2018/08/31 23:59:59 UTC'),
    }

    writer = pd.ExcelWriter(out_path)

    from dateutil import parser
    for period in periods:
        start_time = parser.parse(periods[period][0]).timestamp()
        end_time = parser.parse(periods[period][1]).timestamp()

        comparison = ComparativeEvaluation(strategy_set=strategies, start_cash=1, start_crypto=0, start_time=start_time,
                                           end_time=end_time, resample_periods=[60, 240, 1440],
                                           counter_currencies=["BTC"], sources=[0, 1, 2],
                                           output_file=f"best_performing_{datetime.datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d')}.xlsx",
                                           debug=False)

        comparison.report.all_coins_report(writer=writer, sheet_prefix=f'({period}) ', group_strategy_variants=False)

    writer.save()
    writer.close()


def rsi_vs_rsi_cumulative(start_time, end_time, time_delay=0):
    source = 0
    counter_currency = "BTC"
    transaction_currencies = get_currencies_for_signal(counter_currency, "RSI_Cumulative", source=0)
    resample_periods = [60, 240, 1440]
    tickers = [Ticker(source, transaction_currency, counter_currency) for transaction_currency in transaction_currencies]

    strategies_rsi = StrategyEvaluationSetBuilder.build_from_signal_set(
        ['rsi_buy_2'], ['rsi_sell_2'], 1, 1, SignalCombinationMode.SAME_TYPE)
    strategies_rsi_cumulative = StrategyEvaluationSetBuilder.build_from_signal_set(
        ['rsi_cumulat_buy_2'], ['rsi_cumulat_sell_2'], 1, 1, SignalCombinationMode.SAME_TYPE)
    strategies_rsi.extend(strategies_rsi_cumulative)

    ComparativeEvaluation(strategy_set=strategies_rsi, start_cash=1, start_crypto=0, start_time=start_time,
                          end_time=end_time, resample_periods=resample_periods, tickers=tickers,
                          output_file="RSI_cumulative_delayed.xlsx", time_delay=time_delay, parallelize=False)

    find_num_cumulative_outperforms(
        tickers=tickers,
        resample_periods=resample_periods,
        start_cash=1,
        start_crypto=0,
        start_time=start_time,
        end_time=end_time
    )


def delayed_trading_stats():
    end_time = 1526637600
    start_time = end_time - 60 * 60 * 24 * 30

    counter_currency = "BTC"
    transaction_currencies = get_currencies_for_signal(counter_currency, "RSI")
    currency_pairs = []
    for transaction_currency in transaction_currencies:
        currency_pairs.append((transaction_currency, counter_currency))

    resample_periods = [60, 240, 1440]
    overbought_threshold = 75
    oversold_threshold = 25
    start_cash = 1
    start_crypto = 0
    source = 0

    output = []
    deltas = []

    profits = []
    profits_delayed = []
    per_trade_deltas = []

    for time_delay in [60*1, 60*3, 60*5, 60*60, 60*60*8, 60*60*24]:
        for transaction_currency, counter_currency in currency_pairs:
            for resample_period in resample_periods:
                try:
                    rsi = SimpleRSIStrategy(overbought_threshold, oversold_threshold, "RSI")
                    evaluation_rsi = rsi.evaluate(
                        transaction_currency=transaction_currency,
                        counter_currency=counter_currency,
                        start_cash=start_cash,
                        start_crypto=start_crypto,
                        start_time=start_time,
                        end_time=end_time,
                        source=source,
                        resample_period=resample_period
                    )


                    evaluation_rsi_delayed = rsi.evaluate(
                        transaction_currency=transaction_currency,
                        counter_currency=counter_currency,
                        start_cash=start_cash,
                        start_crypto=start_crypto,
                        start_time=start_time,
                        end_time=end_time,
                        source=source,
                        resample_period=resample_period,
                        time_delay=time_delay
                    )

                    if evaluation_rsi.num_trades == 0 or evaluation_rsi_delayed.num_trades == 0\
                            or evaluation_rsi.profit_percent is None or evaluation_rsi_delayed.profit_percent is None:
                        continue
                    deltas.append(evaluation_rsi_delayed.profit_percent - evaluation_rsi.profit_percent)
                    per_trade_deltas.append(evaluation_rsi_delayed.mean_buy_sell_pair_return - evaluation_rsi.mean_buy_sell_pair_return)
                    profits.append(evaluation_rsi.profit_percent)
                    profits_delayed.append(evaluation_rsi_delayed.profit_percent)
                except NoPriceDataException:
                    logging.error("Price data not found!")
                    continue
        output.append("Time delay: {} delta_mean: {} delta_std: {} num_measurements: {} per_trade_delta_mean: {} per_trade_delta_std {}"
                      .format(time_delay, np.mean(deltas), np.std(deltas), len(deltas), np.mean(per_trade_deltas), np.std(per_trade_deltas)))
        deltas = []
    output.append("Mean profit: {}, std: {}".format(np.mean(profits), np.std(profits)))
    logging.debug(profits)
    logging.debug(len(profits))
    logging.info("\n".join(output))


def random_strategy_backtesting(out_path="random_strat_backtesting.txt"):
    out = open(out_path, "w")
    end = 1526637600
    start = end - 60 * 60 * 24 * 30
    source = 0
    num_tests = 10
    start_cash = 1000
    start_crypto = 0

    transaction_currency = "ETH"
    counter_currency = "BTC"

    results_random = []


    for max_num_signals in [5, 10, 20, 30, 50, 70, 100, 200]:
        num_evaluations = 0
        profit_percent = 0
        profits = []
        for i in range(0, num_tests):
            strategy = RandomTradingStrategy(max_num_signals, start, end, transaction_currency, counter_currency, source)
            evaluation = SignalDrivenBacktester(
                strategy=strategy,
                transaction_currency=transaction_currency,
                counter_currency=counter_currency,
                start_cash=start_cash,
                start_crypto=start_crypto,
                start_time=start,
                end_time=end,
                source=source,
                resample_period=None,
                evaluate_profit_on_last_order=False,
                verbose=False,
                time_delay=0,
                slippage=0,
                signals=strategy.signals,
            )

            if evaluation.num_trades == 0:
                continue
            num_evaluations += 1
            profit_percent += evaluation.profit_percent
            profits.append(evaluation.profit_percent)

        results_random.append({"max_num_signals": max_num_signals,
                               "avg_profit_percent" : np.mean(profits),
                               "std_profit_percent": np.std(profits)})
        logging.info("Average profit percent: {0:0.2f}%".format(profit_percent / num_evaluations))

    df = pd.DataFrame(results_random)
    logging.info(df)
    out.write(str(df))
    out.write("\n--\n")
    out.write(str(df.describe()))
    out.write("\n\n")
    bah = BuyAndHoldTimebasedStrategy(start, end, transaction_currency, counter_currency)
    bah_eval = bah.evaluate(transaction_currency, counter_currency, start_cash, start_crypto, start, end, source, None)
    logging.info("Buy and hold performance: {0:0.2f}%".format(bah_eval.profit_percent))
    out.write("Buy and hold performance: {0:0.2f}%\n".format(bah_eval.profit_percent))
    out.write("\n")

    results_rsi = []
    for overbought_threshold in [70, 75, 80]:
        for oversold_threshold in [20, 25, 30]:
            rsi = SimpleRSIStrategy(overbought_threshold, oversold_threshold)
            rsi_eval = rsi.evaluate(transaction_currency, counter_currency, start_cash, start_crypto, start, end, source, 60)
            logging.info("RSI (overbought = {}, oversold = {}) performance: {:0.2f}".format(overbought_threshold,
                                                                                     oversold_threshold,
                                                                                     rsi_eval.profit_percent))
            results_rsi.append({"overbought_threshold": overbought_threshold,
                                "oversold_threshold" : oversold_threshold,
                                "profit_percent": rsi_eval.profit_percent})

    df = pd.DataFrame(results_rsi)
    out.write(str(df))
    out.write("\n--\n\n")
    out.write(str(df.describe()))
    out.close()




if __name__ == "__main__":
    # Random strategy backtesting
    # random_strategy_backtesting()

    # Delayed trading
    # delayed_trading_stats()

    # Best performing signals
    #best_performing_signals_of_the_period()

    in_depth_signal_comparison('comp_no_ann_Nov.xlsx')

    # RSI vs RSI cumulative
    # start_time = 1518523200  # first instance of RSI_Cumulative signal
    # end_time = 1526637600
    # start_time =  datetime.datetime(2018, 5, 1, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    # end_time = datetime.datetime(2018, 7, 31, 23, 59, tzinfo=datetime.timezone.utc).timestamp()
    # rsi_vs_rsi_cumulative(start_time, end_time, 0)

    # Other runs
    # evaluate_rsi_any_currency("BTC", start, end, 1000, 0, 70, 30)
    # evaluate_rsi_comparatively("BTC", "USDT", start, end, 1000, 0)
    # evaluate_rsi("ETH", "USDT", start, end, 1000, 0, Horizon.short)
    # evaluate_multi("ETH", "USDT", start, end, 1000, 0, 70, 30)

