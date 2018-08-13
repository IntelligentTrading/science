from comparative_evaluation import *
from strategies import RandomTradingStrategy
import numpy as np

def best_performing_signals_of_the_week():
    # for John - best performing signals of the week
    counter_currency = "BTC"
    transaction_currencies = get_currencies_for_signal(counter_currency, "RSI_Cumulative")
    currency_pairs = []
    for transaction_currency in transaction_currencies:
        currency_pairs.append((transaction_currency, counter_currency))

    # for debugging
    currency_pairs = [("BTC","USDT"),]

    end_time = 1531699200
    start_time = end_time - 60*60*24*7

    strategies = StrategyEvaluationSetBuilder.build_from_signal_set(
        buy_signals=['rsi_buy_3', 'rsi_buy_2', 'rsi_cumulat_buy_2', 'rsi_cumulat_buy_3', 'ichi_kumo_up'], #, 'ann_simple_bull'],
        sell_signals=['rsi_sell_3', 'rsi_sell_2', 'rsi_cumulat_sell_2', 'rsi_cumulat_sell_3', 'ichi_kumo_down'], #, 'ann_simple_bear'],
        num_buy=2,
        num_sell=2,
        signal_combination_mode=SignalCombinationMode.SAME_TYPE)

    comparison = ComparativeEvaluation(strategy_set=strategies,
                                       currency_pairs=currency_pairs,
                                       resample_periods=[60,240,1440],
                                       source=0,
                          start_cash=1, start_crypto=0,
                          start_time=start_time, end_time=end_time,
                          output_file="best_performing_2018_07_16_refactor2.xlsx"
                          )

    #comparison.write_comparative_summary("description.xlsx")


def rsi_vs_rsi_cumulative(start_time, end_time, time_delay=0):
    counter_currency = "BTC"
    transaction_currencies = get_currencies_for_signal(counter_currency, "RSI_Cumulative")
    currency_pairs = []
    for transaction_currency in transaction_currencies:
        currency_pairs.append((transaction_currency, counter_currency))


    strategies_rsi = StrategyEvaluationSetBuilder.build_from_rsi_thresholds("RSI",
                                                                        [75], [25],
                                                                        [Horizon.short, Horizon.medium, Horizon.long],
                                                                        start_time,
                                                                        end_time,
                                                                        currency_pairs)
    strategies_rsi_cumulative = StrategyEvaluationSetBuilder.build_from_rsi_thresholds("RSI_Cumulative",
                                                                        [75], [25],
                                                                        [Horizon.short, Horizon.medium, Horizon.long],
                                                                        start_time,
                                                                        end_time,
                                                                        currency_pairs)
    strategies_rsi.extend(strategies_rsi_cumulative)

    ComparativeEvaluation(strategy_set=strategies_rsi,
                          start_cash=1, start_crypto=0,
                          start_time=start_time, end_time=end_time,
                          output_file="RSI_cumulative_delayed.xlsx",
                          time_delay=time_delay
                          )
    find_num_cumulative_outperforms(start_time, end_time, currency_pairs)


def delayed_trading(time_delay = 60*1):
    end = 1526637600
    start = end - 60 * 60 * 24 * 30

    transaction_currency = "ETH"
    counter_currency = "BTC"
    horizon = Horizon.short
    overbought_threshold = 75
    oversold_threshold = 25
    start_cash = 1
    start_crypto = 0

    rsi = SimpleRSIStrategy(start, end, horizon, counter_currency, overbought_threshold,
                            oversold_threshold, transaction_currency, "RSI")
    evaluation_rsi = rsi.evaluate(start_cash, start_crypto, start, end)
    evaluation_rsi_delayed = rsi.evaluate(start_cash, start_crypto, start, end, time_delay=time_delay)


def delayed_trading_stats():
    end = 1526637600
    start = end - 60 * 60 * 24 * 30

    counter_currency = "BTC"
    transaction_currencies = get_currencies_for_signal(counter_currency, "RSI")
    currency_pairs = []
    for transaction_currency in transaction_currencies:
        currency_pairs.append((transaction_currency, counter_currency))

    horizons = [Horizon.short, Horizon.medium, Horizon.long]
    overbought_threshold = 75
    oversold_threshold = 25
    start_cash = 1
    start_crypto = 0

    output = []
    deltas = []

    profits = []
    profits_delayed = []
    per_trade_deltas = []

    for time_delay in [60*5, 60*3, 60*5, 60*60, 60*60*8, 60*60*24]:
        for transaction_currency, counter_currency in currency_pairs:
            for horizon in horizons:
                try:
                    rsi = SimpleRSIStrategy(start, end, horizon, counter_currency, overbought_threshold,
                                    oversold_threshold, transaction_currency, "RSI")
                    evaluation_rsi = rsi.evaluate(start_cash, start_crypto, start, end)
                    evaluation_rsi_delayed = rsi.evaluate(start_cash, start_crypto, start, end, time_delay=time_delay)
                    if evaluation_rsi.get_num_trades() == 0 or evaluation_rsi_delayed.get_num_trades() == 0:
                        continue
                    deltas.append(evaluation_rsi_delayed.profit_percent() - evaluation_rsi.profit_percent())
                    per_trade_deltas.append(evaluation_rsi_delayed.avg_profit_per_trade_pair - evaluation_rsi.avg_profit_per_trade_pair)
                    profits.append(evaluation_rsi.profit_percent())
                    profits_delayed.append(evaluation_rsi_delayed.profit_percent())
                except NoPriceDataException:
                    print("Price data not found!")
                    continue
        output.append("Time delay: {} delta_mean: {} delta_std: {} num_measurements: {} per_trade_delta_mean: {} per_trade_delta_std {}"
                      .format(time_delay, np.mean(deltas), np.std(deltas), len(deltas), np.mean(per_trade_deltas), np.std(per_trade_deltas)))
        deltas = []
    output.append("Mean profit: {}, std: {}".format(np.mean(profits), np.std(profits)))
    print(profits)
    print(len(profits))
    print("\n".join(output))


def random_strategy_backtesting(out_path="random_strat_backtesting.txt"):
    out = open(out_path, "w")
    end= 1526637600
    start = end - 60 * 60 * 24 * 30
    num_tests = 10

    transaction_currency = "ETH"
    counter_currency = "BTC"

    results_random = []
    for max_num_signals in [5, 10, 20, 30, 50, 70, 100, 200]:
        num_evaluations = 0
        profit_percent = 0
        profits = []
        for i in range(0, num_tests):
            strategy = RandomTradingStrategy(max_num_signals, start, end, Horizon.short, counter_currency, transaction_currency)
            evaluation = strategy.evaluate(1, 0, start, end)

            if evaluation.num_trades == 0:
                continue
            num_evaluations += 1
            profit_percent += evaluation.profit_percent()
            profits.append(evaluation.profit_percent())

        results_random.append({"max_num_signals": max_num_signals, "avg_profit_percent" : np.mean(profits), "std_profit_percent": np.std(profits)})
        print("Average profit percent: {0:0.2f}%".format(profit_percent / num_evaluations))
    df = pd.DataFrame(results_random)
    print(df)
    out.write(str(df))
    out.write("\n--\n")
    out.write(str(df.describe()))
    out.write("\n\n")
    bah = BuyAndHoldTimebasedStrategy(start, end, transaction_currency, counter_currency, 0, Horizon.any)
    bah_eval = bah.evaluate(1, 0, start, end, False, True)
    print("Buy and hold performance: {0:0.2f}%".format(bah_eval.profit_percent()))
    out.write("Buy and hold performance: {0:0.2f}%\n".format(bah_eval.profit_percent()))
    out.write("\n")

    results_rsi = []
    for overbought_threshold in [70, 75, 80]:
        for oversold_threshold in [20, 25, 30]:
            rsi = SimpleRSIStrategy(start, end, Horizon.short, counter_currency, overbought_threshold, oversold_threshold, transaction_currency)
            rsi_eval = rsi.evaluate(1, 0, start, end, False, True)
            print("RSI (overbought = {}, oversold = {}) performance: {:0.2f}".format(overbought_threshold,
                                                                                     oversold_threshold,
                                                                                     rsi_eval.profit_percent()))
            results_rsi.append({"overbought_threshold": overbought_threshold,
                                "oversold_threshold" : oversold_threshold,
                                "profit_percent": rsi_eval.profit_percent()})

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
    best_performing_signals_of_the_week()

    # RSI vs RSI cumulative
    # start_time = 1518523200  # first instance of RSI_Cumulative signal
    # end_time = 1526637600
    # rsi_vs_rsi_cumulative(start_time, end_time, 60*5)
    # exit(0)


    # Other runs
    # evaluate_rsi_any_currency("BTC", start, end, 1000, 0, 70, 30)
    # evaluate_rsi_comparatively("BTC", "USDT", start, end, 1000, 0)
    # evaluate_rsi("ETH", "USDT", start, end, 1000, 0, Horizon.short)
    # evaluate_multi("ETH", "USDT", start, end, 1000, 0, 70, 30)

