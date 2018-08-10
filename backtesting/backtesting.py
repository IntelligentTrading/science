from backtester_signals import SignalDrivenBacktester
from strategies import *

### Various sample backtesting runs

def evaluate_rsi_signature(transaction_currency, counter_currency, start_time, end_time,
                 start_cash, start_crypto, source=0, resample_period=60, horizon=Horizon.any, time_delay=0):
    rsi_strategy = SignalSignatureStrategy(['rsi_buy_2', 'rsi_sell_2','rsi_buy_1', 'rsi_sell_1','rsi_buy_3', 'rsi_sell_3'], start_time, end_time, horizon, counter_currency, transaction_currency)
    evaluation = SignalDrivenBacktester(rsi_strategy, transaction_currency, counter_currency,
                            start_cash, start_crypto, start_time, end_time, source, resample_period, False, False, time_delay)
    print(evaluation.get_report())
    return evaluation

def evaluate_rsi(transaction_currency, counter_currency, start_time, end_time,
                 start_cash, start_crypto, overbought_threshold, oversold_threshold, horizon=Horizon.any, signal_type="RSI",
                 time_delay=0):
    rsi_strategy = SimpleRSIStrategy(start_time, end_time, horizon, counter_currency, overbought_threshold,
                                     oversold_threshold, transaction_currency, signal_type)
    evaluation = rsi_strategy.evaluate(start_cash, start_crypto, start_time, end_time)
    print(evaluation.get_report())
    return evaluation

def evaluate_rsi_any_currency(counter_currency, start_time, end_time,
                 start_cash, start_crypto, overbought_threshold, oversold_threshold):
    rsi_strategy = SimpleRSIStrategy(start_time, end_time, Horizon.short, counter_currency, overbought_threshold, oversold_threshold)
    evaluation = SignalDrivenBacktester(rsi_strategy, "", counter_currency, start_cash, start_crypto, start_time, end_time, False)
    print(evaluation.get_report())
    return evaluation


def evaluate_trend_based(signal_type, transaction_currency, counter_currency, start_time, end_time,
                 start_cash, start_crypto, horizon=Horizon.any, strength=Strength.short):
    strategy = SimpleTrendBasedStrategy(signal_type, start_time, end_time, horizon, counter_currency, transaction_currency, strength)
    evaluation = SignalDrivenBacktester(strategy, transaction_currency, counter_currency, start_cash, start_crypto, start_time, end_time, False, False)
    return evaluation


def evaluate_rsi_comparatively(transaction_currency, counter_currency, start_time, end_time,
                 start_cash, start_crypto):
    oversold = [20, 25, 30]
    overbought = [70, 75, 80]
    print("Start time: {}".format(datetime_from_timestamp(start_time)))
    print("End time: {}".format(datetime_from_timestamp(end_time)))
    print("Start price: {}".format(get_price(transaction_currency, start_time, counter_currency)))
    print("End price: {}".format(get_price(transaction_currency, end_time, counter_currency)))

    for overbought_threshold in overbought:
        for oversold_threshold in oversold:
            rsi_strategy = SimpleRSIStrategy(start_time, end_time, Horizon.short, counter_currency,
                                             overbought_threshold, oversold_threshold, transaction_currency)
            baseline = BuyOnFirstSignalAndHoldStrategy(rsi_strategy)
            baseline_evaluation = SignalDrivenBacktester(baseline, transaction_currency, counter_currency, start_cash,
                                             start_crypto, start_time, end_time, False)
            rsi_evaluation = SignalDrivenBacktester(rsi_strategy, transaction_currency, counter_currency, start_cash,
                                        start_crypto, start_time, end_time, False)
            print("RSI overbought = {}, oversold = {}".format(overbought_threshold, oversold_threshold))
            print("  Profit - RSI buy and hold: {0:.2f}%".format(baseline_evaluation.profit_percent()))
            print("  Profit - RSI trading: {0:.2f}% ({1} trades)\n".format(rsi_evaluation.profit_percent(), rsi_evaluation.num_trades))


def evaluate_multi(transaction_currency, counter_currency, start_time, end_time,
                 start_cash, start_crypto, overbought_threshold, oversold_threshold):
    horizon = Horizon.any
    strength = Strength.any
    rsi_strategy = SimpleRSIStrategy(start_time, end_time, horizon, counter_currency, overbought_threshold, oversold_threshold)
    sma_strategy = SimpleTrendBasedStrategy("SMA", start_time, end_time, horizon, counter_currency,
                                        transaction_currency, strength)
    kumo_strategy = SimpleTrendBasedStrategy("kumo_breakout", start_time, end_time, horizon, counter_currency,
                                            transaction_currency, strength)
    rsi_c_strategy = SimpleTrendBasedStrategy("RSI_Cumulative", start_time, end_time, horizon, counter_currency,
                                            transaction_currency, strength)

    buy = (rsi_c_strategy, rsi_strategy )
    sell = (sma_strategy, kumo_strategy)

    multi_strat = MultiSignalStrategy(buy, sell, horizon)
    buy_and_hold = BuyOnFirstSignalAndHoldStrategy(multi_strat)
    evaluation = SignalDrivenBacktester(multi_strat, transaction_currency, counter_currency, start_cash, start_crypto,
                            start_time,
                            end_time, False, True)
    baseline = SignalDrivenBacktester(buy_and_hold, transaction_currency, counter_currency, start_cash, start_crypto,
                          start_time,
                          end_time, False, True)

    print(evaluation.get_report())
    print(baseline.get_report())


def evaluate_rsi_cumulative_compare(start_time, end_time, transaction_currency, counter_currency, horizon=Horizon.any):
    start_cash = 1
    start_crypto = 0
    overbought_threshold = 75
    oversold_threshold = 25

    rsi = SimpleRSIStrategy(start_time, end_time, horizon, counter_currency, overbought_threshold,
                                     oversold_threshold, transaction_currency, "RSI")
    evaluation_rsi = rsi.evaluate(start_cash, start_crypto, start_time, end_time)

    rsi_cumulative = SimpleRSIStrategy(start_time, end_time, horizon, counter_currency, overbought_threshold,
                                     oversold_threshold, transaction_currency, "RSI_Cumulative")
    evaluation_rsi_cumulative = rsi_cumulative.evaluate(start_cash, start_crypto, start_time, end_time)

    f = open("rsi.txt", "w")
    f.write(evaluation_rsi.get_report(True))
    f.close()

    f = open("rsi_cumulative.txt", "w")
    f.write(evaluation_rsi_cumulative.get_report(True))
    f.close()

    return evaluation_rsi_cumulative, evaluation_rsi


def find_num_cumulative_outperforms(start_time, end_time, currency_pairs):
    horizons = (Horizon.short, Horizon.medium, Horizon.long)
    total_rsi_cumulative_better = 0
    total_rsi_cumulative_eq = 0
    total_evaluated_pairs = 0
    sprofit = 0
    rsi_profitable = 0
    rsi_cumulative_profitable = 0
    total_rsi = 0
    total_cumulative = 0
    for transaction_currency, counter_currency in currency_pairs:
        for horizon in horizons:
            try:
                evaluation_cumulative, evaluation_rsi = evaluate_rsi_cumulative_compare(start_time, end_time,
                                                                  transaction_currency,
                                                                  counter_currency,
                                                                  horizon)

                profit_rsi_cumulative = evaluation_cumulative.profit_percent()
                profit_rsi = evaluation_rsi.profit_percent()

                if evaluation_cumulative.get_num_trades() == 0 and evaluation_rsi.get_num_trades() == 0:
                    continue
                if profit_rsi_cumulative is None or profit_rsi is None:
                    continue
                if profit_rsi_cumulative > profit_rsi:
                    total_rsi_cumulative_better += 1
                elif profit_rsi_cumulative == profit_rsi:
                    total_rsi_cumulative_eq += 1

                if profit_rsi_cumulative > 0:
                    rsi_cumulative_profitable += 1
                if profit_rsi > 0:
                    rsi_profitable += 1

                if evaluation_rsi.num_trades > 0:
                    total_rsi += 1
                if evaluation_cumulative.num_trades > 0:
                    total_cumulative += 1

                total_evaluated_pairs += 1
            except(NoPriceDataException):
                print("Error obtaining price, skipping...")

    print (total_rsi_cumulative_better)
    print("Total evaluated pairs: {}".format(total_evaluated_pairs))
    print("RSI cumulative outperforms RSI: {0:0.2f}, equal: {1:0.2f}".format(
        total_rsi_cumulative_better / total_evaluated_pairs,
        total_rsi_cumulative_eq / total_evaluated_pairs))
    print("Sprofit: {}".format(sprofit))
    print("RSI was profitable for {0:0.2f}% of pairs".format(rsi_profitable / total_rsi * 100))
    print("RSI cumulative was profitable for {0:0.2f}% of pairs".format(rsi_cumulative_profitable / total_cumulative * 100))

def evaluate_multi_any_currency(counter_currency, start_time, end_time,
                 start_cash, overbought_threshold, oversold_threshold):
    return evaluate_multi(None, counter_currency, start_time, end_time,
                 1000, 0, overbought_threshold, oversold_threshold)


if __name__ == "__main__":
    end_time = 1531699200
    start_time = end_time - 60*60*24*70
    evaluate_rsi_signature("BTC", "USDT", start_time, end_time, 1000, 0)