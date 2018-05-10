from evaluation import *

### Various sample backtesting runs

def evaluate_rsi(transaction_currency, counter_currency, start_time, end_time,
                 start_cash, start_crypto, horizon=Horizon.any):
    rsi_strategy = SignalSignatureStrategy(['rsi_buy_2', 'rsi_sell_2'], start_time, end_time, horizon, counter_currency, transaction_currency)
    evaluation = Evaluation(rsi_strategy, transaction_currency, counter_currency,
                            start_cash, start_crypto, start_time, end_time, True, False)
    print(evaluation.get_report())
    return evaluation


def evaluate_rsi_any_currency(counter_currency, start_time, end_time,
                 start_cash, start_crypto, overbought_threshold, oversold_threshold):
    rsi_strategy = SimpleRSIStrategy(start_time, end_time, Horizon.short, counter_currency, overbought_threshold, oversold_threshold)
    evaluation = Evaluation(rsi_strategy, "", counter_currency, start_cash, start_crypto, start_time, end_time, False)
    print(evaluation.get_report())
    return evaluation


def evaluate_trend_based(signal_type, transaction_currency, counter_currency, start_time, end_time,
                 start_cash, start_crypto, horizon=Horizon.any, strength=Strength.short):
    strategy = SimpleTrendBasedStrategy(signal_type, start_time, end_time, horizon, counter_currency, transaction_currency, strength)
    evaluation = Evaluation(strategy, transaction_currency, counter_currency, start_cash, start_crypto, start_time, end_time, False, False)
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
            baseline = BuyAndHoldStrategy(rsi_strategy)
            baseline_evaluation = Evaluation(baseline, transaction_currency, counter_currency, start_cash,
                                             start_crypto, start_time, end_time, False)
            rsi_evaluation = Evaluation(rsi_strategy, transaction_currency, counter_currency, start_cash,
                                        start_crypto, start_time, end_time, False)
            print("RSI overbought = {}, oversold = {}".format(overbought_threshold, oversold_threshold))
            print("  Profit - RSI buy and hold: {0:.2f}%".format(baseline_evaluation.get_profit_percent()))
            print("  Profit - RSI trading: {0:.2f}% ({1} trades)\n".format(rsi_evaluation.get_profit_percent(), rsi_evaluation.num_trades))


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
    buy_and_hold = BuyAndHoldStrategy(multi_strat)
    evaluation = Evaluation(multi_strat, transaction_currency, counter_currency, start_cash, start_crypto,
                            start_time,
                            end_time, False, True)
    baseline = Evaluation(buy_and_hold, transaction_currency, counter_currency, start_cash, start_crypto,
                          start_time,
                          end_time, False, True)

    print(evaluation.get_report())
    print(baseline.get_report())


def evaluate_rsi_cumulative_compare(start_time, end_time, transaction_currency, counter_currency, horizon=Horizon.any):
    start_cash = 100000
    start_crypto = 0

    rsi = evaluate_rsi(transaction_currency, counter_currency, start_time, end_time,
                     start_cash, start_crypto, 75, 30, horizon)

    rsi_cumulative = evaluate_trend_based(SignalType.RSI_Cumulative, transaction_currency, counter_currency,
                                          start_time, end_time, start_cash, start_crypto, horizon)

    f = open("rsi.txt", "w")
    f.write(rsi.get_report(True))
    f.close()

    f = open("rsi_cumulative.txt", "w")
    f.write(rsi_cumulative.get_report(True))
    f.close()

    print("cumulative: {}, rsi: {}".format(rsi_cumulative.get_profit_percent(), rsi.get_profit_percent()))
    return rsi_cumulative.get_profit_percent(), rsi.get_profit_percent()


def find_num_cumulative_outperforms(start_time, end_time, counter_currency):
    transaction_currencies = get_currencies_for_signal(counter_currency, "RSI_Cumulative")
    horizons = (Horizon.short, Horizon.medium, Horizon.long)
    total_rsi_cumulative_better = 0
    total_rsi_cumulative_eq = 0
    total = 0
    for transaction_currency in transaction_currencies:
        for horizon in horizons:
            try:
                cumulative, rsi = evaluate_rsi_cumulative_compare(start_time, end_time, transaction_currency, counter_currency, horizon)
                if cumulative == 0 and rsi == 0:
                    continue
                if cumulative > rsi:
                    total_rsi_cumulative_better += 1
                elif cumulative == rsi:
                    total_rsi_cumulative_eq += 1

                total += 1

            except(NoPriceDataException):
                print("Error in price")
    print (total_rsi_cumulative_better)
    print("Total: {}".format(total))
    print("Outperforms: {}, equal: {}".format(total_rsi_cumulative_better/total, total_rsi_cumulative_eq/total))


def evaluate_multi_any_currency(counter_currency, start_time, end_time,
                 start_cash, overbought_threshold, oversold_threshold):
    return evaluate_multi(None, counter_currency, start_time, end_time,
                 1000, 0, overbought_threshold, oversold_threshold)

