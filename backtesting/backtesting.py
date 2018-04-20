from evaluation import *


def evaluate_rsi(transaction_currency, counter_currency, start_time, end_time,
                 start_cash, start_crypto, overbought_threshold, oversold_threshold):
    rsi_signals = get_signals(SignalType.RSI, transaction_currency, start_time, end_time, counter_currency)
    rsi_strategy = SimpleRSIStrategy(rsi_signals, overbought_threshold, oversold_threshold)
    evaluation = Evaluation(rsi_strategy, transaction_currency, counter_currency, start_cash, start_crypto, start_time, end_time, False)

def evaluate_rsi_any_currency(counter_currency, start_time, end_time,
                 start_cash, start_crypto, overbought_threshold, oversold_threshold):
    rsi_signals = get_filtered_signals(signal_type=SignalType.RSI, start_time=start_time, counter_currency=counter_currency)
    rsi_strategy = SimpleRSIStrategy(rsi_signals, overbought_threshold, oversold_threshold)
    evaluation = Evaluation(rsi_strategy, "NOP", counter_currency, start_cash, start_crypto, start_time, end_time, False)

def evaluate_all_signals_any_currency(counter_currency, start_time, end_time,
                 start_cash, start_crypto, overbought_threshold, oversold_threshold):
    signals = get_filtered_signals(start_time=start_time, counter_currency=counter_currency)
    rsi_strategy = SimpleRSIStrategy(rsi_signals, overbought_threshold, oversold_threshold)
    evaluation = Evaluation(rsi_strategy, "NOP", counter_currency, start_cash, start_crypto, start_time, end_time, False)

def evaluate_trend_based(signal_type, transaction_currency, counter_currency, start_time, end_time,
                 start_cash, start_crypto):
    signals = get_signals(signal_type, transaction_currency, start_time, end_time, counter_currency)
    strategy = SimpleTrendBasedStrategy(signals, signal_type)
    evaluation = Evaluation(strategy, transaction_currency, counter_currency, start_cash, start_crypto, start_time, end_time, False)


def evaluate_rsi_comparatively(transaction_currency, counter_currency, start_time, end_time,
                 start_cash, start_crypto):
    rsi_signals = get_signals(SignalType.RSI, transaction_currency, start_time, end_time, counter_currency)
    oversold = [20, 25, 30]
    overbought = [70, 75, 80]
    print("Start time: {}".format(datetime_from_timestamp(start_time)))
    print("End time: {}".format(datetime_from_timestamp(end_time)))
    print("Start price: {}".format(get_price(transaction_currency, start_time, counter_currency)))
    print("End price: {}".format(get_price(transaction_currency, end_time, counter_currency)))

    for overbought_threshold in overbought:
        for oversold_threshold in oversold:
            rsi_strategy = SimpleRSIStrategy(rsi_signals, overbought_threshold, oversold_threshold)
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

    rsi_signals = get_filtered_signals(signal_type=SignalType.RSI, start_time=start_time, counter_currency=counter_currency)

    rsi_signals = get_signals(SignalType.RSI, transaction_currency, start_time, end_time, counter_currency)
    rsi_strategy = SimpleRSIStrategy(rsi_signals, overbought_threshold, oversold_threshold)

    sma_signals = get_signals(SignalType.SMA, transaction_currency, start_time, end_time, counter_currency)
    sma_strategy = SimpleTrendBasedStrategy(sma_signals, SignalType.SMA)

    kumo_signals = get_signals(SignalType.kumo_breakout, transaction_currency, start_time, end_time, counter_currency)
    kumo_strategy = SimpleTrendBasedStrategy(kumo_signals, SignalType.kumo_breakout)

    rsi_c_signals = get_signals(SignalType.RSI_Cumulative, transaction_currency, start_time, end_time, counter_currency)
    rsi_c_strategy = SimpleTrendBasedStrategy(rsi_c_signals, SignalType.RSI_Cumulative)

    buy = (rsi_c_strategy, rsi_strategy )
    sell = (sma_strategy, kumo_strategy)

    horizon = Horizon.any
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


def evaluate_multi_any_currency(counter_currency, start_time, end_time,
                 start_cash, overbought_threshold, oversold_threshold):

    rsi_signals = get_filtered_signals(signal_type=SignalType.RSI, start_time=start_time,
                                       end_time=end_time, counter_currency=counter_currency)
    rsi_strategy = SimpleRSIStrategy(rsi_signals, overbought_threshold, oversold_threshold)

    sma_signals = get_filtered_signals(signal_type=SignalType.SMA, start_time=start_time,
                                    end_time=end_time, counter_currency=counter_currency)
    sma_strategy = SimpleTrendBasedStrategy(sma_signals, SignalType.SMA)

    kumo_signals = get_filtered_signals(signal_type=SignalType.kumo_breakout, start_time=start_time,
                                       end_time=end_time, counter_currency=counter_currency)
    kumo_strategy = SimpleTrendBasedStrategy(kumo_signals, SignalType.kumo_breakout)

    rsi_c_signals = get_filtered_signals(signal_type=SignalType.RSI_Cumulative, start_time=start_time,
                                       end_time=end_time, counter_currency=counter_currency)
    rsi_c_strategy = SimpleTrendBasedStrategy(rsi_c_signals, SignalType.RSI_Cumulative)

    buy = (rsi_c_strategy, sma_strategy, kumo_strategy, rsi_strategy)
    sell = (rsi_c_strategy, sma_strategy, kumo_strategy, rsi_strategy)

    horizon = Horizon.any
    multi_strat = MultiSignalStrategy(buy, sell, horizon)
    buy_and_hold = BuyAndHoldStrategy(multi_strat)

    print("Started evaluation")
    evaluation = Evaluation(multi_strat, None, counter_currency, start_cash, 0,
                            start_time,
                            end_time, False, True)
    baseline = Evaluation(buy_and_hold, None, counter_currency, start_cash, 0,
                          start_time,
                          end_time, False, True)

    print(evaluation.get_report())
    print(baseline.get_report())

if __name__ == "__main__":
    start, end = get_timestamp_range()
    evaluate_multi_any_currency("BTC", start, end, 1000, 70, 30)

    evaluate_multi("ETH", "USDT", start, end, 1000, 0, 70, 30)

    start_time = '1513186735.51707'
    end_time = '1513197243.96346'
    transaction_currency = "BTC"
    counter_currency = "USDT"

    rsi_signals = get_signals(SignalType.RSI, transaction_currency, start_time, end_time, counter_currency)
    rsi_strat = SimpleRSIStrategy(rsi_signals)
    sma_signals = get_signals(SignalType.SMA, transaction_currency, start_time, end_time, counter_currency)
    sma_strat = SimpleTrendBasedStrategy(sma_signals, SignalType.SMA)

    multi_strat = MultiSignalStrategy((rsi_strat,sma_strat),(rsi_strat,))

    start, end = get_timestamp_range()
    evaluate_trend_based(SignalType.SMA, "OMG", "BTC", start, end, 1, 0)
    evaluate_rsi("OMG", "BTC", 0, end, 1, 0, 20, 75)
    evaluate_rsi_comparatively("BTC", "USDT", start, end, 1000, 0)