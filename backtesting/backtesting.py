from strategies import *
from data_sources import *
from evaluation import *


def evaluate_rsi(transaction_currency, counter_currency, start_time, end_time,
                 start_cash, start_crypto, overbought_threshold, oversold_threshold):
    rsi_signals = get_rsi_signals(transaction_currency, start_time, end_time, counter_currency)
    rsi_strategy = SimpleRSIStrategy(rsi_signals, overbought_threshold, oversold_threshold)
    evaluation = Evaluation(rsi_strategy, transaction_currency, counter_currency, start_cash, start_crypto, start_time, end_time, False)


def evaluate_rsi_comparatively(transaction_currency, counter_currency, start_time, end_time,
                 start_cash, start_crypto):
    rsi_signals = get_rsi_signals(transaction_currency, start_time, end_time, counter_currency)
    oversold = [20, 25, 30]
    overbought = [70, 75, 80]
    print("Start time: {}".format(datetime_from_timestamp(start_time)))
    print("End time: {}".format(datetime_from_timestamp(end_time)))
    print("Start price: {}".format(get_price(transaction_currency, start_time, counter_currency)))
    print("End price: {}".format(get_price(transaction_currency, end_time, counter_currency)))

    for overbought_threshold in overbought:
        for oversold_threshold in oversold:
            baseline = RSIBuyAndHoldStrategy(rsi_signals, overbought_threshold, oversold_threshold)
            rsi_strategy = SimpleRSIStrategy(rsi_signals, overbought_threshold, oversold_threshold)
            baseline_evaluation = Evaluation(baseline, transaction_currency, counter_currency, start_cash,
                                             start_crypto, start_time, end_time, False)
            rsi_evaluation = Evaluation(rsi_strategy, transaction_currency, counter_currency, start_cash,
                                        start_crypto, start_time, end_time, False)
            print("RSI overbought = {}, oversold = {}".format(overbought_threshold, oversold_threshold))
            print("  Profit - RSI buy and hold: {0:.2f}%".format(baseline_evaluation.profit_percent))
            print("  Profit - RSI trading: {0:.2f}% ({1} trades)\n".format(rsi_evaluation.profit_percent, rsi_evaluation.num_trades))



if __name__ == "__main__":
    start, end = get_timestamp_range()
    evaluate_rsi("OMG", "BTC", start, end, 1, 0, 25, 70)
    #evaluate_rsi_comparatively("BTC", "USDT", start, end, 10000, 0)



