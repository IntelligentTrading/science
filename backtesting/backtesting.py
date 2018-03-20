from strategies import *
from data_sources import *
from evaluation import *
from utils import MAX_TIME


def evaluate_rsi(transaction_currency, counter_currency, start_time, end_time,
                 start_cash, start_crypto, overbought_threshold, oversold_threshold):
    rsi_signals = get_rsi_signals(transaction_currency, start_time, end_time, counter_currency)
    rsi_strategy = SimpleRSIStrategy(rsi_signals, overbought_threshold, oversold_threshold)
    evaluation = Evaluation(rsi_strategy, transaction_currency, counter_currency, start_cash, start_crypto, start_time, end_time)

if __name__ == "__main__":
    evaluate_rsi("ETH", "USDT", 1513183102.19288, 1514183102.19288, 1000, 0, 80, 25)



