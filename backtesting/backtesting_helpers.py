from backtester_signals import SignalDrivenBacktester
from strategies import *

### Various sample backtesting runs

def evaluate_rsi_signature(**kwargs):
    rsi_strategy = SignalSignatureStrategy(
        kwargs['source'],
        ['rsi_buy_2', 'rsi_sell_2','rsi_buy_1', 'rsi_sell_1','rsi_buy_3', 'rsi_sell_3'])
    return SignalDrivenBacktester(strategy=rsi_strategy, **kwargs)


def evaluate_rsi(overbought_threshold, oversold_threshold, signal_type="RSI", **kwargs):
    rsi_strategy = SimpleRSIStrategy(kwargs['source'], overbought_threshold, oversold_threshold, signal_type)
    return rsi_strategy.evaluate(**kwargs)


def evaluate_trend_based(signal_type, **kwargs):
    strategy = SimpleTrendBasedStrategy(kwargs['source'], signal_type)
    return SignalDrivenBacktester(strategy=strategy, **kwargs)


# TODO: look into this
def evaluate_rsi_any_currency(counter_currency, start_time, end_time,
                 start_cash, start_crypto, overbought_threshold, oversold_threshold):
    rsi_strategy = SimpleRSIStrategy(start_time, end_time, Horizon.short, counter_currency, overbought_threshold, oversold_threshold)
    evaluation = SignalDrivenBacktester(rsi_strategy, "", counter_currency, start_cash, start_crypto, start_time, end_time, False)
    print(evaluation.get_report())
    return evaluation


def evaluate_rsi_cumulative_compare(overbought_threshold, oversold_threshold, **kwargs):
    evaluation_rsi = evaluate_rsi(overbought_threshold, oversold_threshold, "RSI", **kwargs)
    evaluation_rsi_cumulative = evaluate_rsi(overbought_threshold, overbought_threshold, "RSI_Cumulative", **kwargs)
    return evaluation_rsi_cumulative, evaluation_rsi


def find_num_cumulative_outperforms(currency_pairs, **kwargs):
    resample_periods = (60, 240, 1440)
    total_rsi_cumulative_better = 0
    total_rsi_cumulative_eq = 0
    total_evaluated_pairs = 0
    rsi_profitable = 0
    rsi_cumulative_profitable = 0
    total_rsi = 0
    total_cumulative = 0
    for transaction_currency, counter_currency in currency_pairs:
        for resample_period in resample_periods:
            try:
                kwargs['resample_period'] = resample_period
                kwargs['transaction_currency'] = transaction_currency
                kwargs['counter_currency'] = counter_currency
                evaluation_cumulative, evaluation_rsi = evaluate_rsi_cumulative_compare(75, 25, **kwargs)

                profit_rsi_cumulative = evaluation_cumulative.profit_percent
                profit_rsi = evaluation_rsi.profit_percent

                if evaluation_cumulative.num_orders == 0 and evaluation_rsi.num_orders == 0:
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

    print("Total evaluated pairs: {}".format(total_evaluated_pairs))
    print("RSI cumulative outperforms RSI: {0:0.2f}, equal: {1:0.2f}".format(
        total_rsi_cumulative_better / total_evaluated_pairs,
        total_rsi_cumulative_eq / total_evaluated_pairs))
    print("RSI was profitable for {0:0.2f}% of pairs".format(rsi_profitable / total_rsi * 100))
    print("RSI cumulative was profitable for {0:0.2f}% of pairs".format(rsi_cumulative_profitable / total_cumulative * 100))



if __name__ == "__main__":
    end_time = 1531699200
    start_time = end_time - 60*60*24*70

    kwargs = {}
    kwargs['transaction_currency'] = 'BTC'
    kwargs['counter_currency'] = 'USDT'
    kwargs['start_time'] =  start_time
    kwargs['end_time'] = end_time
    kwargs['start_cash'] = 1000
    kwargs['start_crypto'] = 0
    kwargs['source'] = 0
    kwargs['resample_period'] = 60
    kwargs['time_delay'] = 0

    evaluate_rsi_signature(**kwargs)