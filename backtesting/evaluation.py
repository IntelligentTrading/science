from data_sources import *
from orders import *
from utils import *
import logging
from config import transaction_cost_percents
logging.getLogger().setLevel(logging.INFO)
from abc import ABC, abstractmethod
import empyrical
import numpy as np
from charting import BacktestingChart


class Evaluation(ABC):
    def __init__(self, strategy, transaction_currency, counter_currency,
                 start_cash, start_crypto, start_time, end_time, source=0,
                 resample_period=60, evaluate_profit_on_last_order=True, verbose=True,
                 benchmark_backtest=None):
        self._strategy = strategy
        self._transaction_currency = transaction_currency
        self._counter_currency = counter_currency
        self._start_cash = start_cash
        self._start_crypto = start_crypto
        self._start_time = start_time
        self._end_time = end_time
        self._source = source
        self._resample_period = resample_period
        self._evaluate_profit_on_last_order = evaluate_profit_on_last_order
        self._transaction_cost_percent = transaction_cost_percents[source]
        self._verbose = verbose
        self._benchmark_backtest = benchmark_backtest

        if benchmark_backtest is not None:
            assert benchmark_backtest._start_time == self._start_time
            assert benchmark_backtest._end_time == self._end_time

        # Init backtesting variables
        self._cash = start_cash
        self._crypto = start_crypto
        self._num_trades = 0
        self._num_buys = 0
        self._num_sells = 0
        self.orders = []
        self.order_signals = []
        self.trading_df = pd.DataFrame(columns=['close_price', 'signal', 'order', 'cash', 'crypto', 'total_value'])

        # TODO: this goes out

        self.num_profitable_trades = 0
        self.invested_on_buy = 0
        self.avg_profit_per_trade_pair = 0


    @property
    def noncumulative_returns(self):
        return self.trading_df['return_relative_to_past_tick']


    @abstractmethod
    def run(self):
        pass

    @property
    def num_buys(self):
        return self._num_buys

    @property
    def num_sells(self):
        return self._num_sells

    @property
    def num_orders(self):
        return len(self.orders)

    @property
    def num_trades(self):
        return self._num_trades

    @property
    def start_value_usdt(self):
        try:
            start_value_USDT = convert_value_to_USDT(self._start_cash, self._start_time,
                                                     self._counter_currency, self._source)
            if self._start_crypto > 0 and self._transaction_currency is not None:
                start_value_USDT += convert_value_to_USDT(self._start_crypto, self._start_time,
                                                          self.start_crypto_currency, self._source)
            return start_value_USDT
        except NoPriceDataException:
            return None

    @property
    def end_value_usdt(self):
        try:
            end_value_USDT = convert_value_to_USDT(self.end_cash, self._end_time, self._counter_currency, self._source) + \
                             convert_value_to_USDT(self.end_crypto, self._end_time, self._end_crypto_currency, self._source)
            return end_value_USDT
        except NoPriceDataException:
            return None

    @property
    def profit_usdt(self):
        if self.start_value_usdt is None or self.end_value_usdt is None:
            return None
        else:
            return self.end_value_usdt - self.start_value_usdt

    @property
    def profit_percent_usdt(self):
        if self.profit_usdt is None or self.start_value_usdt is None:
            return None
        else:
            return self.profit_usdt / self.start_value_usdt * 100

    @property
    def start_value(self):
        try:
            return self._start_cash + \
                   (self._start_crypto * get_price(
                   self.start_crypto_currency,
                   self._start_time,
                   self._source,
                   self._counter_currency) if self._start_crypto > 0 else 0)
                   # because more often than not we start with 0 crypto and at the "beginning of time"
        except NoPriceDataException:
            return None

    @property
    def end_cash(self):
        return self._end_cash

    @property
    def end_crypto(self):
        return self._end_crypto

    @property
    def end_value(self):
        try:
            return self.end_cash + self.end_price * self.end_crypto
        except:
            return None

    @property
    def end_price(self):
        if self._evaluate_profit_on_last_order and not self.trading_df.empty:
            return self.trading_df.tail(1)['close_price'].item()
        else:
            try:
                return get_price(self._transaction_currency, self._end_time, self._source, self._counter_currency)
            except:
                return None

    @property
    def profit(self):
        if self.end_value is None or self.start_value is None:
            return None
        else:
            return self.end_value - self.start_value

    @property
    def profit_percent(self):
        if self.profit is None or self.start_value is None:
            return None
        else:
            return self.profit/self.start_value*100

    @property
    def max_drawdown(self):
        return self._max_drawdown

    @property
    def sharpe_ratio(self):
        return self._sharpe_ratio

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def min_buy_sell_pair_gain(self):
        return self._buy_sell_pair_gains.min()

    @property
    def max_buy_sell_pair_gain(self):
        return self._buy_sell_pair_gains.max()

    @property
    def mean_buy_sell_pair_gain(self):
        return self._buy_sell_pair_gains.mean()

    @property
    def std_buy_sell_pair_gain(self):
        return self._buy_sell_pair_gains.std()

    @property
    def min_buy_sell_pair_loss(self):
        return self._buy_sell_pair_losses.min()

    @property
    def max_buy_sell_pair_loss(self):
        return self._buy_sell_pair_losses.max()

    @property
    def mean_buy_sell_pair_loss(self):
        return self._buy_sell_pair_losses.mean()

    @property
    def std_buy_sell_pair_loss(self):
        return self._buy_sell_pair_losses.std()

    @property
    def num_buy_sell_pairs(self):
        return self._num_sells

    @property
    def percent_profitable_trades(self):
        num_gains = len(self._buy_sell_pair_gains) \
            if not (len(self._buy_sell_pair_gains) == 1 and np.isnan(self._buy_sell_pair_gains[0])) else 0
        return (num_gains / self.num_buy_sell_pairs) if self.num_buy_sell_pairs != 0 else 0

    @property
    def percent_unprofitable_trades(self):
        num_losses = len(self._buy_sell_pair_losses) \
            if not (len(self._buy_sell_pair_losses) == 1 and np.isnan(self._buy_sell_pair_losses[0])) else 0
        return (num_losses / self.num_buy_sell_pairs) if self.num_buy_sell_pairs != 0 else 0


    def _write_to_trading_df(self):
        total_value = self._crypto * self._current_price + self._cash

        # fill a row in the trading dataframe
        self.trading_df.loc[self._current_timestamp] = pd.Series({
            'close_price': self._current_price,
            'cash': self._cash,
            'crypto': self._crypto,
            'total_value': total_value,
            'order': "" if self._current_order is None else self._current_order.order_type.value,
            'signal': "" if self._current_order is None or self._current_signal is None else self._current_signal.signal_type})

    def _finalize_backtesting(self):
        # set finishing variable values
        self._end_cash = self._cash
        self._end_crypto = self._crypto

        # compute returns for stats
        self.trading_df = self._fill_returns(self.trading_df)
        returns = np.array(self.trading_df['return_relative_to_past_tick'])
        self._max_drawdown = empyrical.max_drawdown(np.array(returns))
        self._sharpe_ratio = empyrical.sharpe_ratio(returns)

        # extract only rows that have orders
        orders_df = self.trading_df[self.trading_df['order'] != ""]
        # recalculate returns
        orders_df = self._fill_returns(orders_df)
        # get profits on sell
        orders_sell_df = orders_df[orders_df['order'] == "SELL"]
        self._buy_sell_pair_returns = np.array(orders_sell_df['return_relative_to_past_tick'])
        self._buy_sell_pair_gains = self._buy_sell_pair_returns[np.where(self._buy_sell_pair_returns > 0)]
        self._buy_sell_pair_losses = self._buy_sell_pair_returns[np.where(self._buy_sell_pair_returns < 0)]

        # if no returns, no gains or no losses, stat functions will return nan
        if len(self._buy_sell_pair_returns) == 0:
            self._buy_sell_pair_returns = np.array([np.nan])

        if len(self._buy_sell_pair_gains) == 0:
            self._buy_sell_pair_gains = np.array([np.nan])

        if len(self._buy_sell_pair_losses) == 0:
            self._buy_sell_pair_losses = np.array([np.nan])

        if self._benchmark_backtest is not None:
            self._alpha, self._beta = \
                empyrical.alpha_beta(self.noncumulative_returns, self._benchmark_backtest.noncumulative_returns)
        else:
            self._alpha = self._beta = np.nan

        if self._verbose:
            logging.info(self.get_report())
            # logging.info(self.trading_df)
            # self.plot_portfolio()

    def _fill_returns(self, df):
        df['return_from_initial_investment'] = (df['total_value'] - self.start_value) / self.start_value
        df['return_relative_to_past_tick'] = df['total_value'].diff() / df['total_value'].shift(1)
        return df


    def get_orders(self):
        return self.orders

    def _format_price_dependent_value(self, value):
        if value is None:
            return float('nan')
        else:
            return value

    def get_report(self, include_order_signals=True):
        output = []
        output.append(str(self._strategy))

        # output.append(self.strategy.get_signal_report())
        output.append("--")

        output.append("\n* Order execution log *\n")
        output.append("Start balance: cash = {} {}, crypto = {} {}".format(self._start_cash, self._counter_currency,
                                                                           self._start_crypto, self.start_crypto_currency
                                                                           if self._start_crypto != 0 else ""))

        output.append("Start time: {}\n--".format(datetime_from_timestamp(self._start_time)))
        output.append("--")

        for i, order in enumerate(self.orders):
            output.append(str(order))
            if include_order_signals and len(self.order_signals) == len(self.orders): # for buy & hold we don't have signals
                output.append("   signal: {}".format(self.order_signals[i]))

        output.append("End time: {}".format(datetime_from_timestamp(self._end_time)))
        output.append("\nSummary")
        output.append("--")
        output.append("Number of trades: {}".format(self._num_trades))
        output.append("End cash: {0:.2f} {1}".format(self.end_cash, self._counter_currency))
        output.append("End crypto: {0:.6f} {1}".format(self.end_crypto, self._transaction_currency))

        sign = "+" if self.profit != None and self.profit >= 0 else ""
        output.append("Total value invested: {} {}".format(self._format_price_dependent_value(self.start_value),
                                                           self._counter_currency))
        output.append(
            "Total value after investment: {0:.2f} {1} ({2}{3:.2f}%)".format(self._format_price_dependent_value(self.end_value),
                                                                             self._counter_currency,
                                                                             sign,
                                                                             self._format_price_dependent_value(self.profit_percent)))
        output.append("Profit: {0:.2f} {1}".format(self._format_price_dependent_value(self.profit), self._counter_currency))

        if self._counter_currency != "USDT":
            sign = "+" if self.profit_usdt is not None and self.profit_usdt >= 0 else ""
            output.append("Total value invested: {:.2f} {} (conversion on {})".format(
                self._format_price_dependent_value(self.start_value_usdt),
                "USDT",
                datetime_from_timestamp(self._start_time)))
            output.append(
                    "Total value after investment: {0:.2f} {1} ({2}{3:.2f}%) (conversion on {4})".format(
                        self._format_price_dependent_value(self.end_value_usdt), "USDT", sign,
                        self._format_price_dependent_value(self.profit_percent_usdt),
                        datetime_from_timestamp(self._end_time)))
            output.append("Profit: {0:.2f} {1}".format(self._format_price_dependent_value(self.profit_usdt),
                                                       "USDT"))

        output.append("\nAdditional stats:")
        output.append("  Max drawdown: {}".format(self.max_drawdown))
        output.append("  Sharpe ratio: {}".format(self.sharpe_ratio))
        output.append("  Alpha: {}".format(self.alpha))
        output.append("  Beta: {}".format(self.beta))

        output.append("  Buy-sell pair gains - overall stats")
        output.append("     min = {}, max = {}, mean = {}, stdev = {}".format(
            self.min_buy_sell_pair_gain,
            self.max_buy_sell_pair_gain,
            self.mean_buy_sell_pair_gain,
            self.std_buy_sell_pair_gain
        ))

        output.append("  Buy-sell pair losses - overall stats")
        output.append("     min = {}, max = {}, mean = {}, stdev = {}".format(
            self.min_buy_sell_pair_loss,
            self.max_buy_sell_pair_loss,
            self.mean_buy_sell_pair_loss,
            self.std_buy_sell_pair_loss
        ))

        output.append("  Total buy-sell pairs: {}".format(self.num_buy_sell_pairs))
        output.append("  Total profitable trades: {}".format(self.num_profitable_trades))
        output.append("  Percent profitable trades: {}".format(self.percent_profitable_trades))
        output.append("  Percent unprofitable trades: {}".format(self.percent_unprofitable_trades))
        
        return "\n".join(output)

    def get_short_summary(self):
        return ("{} \t Invested: {} {}, {} {}\t After investment: {:.2f} {}, {:.2f} {} \t Profit: {}{:.2f}%".format(
            self._strategy.get_short_summary(),
            self._start_cash, self._counter_currency, self._start_crypto, self.start_crypto_currency,
            self.end_cash, self._counter_currency, self.end_crypto, self.end_crypto_currency,
            "+" if self.profit_percent is not None and self.profit_percent >= 0 else "",
            self._format_price_dependent_value(self.profit_percent)))

    def execute_order(self, order):
        delta_crypto, delta_cash = order.execute()
        self._cash += delta_cash
        self._crypto += delta_crypto
        self._num_trades += 1
        if order.order_type == OrderType.BUY:
            self.invested_on_buy = -delta_cash
            self._buy_currency = order.transaction_currency
            self._num_buys += 1
        elif order.order_type == OrderType.SELL:
            # the currency we're selling must match the bought currency
            assert order.transaction_currency == self._buy_currency
            self._num_sells += 1
            buy_sell_pair_profit_percent = (delta_cash - self.invested_on_buy) / self.invested_on_buy * 100
            self.avg_profit_per_trade_pair += buy_sell_pair_profit_percent
            if buy_sell_pair_profit_percent > 0:
                self.num_profitable_trades += 1


    def to_dictionary(self):
        dictionary = vars(self).copy()
        # remove trailing underscores
        tmp = {(k[1:] if k.startswith("_") else k): dictionary[k] for k in dictionary.keys()}
        dictionary = tmp
        del dictionary["orders"]
        dictionary["strategy"] = dictionary["strategy"].get_short_summary()
        dictionary["utilized_signals"] = ", ".join(get_distinct_signal_types(self.order_signals))
        dictionary["start_time"] = datetime_from_timestamp(dictionary["start_time"])
        dictionary["end_time"] = datetime_from_timestamp(dictionary["end_time"])

        dictionary["transaction_currency"] = self._end_crypto_currency
        if "horizon" not in vars(self._strategy):
            dictionary["horizon"] = "N/A"
        else:
            dictionary["horizon"] = self._strategy.horizon.name

        if self.end_price == None:
            dictionary["profit"] = "N/A"
            dictionary["profit_percent"] = "N/A"
            dictionary["profit_USDT"] = "N/A"
            dictionary["profit_percent_USDT"] = "N/A"
        else:
            try:
                dictionary["profit"] = self.profit
                dictionary["profit_percent"] = self.profit_percent
                dictionary["profit_USDT"] = self.profit_usdt
                dictionary["profit_percent_USDT"] = self.profit_percent_usdt
            except NoPriceDataException:
                logging.error("No price data!")
                dictionary["profit"] = "N/A"
                dictionary["profit_percent"] = "N/A"
                dictionary["profit_USDT"] = "N/A"
                dictionary["profit_percent_USDT"] = "N/A"
        return dictionary

    def plot_cumulative_returns(self):
        if self.trading_df.empty:
            return
        chart = BacktestingChart(self, self._benchmark_backtest)
        chart.draw_returns_tear_sheet()

    def plot_returns_tear_sheet(self):
        if self.trading_df.empty:
            return
        chart = BacktestingChart(self, self._benchmark_backtest)
        chart.draw_returns_tear_sheet()




if __name__ == '__main__':
    from strategies import RSITickerStrategy
    end_time = 1531699200
    start_time = end_time - 60*60*24*7
    start_cash = 10000000
    start_crypto = 0
    strategy = RSITickerStrategy(start_time, end_time, Horizon.short, None)
    evaluation = Evaluation(strategy, 'BTC', 'USDT', start_cash, start_crypto, start_time, end_time)
    #evaluation.simulate_events()
