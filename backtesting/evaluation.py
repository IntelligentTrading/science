from data_sources import *
from orders import *
from utils import *
import logging
logging.getLogger().setLevel(logging.INFO)


backtesting_report_columns = ["strategy",
                              "utilized_signals",
                              "transaction_currency",
                              "counter_currency",
                              "num_trades",
                              "profit_percent",
                              "profit_percent_USDT",
                              "buy_and_hold_profit_percent",
                              "buy_and_hold_profit_percent_USDT",
                              "start_time",
                              "end_time",
                              "evaluate_profit_on_last_order",
                              "horizon",
                              "num_profitable_trades",
                              "avg_profit_per_trade_pair",
                              "num_sells"]


class Evaluation:
    def __init__(self, strategy, transaction_currency, counter_currency,
                 start_cash, start_crypto, start_time, end_time, source=0,
                 resample_period=60, evaluate_profit_on_last_order=False):
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.start_time = start_time
        self.end_time = end_time
        self.strategy = strategy
        self.source = source
        self.evaluate_profit_on_last_order = evaluate_profit_on_last_order
        # self.orders, self.order_signals = strategy.get_orders(start_cash=start_cash, start_crypto=start_crypto,
        #                                                      time_delay=time_delay)

        self.signals = get_filtered_signals(start_time=start_time, end_time=end_time, counter_currency=counter_currency,
                                            transaction_currency=transaction_currency,
                                            resample_period=resample_period, source=source, return_df=True)
        self.price_data = get_resampled_prices_in_range(start_time, end_time, transaction_currency, counter_currency, resample_period)

        # self.execute_orders(self.orders)

        self.simulate_events()

        # TODO: reports
        # if verbose:
        #    print(self.get_report())
        # if self.end_price is None:
        #   raise NoPriceDataException()

    def simulate_events(self):
        transaction_cost_percent = 0.02 # TODO move

        # connect prices and signals in time (find signal nearest in time to price tick)
        orders = []
        order_signals = []
        cash = self.start_cash
        crypto = self.start_crypto
        prices_df = self.price_data
        signals_df = self.signals

        # reindex signals_df to match prices
        prices_df.reset_index(level=0, inplace=True)
        signals_df.reset_index(level=0, inplace=True)
        reindexed_signals_df = pd.merge_asof(signals_df, prices_df, direction='nearest')

        trading_df = pd.DataFrame(columns=['close_price', 'signal', 'cash', 'crypto', 'total_value'])

        for i, row in prices_df.iterrows():
            timestamp = row['timestamp']
            price = row['close_price']
            signals_now = reindexed_signals_df[reindexed_signals_df['timestamp'] == timestamp]
            signals_now = Signal.pandas_to_objects_list(signals_now)
            decision, order_signal = self.strategy.process_ticker(row, signals_now)

            if decision == "SELL" and crypto > 0:
                order = Order(OrderType.SELL, self.transaction_currency, self.counter_currency,
                              timestamp, crypto, price, transaction_cost_percent, 0)
                orders.append(order)
                order_signals.append(order_signal)
                delta_crypto, delta_cash = order.execute()
                cash = cash + delta_cash
                crypto = crypto + delta_crypto
                assert crypto == 0

            elif decision == "BUY" and cash > 0:
                order = Order(OrderType.BUY, self.transaction_currency, self.counter_currency,
                              timestamp, cash, price, transaction_cost_percent, 0)
                orders.append(order)
                order_signals.append(order_signal)
                delta_crypto, delta_cash = order.execute()
                cash = cash + delta_cash
                crypto = crypto + delta_crypto
                assert cash == 0

            total_value = crypto * price + cash
            # compute asset value at this tick, regardless of the signal
            trading_df.loc[timestamp] = pd.Series({'close_price': price,
                                               'cash': cash,
                                               'crypto': crypto,
                                               'total_value': total_value})

        self.end_cash = cash
        self.end_crypto = crypto
        self.orders = orders
        self.order_signals = order_signals
        logging.info(trading_df)

        # TODO move plotting outside
        import matplotlib.pyplot as plt
        #trading_df.plot()
        ax1 = trading_df['close_price'].plot()
        ax2 = trading_df['total_value'].plot(secondary_y=True)

        plt.show()


    def get_start_value_USDT(self):
        try:
            start_value_USDT = convert_value_to_USDT(self.start_cash, self.start_time,
                                                     self.counter_currency, self.source)
            if self.start_crypto > 0 and self.start_crypto_currency is not None:
                start_value_USDT += convert_value_to_USDT(self.start_crypto, self.start_time,
                                                          self.start_crypto_currency, self.source)
            return start_value_USDT
        except NoPriceDataException:
            return None

    def get_end_value_USDT(self):
        try:
            end_value_USDT = convert_value_to_USDT(self.end_cash, self.end_time, self.counter_currency, self.source) + \
                             convert_value_to_USDT(self.end_crypto, self.end_time, self.end_crypto_currency, self.source)
            return end_value_USDT
        except NoPriceDataException:
            return None

    def get_profit_USDT(self):
        end_value = self.get_end_value_USDT()
        start_value = self.get_start_value_USDT()

        if start_value is None or end_value is None:
            return None
        else:
            return end_value - start_value

    def get_profit_percent_USDT(self):
        profit = self.get_profit_USDT()
        start_value = self.get_start_value_USDT()

        if profit is None or start_value is None:
            return None
        else:
            return profit / start_value * 100

    def get_start_value(self):
        try:
            return self.start_cash + \
               (self.start_crypto * get_price(
                   self.start_crypto_currency,
                   self.start_time,
                   self.source,
                   self.counter_currency) if self.start_crypto > 0 else 0)
                   # because more often than not we start with 0 crypto and at the "beginning of time"
        except NoPriceDataException:
            return None

    def get_end_cash(self):
        return self.end_cash

    def get_end_crypto(self):
        return self.end_crypto

    def get_end_value(self):
        try:
            return self.end_cash + self.end_price * self.end_crypto
        except:
            return None

    def get_profit_value(self):
        start_value = self.get_start_value()
        end_value = self.get_end_value()
        if end_value is None or start_value is None:
            return None
        else:
            return end_value - start_value

    def get_profit_percent(self):
        profit = self.get_profit_value()
        start_value = self.get_start_value()
        if profit is None or start_value is None:
            return None
        else:
            return profit/start_value*100

    def get_num_trades(self):
        return self.num_trades

    def get_orders(self):
        return self.orders

    def format_price_dependent_value(self, value):
        if value is None:
            return float('nan')
        else:
            return value

    def get_report(self, include_order_signals=True):
        output = []
        output.append(str(self.strategy))

        # output.append(self.strategy.get_signal_report())
        output.append("--")

        output.append("\n* Order execution log *\n")
        output.append("Start balance: cash = {} {}, crypto = {} {}".format(self.start_cash, self.counter_currency,
                                                                           self.start_crypto, self.start_crypto_currency
                                                                           if self.start_crypto != 0 else ""))

        output.append("Start time: {}\n--".format(datetime_from_timestamp(self.start_time)))
        output.append("--")

        for i, order in enumerate(self.orders):
            output.append(str(order))
            if include_order_signals and len(self.order_signals) == len(self.orders): # for buy & hold we don't have signals
                output.append("   signal: {}".format(self.order_signals[i]))

        output.append("End time: {}".format(datetime_from_timestamp(self.end_time)))
        output.append("\nSummary")
        output.append("--")
        output.append("Number of trades: {}".format(self.num_trades))
        output.append("End cash: {0:.2f} {1}".format(self.end_cash, self.counter_currency))
        output.append("End crypto: {0:.6f} {1}".format(self.end_crypto, self.end_crypto_currency))

        sign = "+" if self.get_profit_value() != None and self.get_profit_value() >= 0 else ""
        output.append("Total value invested: {} {}".format(self.format_price_dependent_value(self.get_start_value()),
                                                           self.counter_currency))
        output.append(
            "Total value after investment: {0:.2f} {1} ({2}{3:.2f}%)".format(self.format_price_dependent_value(self.get_end_value()),
                                                                             self.counter_currency,
                                                                             sign,
                                                                             self.format_price_dependent_value(self.get_profit_percent())))
        output.append("Profit: {0:.2f} {1}".format(self.format_price_dependent_value(self.get_profit_value()), self.counter_currency))

        if self.counter_currency != "USDT":
            sign = "+" if self.get_profit_USDT() is not None and self.get_profit_USDT() >= 0 else ""
            output.append("Total value invested: {:.2f} {} (conversion on {})".format(
                self.format_price_dependent_value(self.get_start_value_USDT()),
                "USDT",
                datetime_from_timestamp(self.start_time)))
            output.append(
                    "Total value after investment: {0:.2f} {1} ({2}{3:.2f}%) (conversion on {4})".format(
                        self.format_price_dependent_value(self.get_end_value_USDT()), "USDT", sign,
                        self.format_price_dependent_value(self.get_profit_percent_USDT()),
                        datetime_from_timestamp(self.end_time)))
            output.append("Profit: {0:.2f} {1}".format(self.format_price_dependent_value(self.get_profit_USDT()),
                                                       "USDT"))

        return "\n".join(output)

    def get_short_summary(self):
        return ("{} \t Invested: {} {}, {} {}\t After investment: {:.2f} {}, {:.2f} {} \t Profit: {}{:.2f}%".format(
            self.strategy.get_short_summary(),
            self.start_cash, self.counter_currency, self.start_crypto, self.start_crypto_currency,
            self.end_cash, self.counter_currency, self.end_crypto, self.end_crypto_currency,
            "+" if self.get_profit_percent() is not None and self.get_profit_percent() >= 0 else "",
            self.format_price_dependent_value(self.get_profit_percent())))

    def execute_orders(self, orders):
        cash = self.start_cash
        crypto = self.start_crypto
        num_trades = 0
        num_profitable_trades = 0
        invested_on_buy = 0
        avg_profit_per_trade_pair = 0
        num_sells = 0

        for i, order in enumerate(orders):
            if i == 0: # first order
                assert order.order_type == OrderType.BUY
                buy_currency = order.transaction_currency

            delta_crypto, delta_cash = order.execute()
            cash += delta_cash
            crypto += delta_crypto
            num_trades += 1
            if order.order_type == OrderType.BUY:
                invested_on_buy = -delta_cash
                buy_currency = order.transaction_currency
            else:
                # the currency we're selling must match the bought currency
                assert order.transaction_currency == buy_currency
                num_sells += 1
                buy_sell_pair_profit_percent = (delta_cash - invested_on_buy) / invested_on_buy * 100
                avg_profit_per_trade_pair += buy_sell_pair_profit_percent
                if buy_sell_pair_profit_percent > 0:
                    num_profitable_trades += 1

        if self.evaluate_profit_on_last_order and num_trades > 0:
            end_price = orders[-1].unit_price
        else:
            if num_trades == 0:
                logging.warning("No orders were generated by the chosen strategy.")
            try:
                if num_trades > 0:
                    end_price = get_price(buy_currency, self.end_time, self.source, self.counter_currency)
                    if orders[-1].order_type == OrderType.BUY:
                        delta_cash = cash + end_price * crypto
                        if delta_cash > invested_on_buy:
                            num_profitable_trades += 1
                            buy_sell_pair_profit_percent = (delta_cash - invested_on_buy) / invested_on_buy * 100
                            avg_profit_per_trade_pair += buy_sell_pair_profit_percent
                else:
                    end_price = get_price(self.start_crypto_currency, self.end_time, self.source, self.counter_currency)

            except NoPriceDataException:
                logging.error("No price data found")
                end_price = None

        if num_sells != 0:
            avg_profit_per_trade_pair /= num_sells

        end_crypto_currency = buy_currency if num_trades > 0 else self.start_crypto_currency

        self.num_trades = num_trades
        self.end_cash = cash
        self.end_crypto = crypto
        self.end_price = end_price
        self.num_profitable_trades = num_profitable_trades
        self.avg_profit_per_trade_pair = avg_profit_per_trade_pair
        self.num_sells = num_sells
        self.end_crypto_currency = end_crypto_currency


    def to_dictionary(self):
        dictionary = vars(self).copy()
        del dictionary["orders"]
        dictionary["strategy"] = dictionary["strategy"].get_short_summary()
        dictionary["utilized_signals"] = ", ".join(get_distinct_signal_types(self.order_signals))
        dictionary["start_time"] = datetime_from_timestamp(dictionary["start_time"])
        dictionary["end_time"] = datetime_from_timestamp(dictionary["end_time"])

        dictionary["transaction_currency"] = self.end_crypto_currency
        if "horizon" not in vars(self.strategy):
            dictionary["horizon"] = "N/A"
        else:
            dictionary["horizon"] = self.strategy.horizon.name

        if self.end_price == None:
            dictionary["profit"] = "N/A"
            dictionary["profit_percent"] = "N/A"
            dictionary["profit_USDT"] = "N/A"
            dictionary["profit_percent_USDT"] = "N/A"
        else:
            try:
                dictionary["profit"] = self.get_profit_value()
                dictionary["profit_percent"] = self.get_profit_percent()
                dictionary["profit_USDT"] = self.get_profit_USDT()
                dictionary["profit_percent_USDT"] = self.get_profit_percent_USDT()
            except NoPriceDataException:
                logging.error("No price data!")
                dictionary["profit"] = "N/A"
                dictionary["profit_percent"] = "N/A"
                dictionary["profit_USDT"] = "N/A"
                dictionary["profit_percent_USDT"] = "N/A"
        return dictionary


if __name__ == '__main__':
    from strategies import RSITickerStrategy
    end_time = 1531699200
    start_time = end_time - 60*60*24*7
    start_cash = 10000000
    start_crypto = 0
    strategy = RSITickerStrategy(start_time, end_time, Horizon.short, None)
    evaluation = Evaluation(strategy, 'BTC', 'USDT', start_cash, start_crypto, start_time, end_time)
    #evaluation.simulate_events()
