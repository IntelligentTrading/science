from orders import OrderType, Order
from config import transaction_cost_percents
from data_sources import fetch_delayed_price


class AlternatingOrderGenerator:

    @staticmethod
    def get_orders(strategy, signals, start_cash, start_crypto, source, time_delay=0, slippage=0):
        """
        Produces a list of buy-sell orders based on input signals.
        :param strategy: The strategy for which to produce the orders.
        :param signals: A list of input signals.
        :param start_cash: Starting amount of counter_currency (counter_currency is read from first signal).
        :param start_crypto: Starting amount of transaction_currency (transaction_currency is read from first signal).
        :param source: ITF exchange code.
        :param time_delay: Parameter specifying the delay applied when fetching price info (in seconds).
        :param slippage: Parameter specifying the slippage percentage, applied in the direction of the trade.
        :return: A list of orders produced by the strategy.
        """
        orders = []
        order_signals = []
        cash = start_cash
        crypto = start_crypto
        buy_currency = None
        for i, signal in enumerate(signals):
            if not strategy.belongs_to_this_strategy(signal):
                continue

            if strategy.indicates_sell(signal) and crypto > 0 and signal.transaction_currency == buy_currency:
                price = fetch_delayed_price(signal, source, time_delay)
                order = Order(OrderType.SELL, signal.transaction_currency, signal.counter_currency,
                              signal.timestamp, crypto, price, transaction_cost_percents[source], time_delay, slippage,
                              signal.price)
                orders.append(order)
                order_signals.append(signal)
                delta_crypto, delta_cash = order.execute()
                cash = cash + delta_cash
                crypto = crypto + delta_crypto
                assert crypto == 0

            elif strategy.indicates_buy(signal) and cash > 0:
                price = fetch_delayed_price(signal, source, time_delay)
                buy_currency = signal.transaction_currency
                order = Order(OrderType.BUY, signal.transaction_currency, signal.counter_currency,
                              signal.timestamp, cash, price, transaction_cost_percents[source], time_delay, slippage,
                              signal.price)
                orders.append(order)
                order_signals.append(signal)
                delta_crypto, delta_cash = order.execute()
                cash = cash + delta_cash
                crypto = crypto + delta_crypto
                assert cash == 0

        return orders, order_signals


class PositionBasedOrderGenerator:
    '''
    This order generator ensures that for each buy signal we buy 1 coin, and for each sell signal we sell 1 coin.
    Shorting is allowed, and we assume we have an unlimited supply of cash and crypto.
    '''

    def __init__(self, quantity=1):
        self._quantity = quantity

    def get_orders(self, strategy, signals, start_cash, start_crypto, source, time_delay=0, slippage=0):
        """
        Produces a list of buy-sell orders based on input signals.
        :param strategy: The strategy for which to produce the orders.
        :param signals: A list of input signals.
        :param start_cash: Starting amount of counter_currency (counter_currency is read from first signal).
        :param start_crypto: Starting amount of transaction_currency (transaction_currency is read from first signal).
        :param source: ITF exchange code.
        :param time_delay: Parameter specifying the delay applied when fetching price info (in seconds).
        :param slippage: Parameter specifying the slippage percentage, applied in the direction of the trade.
        :return: A list of orders produced by the strategy.
        """
        orders = []
        order_signals = []

        cash = start_cash
        crypto = start_crypto

        transaction_currency = None
        for i, signal in enumerate(signals):
            if not strategy.belongs_to_this_strategy(signal):
                continue

            if transaction_currency is None:
                transaction_currency = signal.transaction_currency  # we will trade with this currency

            if strategy.indicates_sell(signal) and signal.transaction_currency == transaction_currency:
                price = fetch_delayed_price(signal, source, time_delay)
                order = Order(OrderType.SELL, signal.transaction_currency, signal.counter_currency,
                              signal.timestamp, self._quantity, price, transaction_cost_percents[source], time_delay, slippage,
                              signal.price)
                orders.append(order)
                order_signals.append(signal)
                delta_crypto, delta_cash = order.execute()
                cash = cash + delta_cash
                crypto = crypto + delta_crypto

            elif strategy.indicates_buy(signal) and signal.transaction_currency == transaction_currency:
                price = fetch_delayed_price(signal, source, time_delay)
                order = Order(OrderType.BUY, signal.transaction_currency, signal.counter_currency,
                              signal.timestamp, price, price, transaction_cost_percents[source], time_delay, slippage,
                              signal.price)
                orders.append(order)
                order_signals.append(signal)
                delta_crypto, delta_cash = order.execute()
                cash = cash + delta_cash
                crypto = crypto + delta_crypto

        return orders, order_signals


