from orders import OrderType, Order
from config import transaction_cost_percents
from data_sources import fetch_delayed_price
from abc import ABC, abstractmethod


class OrderGenerator(ABC):

    ALTERNATING = 'alternating'
    POSITION_BASED = 'position_based'

    def __init__(self, start_cash, start_crypto, time_delay, slippage):
        self._cash = start_cash
        self._crypto = start_crypto
        self._time_delay = time_delay
        self._slippage = slippage

    def _execute_order(self, order):
        delta_crypto, delta_cash = order.execute()
        self._cash = self._cash + delta_cash
        self._crypto = self._crypto + delta_crypto

    def get_orders(self, decisions):
        orders = []
        order_signals = []
        for decision in decisions:
            order = self.generate_order(decision)
            if order is not None:
                orders.append(order)
                order_signals.append(decision.signal)

        return orders, order_signals

    def _get_price(self, decision):
        return fetch_delayed_price(decision.timestamp, decision.transaction_currency,
                                    decision.counter_currency, decision.source, self._time_delay,
                                    decision.signal.price if not decision.signal is None else None)

    def buy_order(self, decision, value):
        return Order(
                order_type=OrderType.BUY,
                transaction_currency=decision.transaction_currency,
                counter_currency=decision.counter_currency,
                timestamp=decision.timestamp,
                value=value,
                unit_price=self._get_price(decision),
                transaction_cost_percent=transaction_cost_percents[decision.source],
                time_delay=self._time_delay,
                slippage=self._slippage,
                original_price=decision.signal.price if decision.signal is not None else None
            )

    def sell_order(self, decision, value):
        return Order(
            order_type=OrderType.SELL,
            transaction_currency=decision.transaction_currency,
            counter_currency=decision.counter_currency,
            timestamp=decision.timestamp,
            value=value,
            unit_price=self._get_price(decision),
            transaction_cost_percent=transaction_cost_percents[decision.source],
            time_delay=self._time_delay,
            slippage=self._slippage,
            original_price=decision.signal.price if decision.signal is not None else None
        )

    @abstractmethod
    def generate_order(self, decision):
        pass

    @staticmethod
    def create(generator_type, **kwargs):
        if generator_type == OrderGenerator.ALTERNATING:
            return AlternatingOrderGenerator(**kwargs)
        elif generator_type == OrderGenerator.POSITION_BASED:
            return PositionBasedOrderGenerator(**kwargs)
        else:
            raise Exception('Unknown order generator type!')


class AlternatingOrderGenerator(OrderGenerator):
    def __init__(self, **kwargs):
        super(AlternatingOrderGenerator, self).__init__(**kwargs)
        self._buy_currency = None

    def generate_order(self, decision):
        order = None
        try:
            if decision.sell() and self._crypto > 0 and decision.transaction_currency == self._buy_currency:
                order = self.sell_order(decision=decision, value=self._crypto)
                self._execute_order(order)
                assert self._crypto == 0

            elif decision.buy() and self._cash > 0:
                self._buy_currency = decision.transaction_currency
                order = self.buy_order(decision=decision, value=self._cash)
                self._execute_order(order)
                assert self._cash == 0
            return order
        except AttributeError:
            pass

class PositionBasedOrderGenerator(OrderGenerator):
    '''
    This order generator ensures that for each buy signal we buy 1 coin, and for each sell signal we sell 1 coin.
    Shorting is allowed, and we assume we have an unlimited supply of cash and crypto.
    '''

    def __init__(self, quantity=1, **kwargs):
        super(PositionBasedOrderGenerator, self).__init__(**kwargs)
        self._quantity = quantity
        self._transaction_currency = None

    def generate_order(self, decision):
        order = None

        if decision.buy() or decision.sell():  # sanity checks
            if self._transaction_currency is None:
                self._transaction_currency = decision.transaction_currency
            else:
                assert decision.transaction_currency == self._transaction_currency

        if decision.sell():
            order = self.sell_order(decision=decision, value=self._quantity)
            self._execute_order(order)

        elif decision.buy():
            self._buy_currency = decision.transaction_currency
            order = self.buy_order(
                decision=decision,
                value=self._get_price(decision)*self._quantity/(1-transaction_cost_percents[decision.source])
            )
            self._execute_order(order)
        return order

