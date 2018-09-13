from orders import OrderType, Order
#from strategies import StrategyDecision
from config import transaction_cost_percents

class TradingSimulator:

    def __init__(self, strategy, signals, start_cash, start_crypto, source, time_delay=0, slippage=0, endless_budget=False):
        self._strategy = strategy
        self._signals = signals
        self._start_cash = start_cash
        self._start_crypto = start_crypto
        self._source = source
        self._time_delay = time_delay
        self._slippage = slippage
        self._endless_budget = endless_budget
        self._orders = []
        self._order_signals = []
        self._crypto = start_crypto
        self._cash = start_cash
        self._transaction_cost_percent = transaction_cost_percents[source]


class AlternatingBuySellTrading(TradingSimulator):

    def __init__(self, **kwargs):
        super(AlternatingBuySellTrading, self).__init__(**kwargs)

        # if we received signals, process all of them and generate orders
        # otherwise we depend on external source to call process_strategy_decision
        if 'signals' in kwargs:
            for signal in self._signals:
                if self._strategy.indicates_sell(signal):
                    self.process_strategy_decision('SELL', signal.transaction_currency,
                                                   signal.counter_currency, signal.timestamp, signal.price, signal)
                elif self._strategy.indicates_buy(signal):
                    self.process_strategy_decision('BUY', signal.transaction_currency,
                                                   signal.counter_currency, signal.timestamp, signal.price, signal)

    def process_strategy_decision(self, decision, transaction_currency, counter_currency, timestamp, price, signal=None):
        order = None
        if decision == 'SELL' and self._crypto > 0:
            order = Order(OrderType.SELL, transaction_currency, counter_currency,
                          timestamp, self._crypto, price, self._transaction_cost_percent, self._time_delay,
                          self._slippage)
            self._orders.append(order)
            self._order_signals.append(signal)
        elif decision == 'BUY' and self._cash > 0:
            order = Order(OrderType.BUY, transaction_currency, counter_currency,
                          timestamp, self._cash, price, self._transaction_cost_percent, self._time_delay,
                          self._slippage)
            self._orders.append(order)
            self._order_signals.append(signal)
        if order is not None:
            delta_crypto, delta_cash = order.execute()
            self._cash += delta_cash
            self._crypto += delta_crypto

    @property
    def orders(self):
        return self._orders

    @property
    def order_signals(self):
        return self._order_signals





