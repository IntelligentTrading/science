from abc import ABC


class TickProvider(ABC):

    def __init__(self):
        self._listeners = []

    def add_listener(self, listener):
        self._listeners.append(listener)

    def notify_listeners(self, price_data, signal_data):
        for listener in self._listeners:
            listener.process_event(price_data, signal_data)

    def broadcast_ended(self):
        for listener in self._listeners:
            listener.broadcast_ended()



