from pyalgotrade import strategy
from pyalgotrade.technical import ma
from pyalgotrade.technical import cross
import pyalgotrade

# Needed to add this to enable fractional shares!!
# Doesn't seem to be directly supported in the library
class DecimalTraits(pyalgotrade.broker.InstrumentTraits):
    def __init__(self, decimals):
        self.__decimals = decimals

    def roundQuantity(self, quantity):
        return round(quantity, self.__decimals)

class DecimalBroker(pyalgotrade.broker.backtesting.Broker):
    def getInstrumentTraits(self, instrument):
        return DecimalTraits(10)

class SMACrossOver(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, smaPeriod):
        super(SMACrossOver, self).__init__(feed, DecimalBroker(1000000, feed))
        self.__instrument = instrument
        self.__position = None
        self.setUseAdjustedValues(True)
        self.__prices = feed[instrument].getPriceDataSeries()
        self.__sma = ma.SMA(self.__prices, smaPeriod)
        # self.prev_shares = 0  # debug stuff

    def getSMA(self):
        return self.__sma

    def onBars(self, bars):
        # NOTE: something weird going on with the cash variable here; why leftover cash?

        shares = self.getBroker().getShares(self.__instrument)
        bar = bars[self.__instrument]

        # self.info(bar.getOpen() * self.prev_shares)
        # self.prev_shares = shares

        if shares == 0 and cross.cross_above(self.__prices, self.__sma) > 0:
            sharesToBuy = self.getBroker().getCash(False) / bar.getClose()
            self.info("{} Bought {} {}, unit price = ${:.2f}, using cash = {} (cash / price = {})".format(
                bars[self.__instrument].getDateTime(),
                sharesToBuy,
                self.__instrument,
                bars[self.__instrument].getClose(),
                self.getBroker().getCash(),
                self.getBroker().getCash() / bar.getClose()))
            self.marketOrder(self.__instrument, sharesToBuy)

        elif shares > 0 and cross.cross_below(self.__prices, self.__sma) > 0:
            self.marketOrder(self.__instrument, -1*shares)
            self.info("{} Gave an order to exit position (1 {} = ${:.2f}), leftover cash?? = {:.2f}".format(
                bars[self.__instrument].getDateTime(),
                self.__instrument,
                bar.getClose(),
                self.getBroker().getCash()))
            self.info("NOTE: the exit will happen at the opening price of the next tick")

