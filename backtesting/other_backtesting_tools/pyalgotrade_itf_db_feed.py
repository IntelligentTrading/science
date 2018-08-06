import pandas as pd

from pyalgotrade.feed import csvfeed
from backtesting.data_sources import dbc


class ITFDatabaseFeed(csvfeed.GenericBarFeed):

    def __init__(self, frequency, timezone=None, maxLen=None):
        super.__init__(frequency, timezone, maxLen)

    def read_data(self, instrument):
        connection = dbc.get_connection()
        query = "SELECT timestamp, " \
                "to_timestamp(TRUNC(CAST(timestamp AS bigint))) AT TIME ZONE 'UTC' as date, " \
                "open_price/1E8 as open_price, " \
                "high_price/1E8 as high_price, " \
                "low_price/1E8 as low_price, " \
                "close_price/1E8 as close_price, " \
                "close_volume, " \
                "close_price/1E8 as adj_close_price " \
                "FROM indicator_priceresampl " \
                "WHERE transaction_currency = '%s' AND" \
                "counter_currency = 2 AND " \
                "source = 0 AND " \
                "resample_period = 60 AND " \
                "close_volume IS NOT NULL" \
                "ORDER BY timestamp ASC"
        price_data = pd.read_sql(query, con=connection, params=(instrument),
                                 index_col="timestamp")

    def addBarsFromCSV(self, instrument, timezone=None):
        """Loads bars for a given instrument from a CSV formatted file.
        The instrument gets registered in the bar feed.

        :param instrument: Instrument identifier.
        :type instrument: string.
        :param path: The path to the CSV file.
        :type path: string.
        :param timezone: The timezone to use to localize bars. Check :mod:`pyalgotrade.marketsession`.
        :type timezone: A pytz timezone.
        """

        if timezone is None:
            timezone = self.__timezone

        rowParser = csvfeed.GenericRowParser(
            self.__columnNames, self.__dateTimeFormat, self.getDailyBarTime(), self.getFrequency(),
            timezone, self.__barClass
        )

        price_data = self.read_data(instrument)
        loadedBars = []
        for timestamp, row in price_data.iterrows():
            row_dict = {}

            row_dict['datetime'] = self._parseDate(row['date'])
            row_dict['open'] = float(row['open_price'])
            row_dict['high'] = float(row['high_price'])
            row_dict['low'] = float(row['low_price'])
            row_dict['close'] = float(row['close'])
            row_dict['volume'] = float(row['close_volume'])
            row_dict['adj_close'] = float(row['close'])

            bar_ = rowParser.parseBar(row_dict)
            if bar_ is not None and (self.__barFilter is None or self.__barFilter.includeBar(bar_)):
                loadedBars.append(bar_)

        self.addBarsFromSequence(instrument, loadedBars)

