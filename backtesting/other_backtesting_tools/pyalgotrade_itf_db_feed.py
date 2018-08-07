import pandas as pd

from pyalgotrade.barfeed import csvfeed
from backtesting.data_sources import dbc


class ITFDatabaseFeed(csvfeed.GenericBarFeed):

    def __init__(self, frequency, timezone=None, maxLen=None):
        super(ITFDatabaseFeed, self).__init__(frequency, timezone, maxLen)

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
                "WHERE transaction_currency = %s AND " \
                "counter_currency = 2 AND " \
                "source = 0 AND " \
                "resample_period = 60 AND " \
                "close_volume IS NOT NULL " \
                "ORDER BY timestamp ASC"
        price_data = pd.read_sql(query, con=connection, params=(instrument,),
                                 index_col="timestamp")
        return price_data

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
            timezone = self._GenericBarFeed__timezone

        rowParser = csvfeed.GenericRowParser(
            self._GenericBarFeed__columnNames, self._GenericBarFeed__dateTimeFormat, self.getDailyBarTime(), self.getFrequency(),
            timezone, self._GenericBarFeed__barClass
        )

        price_data = self.read_data(instrument)
        loadedBars = []
        for timestamp, row in price_data.iterrows():
            row_dict = {}

            row_dict['Date Time'] = str(row['date'])
            row_dict['Open'] = float(row['open_price'])
            row_dict['High'] = float(row['high_price'])
            row_dict['Low'] = float(row['low_price'])
            row_dict['Close'] = float(row['close_price'])
            row_dict['Volume'] = float(row['close_volume'])
            row_dict['Adj_Close'] = float(row['close_price'])


            bar_ = rowParser.parseBar(row_dict)
            if bar_ is not None and (self._BarFeed__barFilter is None or self._BarFeed__barFilter.includeBar(bar_)):
                loadedBars.append(bar_)

        self.addBarsFromSequence(instrument, loadedBars)

