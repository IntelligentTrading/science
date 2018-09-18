'''
Legacy MySql connection
mysql_database_config = {
  'user': 'root',
  'password': 'itfscience',
  'host': '127.0.0.1',
  'database': 'prodclone_core_2018_06_10',
  'raise_on_warnings': True,
}
'''

postgres_connection_string = "host='localhost' dbname='itf_08_05' user='postgres' password='itfscience'"

backtesting_cols_to_names = {
    "strategy": "Strategy",
    "source": "Exchange",
    "utilized_signals": "Signals",
    "transaction_currency": "Transaction currency",
    "counter_currency": "Counter currency",
    "profit_percent": "Profit percent (BTC)",
    "profit_percent_USDT": "Profit percent (USDT)",
    "buy_and_hold_profit_percent": "Buy and hold profit percent (BTC)",
    "buy_and_hold_profit_percent_USDT": "Buy and hold profit percent (USDT)",
    "num_trades": "Number of trades",
    "num_buys": "Number of buys",
    "num_sells": "Nuber of sells",
    "mean_buy_sell_pair_return": "Mean buy-sell pair return",
    "start_time": "Start time",
    "end_time": "End time",
    "resample_period": "Resample period"
}

backtesting_report_columns = list(backtesting_cols_to_names.keys())
backtesting_report_column_names = list(backtesting_cols_to_names.values())


transaction_cost_percents = {
    0: 0.0025,   # Poloniex
    1: 0.0025,   # Bittrex
    2: 0.001     # Binance
}

# top 20 altcoins on Coinmarketcap (USDT not included)
COINMARKETCAP_TOP_20_ALTS = "ETH,XRP,BCH,EOS,XLM,LTC,ADA,MIOTA,XMR,TRX,DASH,ETC,NEO,BNB,VET,XEM,XTZ,ZEC,OMG,LSK".split(",")