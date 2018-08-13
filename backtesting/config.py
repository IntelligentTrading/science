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

backtesting_report_columns = [
    "strategy",
    "utilized_signals",
    "transaction_currency",
    "counter_currency",
    "profit_percent",
    "profit_percent_USDT",
    "buy_and_hold_profit_percent",
    "buy_and_hold_profit_percent_USDT",
    "num_trades",
    "num_buys",
    "num_sells",
    "mean_buy_sell_pair_return",
    "start_time",
    "end_time",
    "evaluate_profit_on_last_order",
    "resample_period"
]

transaction_cost_percents = {
    0: 0.002,   # Poloniex
                # TODO: add the rest
}