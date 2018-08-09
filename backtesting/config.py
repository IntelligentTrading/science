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

transaction_cost_percents = {
    0: 0.02,    # Poloniex
                # TODO: add the rest
}