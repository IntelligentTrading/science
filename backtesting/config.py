mysql_database_config = {
  'user': 'root',
  'password': 'itfscience',
  'host': '127.0.0.1',
  'database': 'prodclone_core_2018_06_10',
  'raise_on_warnings': True,
}

postgres_connection_string = "host='localhost' dbname='itf_07_08' user='postgres' password='itfscience'"

SUPPRESS_ALL_OUTPUT = True

def output_log(str):
    if not SUPPRESS_ALL_OUTPUT:
        print(str)
