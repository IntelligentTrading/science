database_config = {
  'user': 'root',
  'password': 'karla',
  'host': '127.0.0.1',
  'database': 'prodclone_core_2018_05_26',
  'raise_on_warnings': True,
}

SUPPRESS_ALL_OUTPUT = True

def output_log(str):
    if not SUPPRESS_ALL_OUTPUT:
        print(str)
