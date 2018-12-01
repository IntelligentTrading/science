### autotrading proof of concept implementation

## two tasks: 1) training and storing doges, 2) running the autotrade


## genetic_programs model

# run training
# save data (a batch of doges every X minutes)

# schedule this periodically

from gp_artemis import ExperimentManager
from data_sources import redis_db
from genetic_program import GeneticTickerStrategy
from tick_provider import PriceDataframeTickProvider
from tick_listener import TickListener
from utils import datetime_from_timestamp

def retrain_doges(max_doges_to_save=10):
    # create an experiment manager
    e = ExperimentManager('gv5_experiments.json', database=redis_db) # we will have one central json with all the parameters

    # run experiments
    e.run_experiments() # if we can run it in parallel, otherwise call e.run_experiments()

    # retrieve best performing doges
    doge_df = e.get_best_performing_across_variants_and_datasets(datasets=e.training_data, sort_by=['mean_profit'], top_n_per_variant=1)

    dummy_db = open('dogi.db', 'w')
    # TODO: write these guys to the actual database
    for i, row in enumerate(doge_df.itertuples()):
        if i > max_doges_to_save:
            break
        # save experiment id and doge
        dummy_db.write(f'{row.variant.name}\t{i}\t{str(row.doge)}\t{row.mean_profit}\n')
    dummy_db.close()


class DogeTrader(TickListener):

    def __init__(self, database):
        # load current batch of doges from the db
        # TODO: fill this from actual db

        e = ExperimentManager('gv5_experiments.json', database=database)
        doge_strategies = []
        transaction_currency = 'BTC'
        counter_currency = 'USDT'
        resample_period = 60
        source = 0

        with open('gv5_experiments.json', 'r') as json_file:
            experiment_json = json_file.read()

        with open('dogi.db') as dummy_db:
            for line in dummy_db:
                experiment_id, i, individual_str, mean_profit = line.split('\t')
                doge, gp = ExperimentManager.resurrect_doge(experiment_json, experiment_id, individual_str, database)
                strategy = GeneticTickerStrategy(doge, transaction_currency, counter_currency, resample_period,
                                                 source, gp)
                doge_strategies.append(strategy)

        self.doge_strategies = doge_strategies
        tick_provider = PriceDataframeTickProvider(e.training_data[0].price_data)
        tick_provider.add_listener(self)
        tick_provider.run()

        # simulate the decision process over all the strategies

    def process_event(self, price_data, signal_data):
        print(f'So wow! Price arrived ({datetime_from_timestamp(price_data.Index)})')
        for i, doge in enumerate(self.doge_strategies):
            print(f'  Doge {i} says: {str(doge.process_ticker(price_data, signal_data).outcome)}')


    def broadcast_ended(self):
        print('Doges have spoken.')



# autotrading:
# get the most recent doge
#


if __name__ == '__main__':
    retrain_doges()
    trader = DogeTrader(database=redis_db)