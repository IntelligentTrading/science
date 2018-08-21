from artemis.experiments import experiment_function
from gp_data import Data
from data_sources import Horizon
from genetic_program import GeneticProgram, FitnessFunctionV1
from leaf_functions import TAProvider
import logging
from grammar import GrammarV2, GrammarV1

@experiment_function
def run_evolution(genetic_program, mating_prob, mutation_prob, population_size, num_generations):
    hof, best = genetic_program.evolve(mating_prob, mutation_prob, population_size, num_generations)
    return hof, best


def construct_data():
    transaction_currency = "OMG"
    counter_currency = "BTC"
    end_time = 1526637600
    start_time = end_time - 60 * 60 * 24 * 30
    validation_start_time = start_time - 60 * 60 * 24 * 30
    validation_end_time = start_time
    resample_period = 60
    horizon = Horizon.short
    start_cash = 1
    start_crypto = 0
    source = 0

    training_data = Data(start_time, end_time, transaction_currency, counter_currency, resample_period, horizon,
                         start_cash, start_crypto, source)

    """
    validation_data = Data(validation_start_time, validation_end_time, transaction_currency, counter_currency,
                           resample_period, horizon, start_cash, start_crypto, source)
    """

    return training_data



def register_variants():
    data = construct_data()

    function_provider = TAProvider(data=data)

    gprog = GeneticProgram(data, function_provider=function_provider, grammar=GrammarV1(function_provider=function_provider))

    # add variants of experiments
    variant1 = run_evolution.add_variant(variant_name='evolution_v1', genetic_program=gprog, mating_prob=0.7, mutation_prob=0.5,
                                         population_size=50, num_generations=2)
    variant1.run()


    variant2 = run_evolution.add_variant(variant_name='evolution_v2', genetic_program=gprog, mating_prob=0.5, mutation_prob=0.7,
                                         population_size=50,
                                         num_generations=3)
    variant2.run()

    #return variant1, variant1, variant1


    gprog2 = GeneticProgram(data, function_provider=function_provider, grammar=GrammarV2(function_provider=function_provider))
    #gprog2 = GeneticProgram(data, fitness_function=FitnessFunctionV1())

    variant3 = \
        run_evolution.add_variant(variant_name='evolution_v3', genetic_program=gprog2, mating_prob=0.5, mutation_prob=0.7,
                                         population_size=50,
                                         num_generations=3)
    variant3.run()

    return variant1, variant2, variant3

def run_experiments():
    variant1.run(keep_record=True, display_results=True)
    variant2.run(keep_record=True, display_results=True)
    #variant3.run(keep_record=True, display_results=True)



def explore_records():
    variant_names = ['evolution_v1', 'evolution_v2', 'evolution_v3']
    # variants = run_evolution.get_all_variants(include_roots=True, include_self=True)
    for variant_name in variant_names:
        variant = run_evolution.get_variant(variant_name)
        records = variant.get_records()
        logging.info(f"===== Exploring experiment variant {variant_name} =====")
        logging.info("== Variant records:")
        logging.info(records)
        logging.info("== Last result:")
        logging.info(records[-1].get_result())
        logging.info("== Experiment output log:")
        logging.info(records[0].get_log())

####################################
#
# Artemis workflow:
#
# 1. Register all experiment variants
# 2. Run experiments (if not already done)
# 3. Browse through experiment records
# Need to run the code that registers the variants before browsing through experiment records, otherwise the browsing
# won't work!
#
####################################

if __name__ == "__main__":
    variant1, variant2, variant3 = register_variants()
    #run_experiments()
    #explore_records()


