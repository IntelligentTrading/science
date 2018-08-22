from artemis.experiments import experiment_function
from gp_data import Data
from data_sources import Horizon
from genetic_program import GeneticProgram, FitnessFunctionV1, FitnessFunctionV2
from leaf_functions import TAProvider
import logging
from grammar import GrammarV2, GrammarV1, Grammar


@experiment_function
def run_evolution(experiment_id, data, function_provider, grammar_version, fitness_function, mating_prob,
                  mutation_prob, population_size, num_generations):
    grammar = Grammar.construct(grammar_version, function_provider, ephemeral_suffix=experiment_id)
    genetic_program = GeneticProgram(data=data, function_provider=function_provider,
                            grammar=grammar, fitness_function=fitness_function)
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


class ExperimentDB:

    def __init__(self):
        self._experiments = {}
        self._num_records = 0

    def add(self, data, function_provider, grammar_version, fitness_function,
            mating_prob, mutation_prob, population_size, num_generations):
        entry = {}
        entry['data'] = data
        entry['function_provider'] = function_provider
        entry['grammar_version'] = grammar_version
        entry['fitness_function'] = fitness_function
        entry['mating_prob'] = mating_prob
        entry['mutation_prob'] = mutation_prob
        entry['population_size'] = population_size
        entry['num_generations'] = num_generations
        entry['experiment_id'] = self._num_records
        name = self.build_experiment_id(**entry)
        self._experiments[name] = entry
        self._num_records += 1

    def get_all(self):
        for k, v in self._experiments.items():
            yield k, v

    def build_experiment_id(**kwargs):
        return f"data_{kwargs['data']};" \
               f"provider_{kwargs['function_provider']};" \
               f"grammar_{kwargs['grammar_version']};" \
               f"fitness_{kwargs['fitness_function']};" \
               f"matingprob_{kwargs['mating_prob']};" \
               f"mutationprob_{kwargs['mutation_prob']};" \
               f"populationsize_{kwargs['population_size']};" \
               f"generations_{kwargs['num_generations']}"

    # def build_experiment_id(**kwargs):
    #    return ';'.join(['{}_{}'.format(k, v) for k, v in kwargs.iteritems()])


def register_experiments():
    experiments = ExperimentDB()
    data = construct_data()
    function_provider = TAProvider(data=data)

    experiments.add(
        data=data,
        function_provider=function_provider,
        grammar_version="gv1",
        fitness_function=FitnessFunctionV1(),
        mating_prob=0.7,
        mutation_prob=0.5,
        population_size=50,
        num_generations=2)


    experiments.add(  # variant_name='evolution_v2',
        data=data,
        function_provider=function_provider,
        grammar_version="gv2",
        fitness_function=FitnessFunctionV1(),
        mating_prob=0.7,
        mutation_prob=0.5,
        population_size=50,
        num_generations=2)

    experiments.add(  # variant_name='evolution_v3',
        data=data,
        function_provider=function_provider,
        grammar_version="gv1",
        fitness_function=FitnessFunctionV2(),
        mating_prob=0.7,
        mutation_prob=0.5,
        population_size=50,
        num_generations=2)


    return experiments

def register_variants(rebuild_grammar=False):
    experiments = register_experiments()
    if rebuild_grammar:
        for experiment_name, experiment in experiments.get_all():
            Grammar.construct(experiment['grammar_version'], experiment['function_provider'], experiment['experiment_id'])

    for experiment_name, experiment in experiments.get_all():
        run_evolution.add_variant(variant_name=experiment_name, **experiment)
    return run_evolution.get_variants()


def run_experiments(variants):
    for i, variant in enumerate(variants):
        print(f"Running variant {i}")
        variant.run(keep_record=True, display_results=True)


def explore_records(variants):
#   variants = run_evolution.get_variants()
    for variant in variants:
    #for variant_name in variant_names:
    #   variant = run_evolution.get_variant(variant_name)
        records = variant.get_records()
        logging.info(f"\n\n===== Exploring experiment variant {variant} =====")
        logging.info("== Variant records:")
        logging.info(records)
        logging.info("== Last result:")
        logging.info(records[-1].get_result())
        hof, best = records[-1].get_result()
        for individual in hof:
            print(individual)

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
    variants = register_variants(rebuild_grammar=True)
    #run_experiments(variants)
    explore_records(variants)


