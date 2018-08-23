from artemis.experiments import experiment_root
from gp_data import Data
from genetic_program import GeneticProgram, FitnessFunction
from leaf_functions import TAProvider
import logging
from grammar import Grammar
import json
import itertools
from chart_plotter import *

@experiment_root
def run_evolution(experiment_id, data, function_provider, grammar_version, fitness_function, mating_prob,
                  mutation_prob, population_size, num_generations):
    grammar = Grammar.construct(grammar_version, function_provider, ephemeral_suffix=experiment_id)
    genetic_program = GeneticProgram(data=data, function_provider=function_provider,
                            grammar=grammar, fitness_function=fitness_function)
    hof, best = genetic_program.evolve(mating_prob, mutation_prob, population_size, num_generations)
    return hof, best


class ExperimentManager:

    START_CASH = 1000
    START_CRYPTO = 0

    def __init__(self, experiment_file_path):
        with open(experiment_file_path) as f:
            experiment_info = json.load(f)
        self.experiment_json = experiment_info
        # initialize data
        self.training_data = Data(start_cash=self.START_CASH, start_crypto=self.START_CRYPTO,
                                  **self.experiment_json["training_data"])
        self.validation_data = Data(start_cash=self.START_CASH, start_crypto=self.START_CRYPTO,
                                  **self.experiment_json["validation_data"])
        # create function provider based on data
        self.function_provider = TAProvider(self.training_data)
        # initialize fitness function
        self.fitness_function = FitnessFunction.construct(experiment_info["fitness_function"])
        self._fill_experiment_db()
        self._register_variants()

    def _fill_experiment_db(self):
        self.experiment_db = ExperimentDB()

        variants = itertools.product(self.experiment_json["mating_probabilities"],
                                     self.experiment_json["mutation_probabilities"],
                                     self.experiment_json["population_sizes"])
        for mating_prob, mutation_prob, population_size in variants:
            self.experiment_db.add(
                data=self.training_data,
                function_provider=self.function_provider,
                grammar_version=self.experiment_json["grammar_version"],
                fitness_function=self.fitness_function,
                mating_prob=mating_prob,
                mutation_prob=mutation_prob,
                population_size=population_size,
                num_generations=self.experiment_json["num_generations"])

    def _register_variants(self, rebuild_grammar=True):
        if rebuild_grammar:
            for experiment_name, experiment in self.experiment_db.get_all():
                Grammar.construct(experiment['grammar_version'],
                                  experiment['function_provider'],
                                  experiment['experiment_id'])

        for experiment_name, experiment in self.experiment_db.get_all():
            run_evolution.add_variant(variant_name=experiment_name, **experiment)
        self.variants = run_evolution.get_variants()

    def run_experiments(self, rerun_existing=False):
        for i, variant in enumerate(self.variants):
            if len(variant.get_records(only_completed=True)) > 0 and not rerun_existing:
                logging.info(f">>> Variant {variant.name} already has completed records, skipping...")
                continue

            logging.info(f"Running variant {i}")
            variant.run(keep_record=True, display_results=True, saved_figure_ext='.fig.png')

    def explore_records(self, use_validation_data=True):
        if use_validation_data:
            data = self.validation_data
        else:
            data = self.training_data
        #   variants = run_evolution.get_variants()
        for variant in self.variants:
            # for variant_name in variant_names:
            #   variant = run_evolution.get_variant(variant_name)
            records = variant.get_records()
            latest = variant.get_latest_record(only_completed=True)
            if latest is None:
                logging.warning(f">>> No records found for variant {variant.name}, skipping...")
                continue
            logging.info(f"\n\n===== Exploring experiment variant {variant} =====")
            logging.info("== Variant records:")
            logging.info(records)
            logging.info("== Last result:")
            logging.info(latest)
            hof, best = latest.get_result()
            for individual in hof:
                print(individual)
            best_individual = hof[0]
            evaluation = self._build_evaluation_object(best_individual, variant, data)
            draw_price_chart(data.timestamps, data.prices, evaluation.orders)
            draw_tree(best_individual)

            logging.info("== Experiment output log:")
            logging.info(latest.get_log())

    def analyze_and_find_best(self, data=None):
        if data is None:
            data = self.validation_data

        # we'll go through all variants and record performance
        for variant in self.variants:
            records = variant.get_records()
            latest = variant.get_latest_record(only_completed=True)
            if latest is None:
                logging.warning(f">>> No records found for variant {variant.name}, skipping...")
                continue
            hof, best = latest.get_result()

            for individual in hof:
                evaluation = self._build_evaluation_object(individual, variant, data)
                draw_price_chart(data.timestamps, data.prices, evaluation.orders)
                draw_tree(individual)



    def browse_variants(self):
        run_evolution.browse()

    def _build_evaluation_object(self, individual, variant, data):
        db_record = self.experiment_db[variant.name[len("run_evolution."):]]
        grammar = Grammar.construct(
            grammar_name=db_record['grammar_version'],
            function_provider=self.function_provider,
            ephemeral_suffix=db_record['experiment_id']
        )
        genetic_program = GeneticProgram(
            data=data,
            function_provider=self.function_provider,
            grammar=grammar,
            fitness_function=self.fitness_function
        )
        return genetic_program.build_evaluation_object(individual)


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

    def build_experiment_id(self, **kwargs):
        return f"data_{kwargs['data']};" \
               f"provider_{kwargs['function_provider']};" \
               f"grammar_{kwargs['grammar_version']};" \
               f"fitness_{kwargs['fitness_function']};" \
               f"matingprob_{kwargs['mating_prob']};" \
               f"mutationprob_{kwargs['mutation_prob']};" \
               f"populationsize_{kwargs['population_size']};" \
               f"generations_{kwargs['num_generations']}"

    def __getitem__(self, key):
        return self._experiments[key]

    # def build_experiment_id(**kwargs):
    #    return ';'.join(['{}_{}'.format(k, v) for k, v in kwargs.iteritems()])


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
    e = ExperimentManager("sample_experiment.json")
    #e.run_experiments()
    #e.explore_records()
    e.analyze_and_find_best()
    #e.browse_variants()


