from artemis.experiments import experiment_root
from gp_data import Data
from genetic_program import GeneticProgram, FitnessFunction
from leaf_functions import TAProviderCollection
import logging
from grammar import Grammar
import json
import itertools
from chart_plotter import *

@experiment_root
def run_evolution(experiment_id, data, function_provider, grammar_version, fitness_function, mating_prob,
                  mutation_prob, population_size, num_generations):
    grammar = Grammar.construct(grammar_version, function_provider, ephemeral_suffix=experiment_id)
    genetic_program = GeneticProgram(data_collection=data, function_provider=function_provider,
                                     grammar=grammar, fitness_function=fitness_function)
    hof, best = genetic_program.evolve(mating_prob, mutation_prob, population_size, num_generations, verbose=False)
    return hof, best


class ExperimentManager:

    START_CASH = 1000
    START_CRYPTO = 0

    def __init__(self, experiment_container, read_from_file=True):
        if read_from_file:
            with open(experiment_container) as f:
                self.experiment_json = json.load(f)
        else:
            self.experiment_json = experiment_container
        # initialize data
        self.training_data = [Data(start_cash=self.START_CASH, start_crypto=self.START_CRYPTO,
                                  **dataset) for dataset in self.experiment_json["training_data"]]
        self.validation_data = [Data(start_cash=self.START_CASH, start_crypto=self.START_CRYPTO,
                                  **dataset) for dataset in self.experiment_json["validation_data"]]
        # create function provider objects based on data
        self.function_provider = TAProviderCollection(self.training_data + self.validation_data)
        # initialize fitness function
        self.fitness_function = FitnessFunction.construct(self.experiment_json["fitness_function"])
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

    def get_variants(self):
        return run_evolution.get_variants()

    def run_experiments(self, rerun_existing=False, display_results=True):
        for i, variant in enumerate(self.variants):
            if len(variant.get_records(only_completed=True)) > 0 and not rerun_existing:
                logging.info(f">>> Variant {variant.name} already has completed records, skipping...")
                continue

            logging.info(f"Running variant {i}")
            variant.run(keep_record=True, display_results=display_results, saved_figure_ext='.fig.png')

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
            evaluation.plot_price_and_orders()
            draw_tree(best_individual)

            logging.info("== Experiment output log:")
            logging.info(latest.get_log())



    def print_best_individual_statistics(self, variant):
        best_individual, evaluations = self.get_best_performing_across_datasets(variant)
        print(f"Best performing individual across datasets in variant {variant}:\n")
        for evaluation in evaluations:
            self._print_individual_info(best_individual, evaluation)


    def get_best_performing_across_datasets(self, variant):
        """
        Finds the best performing individual in an experiment variant, across all training datasets.
        :param variant: experiment variant
        :return: individual (DEAP object) and its corresponding evaluations across datasets
                 (a list of Evaluation objects)
        """
        latest = variant.get_latest_record(only_completed=True)
        if latest is None:
            logging.warning(f">>> No records found for variant {variant.name}, skipping...")
            return
        hof, best = latest.get_result()
        best_individual = hof[0]
        evaluations = []
        for data in self.training_data:
            evaluation = self._build_evaluation_object(best_individual, variant, data)
            evaluations.append(evaluation)
        return best_individual, evaluations

    def get_best_performing_across_variants_and_datasets(self):
        """
        Returns a list of best performing individuals, one per experiment variant.
        :return:
        """
        best_individuals = {}
        for variant in self.variants:
            best_individual, evaluation = self.get_best_performing_across_datasets(variant)
            best_individuals[variant] = (best_individual, evaluation)
        return best_individuals

    def get_performance_df_for_dataset_and_variant(self, variant, data):
        """
        Produces a performance dataframe for the given training set and experiment variant.
        :param variant: experiment variant (get a list of all by invoking get_variants() on this object)
        :param data: a Data object, part of the training_data collection
        :return: a dataframe showing performance
        """
        performance_rows = []
        latest = variant.get_latest_record(only_completed=True)
        if latest is None:
            logging.warning(f">>> No records found for variant {variant.name}, skipping...")
            return
        hof, best = latest.get_result()

        for rank, individual in enumerate(hof):
            evaluation = self._build_evaluation_object(individual, variant, data)
            row = evaluation.to_primitive_types_dictionary()
            row['experiment_id'] = variant.name
            row['hof_ranking'] = rank
            row["individual"] = individual
            row["evaluation"] = evaluation
            performance_rows.append(row)

        performance_info = pd.DataFrame(performance_rows)
        performance_info = performance_info.sort_values(by=['profit_percent'], ascending=False)
        return performance_info

    def get_performance_dfs_for_variant(self, variant):
        """
        Gets performance dataframes for an experiment variant, one per training subset.
        :param variant: Experiment variant
        :return: a list of performance dataframes
        """
        performance_dfs = []
        for data in self.training_data:
            performance_dfs.append(self.get_performance_df_for_dataset_and_variant(variant, data))
        return performance_dfs

    def get_joined_performance_dfs_over_all_variants(self, verbose=True):
        """
        Produces a list of N dataframes, where N is the number of training sets in the training collection.
        Each dataframe shows profits on the the corresponding training set across all experiment variants.
        :return: a list of dataframes
        """
        joined_dfs = []

        for data in self.training_data:
            performance_info = pd.DataFrame()
            for variant in self.variants:
                df = self.get_performance_df_for_dataset_and_variant(variant, data)
                performance_info = performance_info.append(df)

            performance_info = performance_info.sort_values(by=['profit_percent'], ascending=False)
            if verbose:
                self.performance_df_row_info(performance_info.iloc[0])
            joined_dfs.append(performance_info)
        return joined_dfs

    def performance_df_row_info(self, performance_df_row):
        print(f'Experiment id: {performance_df_row.experiment_id}\n')
        individual = performance_df_row.individual
        evaluation = performance_df_row.evaluation
        self._print_individual_info(individual, evaluation)
        return individual

    def _print_individual_info(self, individual, evaluation):
        evaluation.plot_price_and_orders()
        print(f'String representation:\n{str(individual)}\n')
        draw_tree(individual)
        try:
            networkx_graph(individual)
        except:
            logging.warning("Failed to plot networkx graph (not installed?), skipping...")
        print(f'Backtesting report:\n {evaluation.get_report()}\n')
        print(f'Benchmark backtesting report:\n {evaluation.benchmark_backtest.get_report()}\n')

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
            data_collection=[data],
            function_provider=self.function_provider,
            grammar=grammar,
            fitness_function=self.fitness_function
        )
        return genetic_program.build_evaluation_object(individual, data)


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
        return f"data_{'-'.join(map(str,kwargs['data']))};" \
               f"grammar_{kwargs['grammar_version']};" \
               f"fitness_{kwargs['fitness_function']};" \
               f"x_{kwargs['mating_prob']};" \
               f"m_{kwargs['mutation_prob']};" \
               f"n_{kwargs['population_size']};" \
               f"gen_{kwargs['num_generations']}"
    #          f"provider_{kwargs['function_provider']};" \

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
    dfs = e.get_joined_performance_dfs_over_all_variants()
    df = e.analyze_and_find_best_in_dataset()
    from chart_plotter import rewrite_graph_as_tree
    rewrite_graph_as_tree(df.iloc[3].individual, "")
    #e.browse_variants()


