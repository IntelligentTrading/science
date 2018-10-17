import logging
import json
import itertools
import numpy as np
import pandas as pd

from artemis.experiments import experiment_root
from gp_data import Data
from genetic_program import GeneticProgram, FitnessFunction
from leaf_functions import TAProviderCollection
from grammar import Grammar
from chart_plotter import *
from order_generator import OrderGenerator
from config import INF_CASH, INF_CRYPTO
from comparative_evaluation import ComparativeEvaluation, ComparativeReportBuilder
from data_sources import get_currencies_trading_against_counter
from backtesting_runs import build_itf_baseline_strategies
from data_sources import NoPriceDataException
from utils import time_performance
from functools import partial

SAVE_HOF_AND_BEST = True
HOF_AND_BEST_FILENAME = 'rockstars.txt'
LOAD_ROCKSTARS = True





@experiment_root
def run_evolution(experiment_id, data, function_provider, grammar_version, fitness_function, mating_prob,
                  mutation_prob, population_size, num_generations, premade_individuals, order_generator, tree_depth,
                  reseed_params):
    grammar = Grammar.construct(grammar_version, function_provider, ephemeral_suffix=experiment_id)
    genetic_program = GeneticProgram(data_collection=data, function_provider=function_provider,
                                     grammar=grammar, fitness_function=fitness_function,
                                     premade_individuals=premade_individuals, order_generator=order_generator,
                                     tree_depth=tree_depth, reseed_params=reseed_params)
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

        if self.experiment_json['order_generator'] == OrderGenerator.POSITION_BASED:
            self.START_CASH = INF_CASH
            self.START_CRYPTO = INF_CRYPTO

        # initialize data
        self.training_data = [Data(start_cash=self.START_CASH, start_crypto=self.START_CRYPTO,
                                  **dataset) for dataset in self.experiment_json["training_data"]]
        self.validation_data = [Data(start_cash=self.START_CASH, start_crypto=self.START_CRYPTO,
                                  **dataset) for dataset in self.experiment_json["validation_data"]]
        # create function provider objects based on data
        self.function_provider = TAProviderCollection(self.training_data + self.validation_data)
        # generate and register variants
        self._fill_experiment_db()
        self._register_variants()

    def _fill_experiment_db(self):
        self.experiment_db = ExperimentDB()

        variants = itertools.product(self.experiment_json["mating_probabilities"],
                                     self.experiment_json["mutation_probabilities"],
                                     self.experiment_json["population_sizes"],
                                     self.experiment_json["fitness_functions"])
        for mating_prob, mutation_prob, population_size, fitness_function in variants:
            if LOAD_ROCKSTARS:
                grammar_version = self.experiment_json["grammar_version"]
                rockstars = self._load_rockstars(grammar_version, fitness_function, to_load=['hof'], limit_top=20)
                logging.info(f'Loaded {len(rockstars)} rockstars for {grammar_version}, {fitness_function}.')
            else:
                rockstars = []

            self.experiment_db.add(
                data=self.training_data,
                function_provider=self.function_provider,
                grammar_version=self.experiment_json["grammar_version"],
                fitness_function=FitnessFunction.construct(fitness_function),
                mating_prob=mating_prob,
                mutation_prob=mutation_prob,
                population_size=population_size,
                num_generations=self.experiment_json["num_generations"],
                premade_individuals=self.experiment_json["premade_individuals"]+rockstars,
                order_generator=self.experiment_json["order_generator"],
                tree_depth=self.experiment_json["tree_depth"],
                reseed_params=self.experiment_json["reseed_initial_population"]
            )


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

    @staticmethod
    def run_variant(variant, keep_record, display_results, rerun_existing, saved_figure_ext):
        if len(variant.get_records(only_completed=True)) > 0 and not rerun_existing:
            logging.info(f">>> Variant {variant.name} already has completed records, skipping...")
            return

        return variant.run(keep_record=keep_record, display_results=display_results, saved_figure_ext=saved_figure_ext)

    @time_performance
    def run_parallel_experiments(self, num_processes=8, rerun_existing=False, display_results=True):
        from pathos.multiprocessing import ProcessingPool as Pool

        partial_run_func = partial(ExperimentManager.run_variant, keep_record=True, display_results=display_results,
                           rerun_existing=rerun_existing, saved_figure_ext='.fig.png')

        with Pool(num_processes) as pool:
            records = pool.map(partial_run_func, self.variants)
            pool.close()
            pool.join()

        for record in records:
            if record is not None: # empty records if experiments already exist
                self._save_rockstars(record)


    @time_performance
    def run_experiments(self, rerun_existing=False, display_results=True):
        for i, variant in enumerate(self.variants):
            if len(variant.get_records(only_completed=True)) > 0 and not rerun_existing:
                logging.info(f">>> Variant {variant.name} already has completed records, skipping...")
                continue

            logging.info(f"Running variant {i}")
            record = variant.run(keep_record=True, display_results=display_results, saved_figure_ext='.fig.png')
            self._save_rockstars(record)

    def _save_rockstars(self, record):
        if SAVE_HOF_AND_BEST:
            hof, best = record.get_result()
            fitness_function = record.get_args()['fitness_function']
            grammar_version = record.get_args()['grammar_version']
            with open(HOF_AND_BEST_FILENAME, 'a') as f:
                f.write(
                    f'hof:{record.get_experiment_id()}:{grammar_version}:{fitness_function._name}:'
                    f'{"&".join([str(i) + "/" + str(i.fitness.values[0]) for i in hof])}\n'
                )
                f.write(
                    f'gen_best:{record.get_experiment_id()}:{grammar_version}:{fitness_function._name}:'
                    f'{"&".join([str(i) + "/" + str(i.fitness.values[0]) for i in best])}\n'
                )

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


    def print_best_individual_statistics(self, variant, datasets):
        best_individual, evaluations = self.get_best_performing_across_datasets(variant, datasets)
        print(f"Best performing individual across datasets in variant {variant}:\n")
        for evaluation in evaluations:
            self._print_individual_info(best_individual, evaluation)


    def get_best_performing_across_datasets(self, variant, datasets):
        """
        Finds the best performing individual in an experiment variant, across all datasets.
        :param variant: experiment variant
        :param datasets: a collection of Data objects
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
        for data in datasets:
            evaluation = self._build_evaluation_object(best_individual, variant, data)
            evaluations.append(evaluation)
        return best_individual, evaluations

    def get_best_performing_across_variants_and_datasets(self, datasets, sort_by=["mean_profit"]):
        """
        Returns a list of best performing individuals, one per experiment variant.
        :return:
        """
        df = pd.DataFrame(columns=["experiment_name", "doge", "fitness_function", "fitness_value",
                                   "mean_profit", "std_profit", "max_profit", "min_profit", "all_profits",
                                   "benchmark_profits", "differences", "variant",
                                   "evaluations", "individual"])
        for variant in self.get_variants():
            best_individual, evaluations = self.get_best_performing_across_datasets(variant, datasets)
            profits = [evaluation.profit_percent for evaluation in evaluations]
            benchmark_profits = [evaluation.benchmark_backtest.profit_percent for evaluation in evaluations]
            differences = [x1-x2 for (x1, x2) in zip(profits, benchmark_profits)]
            df = df.append({"experiment_name": str(variant.name),
                       "doge": str(best_individual),
                       "fitness_function": self.get_db_record(variant)["fitness_function"]._name,
                       "fitness_value" : self._get_fitness(best_individual, variant, datasets),
                       "mean_profit": np.mean(profits),
                       "std_profit": np.std(profits),
                       "max_profit": np.max(profits),
                       "min_profit": np.min(profits),
                       "all_profits": ", ".join(map(str,profits)),
                       "benchmark_profits": ", ".join(map(str, benchmark_profits)),
                       "differences": ", ".join(map(str, differences)),
                       "variant" : variant,
                       "evaluations": evaluations,
                       "individual": best_individual}, ignore_index=True)

        df = df.sort_values(by=sort_by, ascending=False)
        return df


    @time_performance
    def produce_report(self, top_n, out_filename, start_time, end_time, counter_currency,
                       sources=[0], resample_periods=[60], start_cash=1000, start_crypto=0):
        performance_df = self.get_best_performing_across_variants_and_datasets(self.training_data)

        genetic_strategies = []

        # we will evaluate the performance of top n individuals
        for i in range(top_n):
            individual = performance_df.iloc[i].individual
            genetic_strategies.append(individual)

        ann_rsi_strategies, basic_strategies, vbi_strategies = build_itf_baseline_strategies()
        strategies = ann_rsi_strategies + basic_strategies + vbi_strategies


        logging.info('Running baseline evaluations')

        genetic_backtests = []
        genetic_baselines = []

        for (source, resample_period) in itertools.product(sources, resample_periods):
            transaction_currencies = get_currencies_trading_against_counter(counter_currency, source)

            logging.info(f'Running genetic backtests for source {source}, resample period {resample_period}')
            data_collection = []

            for transaction_currency in transaction_currencies:
                try:
                    data = Data(
                        start_time=start_time,
                        end_time=end_time,
                        transaction_currency=transaction_currency,
                        counter_currency=counter_currency,
                        resample_period=resample_period,
                        source=source,
                        start_cash=start_cash,
                        start_crypto=start_crypto
                    )
                    data_collection.append(data)
                except Exception as e: # in case of missing data
                    logging.error(f'Unable to load data for {transaction_currency}-{counter_currency}, skipping...')
                    logging.error(str(e))

            # build a new TAProvider based on the loaded data
            ta_provider = TAProviderCollection(data_collection)

            # do genetic program backtesting on the new data
            for data in data_collection:
                for i in range(top_n):
                    row = performance_df.iloc[i]
                    try:
                        evaluation = self._build_evaluation_object(row.individual, row.variant, data, function_provider=ta_provider)
                        genetic_backtests.append(evaluation)
                        genetic_baselines.append(evaluation.benchmark_backtest)

                    except NoPriceDataException as e:
                        logging.error(e)

        comp = ComparativeEvaluation(strategies, counter_currencies=[counter_currency],
                                     resample_periods=resample_periods,
                                     sources=sources, start_cash=start_cash, start_crypto=start_crypto,
                                     start_time=start_time, end_time=end_time, debug = False)


        report = ComparativeReportBuilder(comp.backtests + genetic_backtests, comp.baselines + genetic_baselines)
        report.all_coins_report(out_filename, group_strategy_variants=False)


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
            row = self.generate_performance_df_row(data, individual, variant, rank)
            performance_rows.append(row)

        performance_info = pd.DataFrame(performance_rows)
        performance_info = performance_info.sort_values(by=['profit_percent'], ascending=False)
        return performance_info

    def generate_performance_df_row(self, data, individual, variant, rank=None):
        evaluation = self._build_evaluation_object(individual, variant, data)
        row = evaluation.to_primitive_types_dictionary()
        row['experiment_id'] = variant.name
        row['hof_ranking'] = rank
        row["individual"] = individual
        row["individual_str"] = str(individual)
        row["evaluation"] = evaluation
        return row

    def evaluate_individual_on_data_collection(self, individual, variant, data_collection):
        performance_df = pd.DataFrame()
        for data in data_collection:
            row = self.generate_performance_df_row(data, individual, variant)
            performance_df = performance_df.append(row, ignore_index=True)
        return performance_df

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
                self.performance_df_row_info(performance_info.iloc[0], data)
            joined_dfs.append(performance_info)
        return joined_dfs

    def performance_df_row_info(self, performance_df_row, data=None):
        print(f'Experiment id: {performance_df_row.experiment_id}\n')
        individual = performance_df_row.individual
        evaluation = performance_df_row.evaluation
        self._print_individual_info(individual, evaluation, data)
        return individual

    def _print_individual_info(self, individual, evaluation, data=None):
        evaluation.plot_price_and_orders()
        if not data is None:
            data.plot(evaluation.orders, str(individual))
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

    def get_db_record(self, variant):
        return self.experiment_db[variant.name[len("run_evolution."):]]

    def _build_genetic_program(self, variant, data, function_provider=None):
        if function_provider is None:
            function_provider = self.function_provider
        db_record = self.get_db_record(variant)
        grammar = Grammar.construct(
            grammar_name=db_record['grammar_version'],
            function_provider=function_provider,
            ephemeral_suffix=db_record['experiment_id']
        )
        genetic_program = GeneticProgram(
            data_collection=data,
            function_provider=function_provider,
            grammar=grammar,
            fitness_function=db_record['fitness_function'],
            tree_depth=db_record['tree_depth'],
            order_generator=db_record['order_generator']

        )
        return genetic_program

    def _build_evaluation_object(self, individual, variant, data, function_provider=None):
        genetic_program = self._build_genetic_program(variant, data, function_provider)
        if function_provider is not None: # rebuild individual with the new data
            individual = genetic_program.individual_from_string(str(individual))
        return genetic_program.build_evaluation_object(individual, data)

    def _get_fitness(self, individual, variant, data):
        genetic_program = self._build_genetic_program(variant, data)
        return genetic_program.compute_fitness_over_datasets(individual)[0]

    def plot_data(self):
        self.training_data[0].plot()

    def _load_rockstars(self, grammar_version, fitness_function_name, to_load = ['hof', 'gen_best'], limit_top=None):
        individuals = []
        fitnesses = []

        with open(HOF_AND_BEST_FILENAME, 'r') as f:
            for line in f:
                line = line.split(':')
                if not line[0] in to_load or line[2] != grammar_version or line[3] != fitness_function_name:
                    continue
                entries = line[4].split('&')
                for entry in entries:
                    individual_str, fitness = entry.split('/')
                    if individual_str in individuals:  # we don't want repeat individuals
                        continue
                    individuals.append(individual_str)
                    fitnesses.append(fitness)

        individuals = [x for _, x in sorted(zip(fitnesses, individuals))]
        if limit_top is None:
            return individuals
        else:
            return individuals[:min(len(individuals), limit_top)]


class ExperimentDB:

    def __init__(self):
        self._experiments = {}
        self._num_records = 0

    def add(self, data, function_provider, grammar_version, fitness_function,
            mating_prob, mutation_prob, population_size, num_generations, premade_individuals,
            order_generator, tree_depth, reseed_params):
        entry = {}
        entry['data'] = data
        entry['function_provider'] = function_provider
        entry['grammar_version'] = grammar_version
        entry['fitness_function'] = fitness_function
        entry['mating_prob'] = mating_prob
        entry['mutation_prob'] = mutation_prob
        entry['population_size'] = population_size
        entry['num_generations'] = num_generations
        entry['premade_individuals'] = premade_individuals
        entry['order_generator'] = order_generator
        entry['tree_depth'] = tree_depth
        entry['reseed_params'] = reseed_params
        entry['experiment_id'] = self._num_records
        name = self.build_experiment_id(**entry)
        self._experiments[name] = entry
        self._num_records += 1

    def get_all(self):
        for k, v in self._experiments.items():
            yield k, v

    def build_experiment_id(self, **kwargs):
        return f"d_{'-'.join(map(str,kwargs['data']))};" \
               f"{kwargs['grammar_version']};" \
               f"{kwargs['fitness_function']};" \
               f"x_{kwargs['mating_prob']};" \
               f"m_{kwargs['mutation_prob']};" \
               f"n_{kwargs['population_size']};" \
               f"gen_{kwargs['num_generations']};" \
               f"td_{kwargs['tree_depth']};" \
               f"{'p' if kwargs['order_generator'] == 'position_based' else 'a'};" \
               f"{'rs' if kwargs['reseed_params']['enabled'] == True else 'nrs'}"

    #          f"provider_{kwargs['function_provider']};" \

    def __getitem__(self, key):
        return self._experiments[key]



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

#e = ExperimentManager("parallel_test.json")
e = ExperimentManager("gv4_experiments.json")

if __name__ == "__main__":
    import datetime
    start_time = datetime.datetime(2018, 4, 9, 14, 0, tzinfo=datetime.timezone.utc).timestamp()
    end_time = datetime.datetime(2018, 6, 1, 0, 0, tzinfo=datetime.timezone.utc).timestamp()

    #e.run_experiments()
    #e.produce_report(5, 'gp_backtesting_BTC_train.xlsx', start_time, end_time, 'BTC')


    #start_time = datetime.datetime(2018, 6, 1, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    #end_time = datetime.datetime(2018, 8, 1, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    #e.produce_report(5, 'gp_backtesting_USDT_6_res60_2.xlsx', start_time, end_time, 'USDT')
    e.produce_report(5, 'gp_backtesting_BTC_6_res60_2.xlsx', start_time, end_time, 'BTC')


    #e.run_experiments()
    #e.run_parallel_experiments()
    #from pathos.multiprocessing import ProcessingPool as Pool

    #with Pool(1) as pool:
    #    print(pool.map(run_variant, e.variants))

    #e.produce_report(1, 'gp_backtesting.xlsx')
    exit(0)


    performance_dfs = e.get_joined_performance_dfs_over_all_variants()
    e.performance_df_row_info(performance_dfs[0].iloc[0])


    #e.explore_records()
    #e.best_individuals_across_variants_and_datasets = e.get_best_performing_across_variants_and_datasets(e.training_data)
    #dfs = e.get_joined_performance_dfs_over_all_variants()
    df = e.get_best_performing_across_variants_and_datasets(e.training_data)
    from gp_utils import compress

    print(compress(df.iloc[0].individual))
    print(str(df.iloc[0].individual))
    #e.browse_variants()


