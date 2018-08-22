import operator
import os
import logging
import types
from deap import creator, tools, base
from backtesting.signals import Signal
from backtesting.strategies import SignalStrategy, Horizon, Strength, TickerStrategy, StrategyDecision
from chart_plotter import *
from custom_deap_algorithms import combined_mutation, eaSimpleCustom
from gp_data import Data
from backtester_ticks import TickDrivenBacktester
from grammar import GrammarV1
from leaf_functions import TAProvider
from tick_provider import PriceDataframeTickProvider
from abc import ABC, abstractmethod
import dill as pickle
import random

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

HISTORY_SIZE = 200

class Grammar(ABC):

    def __init__(self, function_provider):
        self.function_provider = function_provider

    @property
    def pset(self):
        return self._pset

    @property
    @abstractmethod
    def name(self):
        pass


class GeneticTickerStrategy(TickerStrategy):
    def __init__(self, tree, data, gp_object, history_size=HISTORY_SIZE):
        self.data = data
        self.transaction_currency = data.transaction_currency
        self.counter_currency = data.counter_currency
        self.resample_period = data.resample_period
        self.strength = Strength.any
        self.source = data.source
        self.start_time = data.start_time
        self.end_time = data.end_time
        self.tree = tree
        self.gp_object = gp_object
        self.history_size = history_size
        self.i = 0
        self.func = self.gp_object.toolbox.compile(expr=self.tree)

    def process_ticker(self, price_data, signals):
        """
        :param price_data: Pandas row with OHLC data and timestamp.
        :param signals: ITF signals co-occurring with price tick.
        :return: StrategyDecision.BUY or StrategyDecision.SELL or StrategyDecision.IGNORE
        """
        self.i += 1

        price = price_data.close_price
        timestamp = price_data.Index

        if self.i < self.history_size:
            # outcomes.append("skipped")
            return StrategyDecision.IGNORE, None
        outcome = self.func([timestamp])

        decision = StrategyDecision.IGNORE
        signal = None
        if outcome == self.gp_object.function_provider.buy:
            decision = StrategyDecision.BUY
            signal = Signal("Genetic", 1, None, 3, 3, price, 0, timestamp, None, self.transaction_currency,
                            self.counter_currency, self.source, self.resample_period)
        elif outcome == self.gp_object.function_provider.sell:
            decision = StrategyDecision.SELL
            signal = Signal("Genetic", -1, None, 3, 3, price, 0, timestamp, None, self.transaction_currency,
                            self.counter_currency, self.source, self.resample_period)
        elif not outcome == self.gp_object.function_provider.ignore:
            logging.warning("Invalid outcome encountered")

        return decision, signal

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == "Genetic"

    def get_short_summary(self):
        return("Strategy: evolved using genetic programming\nRule set: {}".format(str(self.tree)))


class GeneticSignalStrategy(SignalStrategy):
    def __init__(self, tree, data, gp_object, history_size=HISTORY_SIZE):
        self.data = data
        self.transaction_currency = data.transaction_currency
        self.counter_currency = data.counter_currency
        self.resample_period = data.resample_period
        self.strength = Strength.any
        self.source = data.source
        self.start_time = data.start_time
        self.end_time = data.end_time
        self.tree = tree
        self.gp_object = gp_object
        self.history_size = history_size
        self.build_from_gp_tree(tree)

    def get_orders(self, signals, start_cash, start_crypto, source, time_delay=0, slippage=0):
        return SignalStrategy.get_orders(self, self.signals, start_cash, start_crypto, source, time_delay, slippage)

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == "Genetic"

    def build_from_gp_tree(self, tree):
        self.signals = []
        func = self.gp_object.toolbox.compile(expr=tree)

        outcomes = []
        for i, row in enumerate(self.data.price_data.itertuples()):
            price = row.close_price
            timestamp = row.Index
            if i < self.history_size:
                outcomes.append("skipped")
                continue
            outcome = func([timestamp])

            trend = None
            if outcome == self.gp_object.buy:
                trend = 1
            elif outcome == self.gp_object.sell:
                trend = -1
            elif not outcome == self.gp_object.ignore:
                logging.warning("Invalid outcome encountered")
            if trend != None:
                signal = Signal("Genetic", trend, self.horizon, 3, 3, price, 0, timestamp, None, self.transaction_currency,
                                self.counter_currency, self.source, self.resample_period)
                self.signals.append(signal)

            outcomes.append(outcome.__name__)
        df = self.data.to_dataframe()
        df['outcomes'] = pd.Series(outcomes, index=df.index)
        self.df_data_and_outcomes = df

    def get_short_summary(self):
        return("Strategy: evolved using genetic programming\nRule set: {}".format(str(self.tree)))

    def get_dataframe_with_outcomes(self):
        return self.df_data_and_outcomes


class FitnessFunction(ABC):
    @abstractmethod
    def compute(self, individual, evaluation, genetic_program):
        pass

    @staticmethod
    def construct(fitness_function_name):
        for subclass in FitnessFunction.__subclasses__():
            if subclass._name == fitness_function_name:
                return subclass()
        raise Exception(f"Unknown function {fitness_function_name}!")

    @property
    @abstractmethod
    def name(self):
        pass

    def __str__(self):
        return self._name


class FitnessFunctionV1(FitnessFunction):
    _name = "ff_v1"

    def compute(self, individual, evaluation, genetic_program):
        max_len = 3 ** genetic_program.tree_depth
        return evaluation.profit_percent + (max_len - len(individual)) / float(max_len) * 20 \
               + evaluation.num_sells * 5,
    @property
    def name(self):
        return self._name

class FitnessFunctionV2(FitnessFunction):
    _name = "ff_v2"

    def compute(self, individual, evaluation, genetic_program):
        max_len = 1 ** genetic_program.tree_depth
        return evaluation.profit_percent + (max_len - len(individual)) / float(max_len) * 10 \
               + evaluation.num_sells * 4,
    @property
    def name(self):
        return self._name

class GeneticProgram:
    def __init__(self, data, **kwargs):
        self.data = data
        self.function_provider = kwargs.get('function_provider')
        self.grammar = kwargs.get('grammar')
        assert self.function_provider == self.grammar.function_provider
        self.fitness = kwargs.get('fitness_function', FitnessFunctionV1())
        self.tree_depth = kwargs.get('tree_depth', 5)
        self._build_toolbox()

    @property
    def pset(self):
        return self.grammar.pset

    def _build_toolbox(self):

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=self.tree_depth)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)
        toolbox.register("evaluate", self.compute_fitness)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", combined_mutation, min_=0, max_=self.tree_depth)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)
        toolbox.register("mutate", combined_mutation, expr=toolbox.expr_mut, pset=self.pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.tree_depth))  # 17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.tree_depth))  # 17))
        self.toolbox = toolbox


    def evolve(self, mating_prob, mutation_prob, population_size,
               num_generations, verbose=True, output_folder=None, run_id=None):
        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(10)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log, best = eaSimpleCustom(pop, self.toolbox, mating_prob, mutation_prob, num_generations, stats=mstats,
                                        halloffame=hof, verbose=verbose)

        if verbose:
            logging.info("Winners in each generation: ")
            for i in range(len(best)):
                logging.info(f"  {i}  {best[i]}")
            logging.info("Hall of fame: ")
            for i in range(len(hof)):
                logging.info(f"  {i}  {hof[i]}")

            best_individual = hof[0]
            evaluation = self.build_evaluation_object(best_individual)
            draw_price_chart(self.data.timestamps, self.data.prices, evaluation.orders)
            #logging.info(evaluation.get_report())
            #draw_tree(best_individual)

        if output_folder != None:
            hof_name = self.get_hof_filename(mating_prob, mutation_prob, run_id)
            gen_best_name = self.get_gen_best_filename(mating_prob, mutation_prob, run_id)
            out_path_hof = os.path.join(output_folder, hof_name)
            out_path_gen_best = os.path.join(output_folder, gen_best_name)
            if os.path.exists(out_path_hof) and os.path.exists(out_path_gen_best):
                logging.warning(f"{hof_name} already exists, skipping...")
            else:
                pickle.dump(hof, open(out_path_hof, "wb"))
                pickle.dump(best, open(out_path_gen_best, "wb"))

        return hof, best


    #@time_performance
    def compute_fitness(self, individual, super_verbose=False):
        evaluation = self.build_evaluation_object(individual)

        if evaluation.num_trades > 1 and super_verbose:
            draw_price_chart(self.data.timestamps, self.data.prices, evaluation.orders)
            logging.info(str(individual))
            logging.info(evaluation.get_report())
            draw_tree(individual)

        return self.fitness.compute(individual, evaluation, self)


    def build_evaluation_object(self, individual, ticker=True):
        if not ticker:
            strategy = GeneticSignalStrategy(individual, self.data, self)
            evaluation = strategy.evaluate(self.data.transaction_currency, self.data.counter_currency,
                                           self.data.start_cash, self.data.start_crypto,
                                           self.data.start_time, self.data.end_time, self.data.source, 60, verbose=False)

        else:
            strategy = GeneticTickerStrategy(individual, self.data, self)
            tick_provider = PriceDataframeTickProvider(self.data.price_data)

            # create a new tick based backtester
            evaluation = TickDrivenBacktester(
                tick_provider=tick_provider,
                strategy=strategy,
                transaction_currency=self.data.transaction_currency,
                counter_currency=self.data.counter_currency,
                start_cash=self.data.start_cash,
                start_crypto=self.data.start_crypto,
                start_time=self.data.start_time,
                end_time=self.data.end_time,
                benchmark_backtest=None,
                time_delay=0,
                slippage=0,
                verbose=False
            )
        return evaluation


    @staticmethod
    def get_gen_best_filename(mating_prob, mutation_prob, run_id):
        gen_best_name = "doge_{}_x-{}_m-{}-gen_best.p".format(run_id, mating_prob, mutation_prob)
        return gen_best_name

    @staticmethod
    def get_hof_filename(mating_prob, mutation_prob, run_id):
        hof_name = "doge_{}_x-{}_m-{}-hof.p".format(run_id, mating_prob, mutation_prob)
        return hof_name

    @staticmethod
    def parse_evolution_filename(filename):
        filename = filename.split("_")
        run = int(filename[1])
        mating_prob = float(filename[2].split("-")[1])
        mutation_prob = float(filename[3].split("-")[1])
        return run, mating_prob, mutation_prob

    def load_evolution_file(self, file_path):
        return pickle.load(open(file_path, "rb"))


if __name__ == '__main__':
    transaction_currency = "OMG"
    counter_currency = "BTC"
    end_time = 1526637600
    start_time = end_time - 60 * 60 * 24 * 30
    validation_start_time = start_time - 60 * 60 * 24 * 30
    validation_end_time = start_time
    horizon = Horizon.short
    resample_period = 60
    start_cash = 1
    start_crypto = 0
    source = 0
    history_size = 100

    training_data = Data(start_time, end_time, transaction_currency, counter_currency, resample_period, start_cash,
                         start_crypto, source)

    validation_data = Data(validation_start_time, validation_end_time, transaction_currency, counter_currency,
                           resample_period, start_cash, start_crypto, source)

    data = training_data
    gprog = GeneticProgram(data)
    #record = run_experiment.run(gprog=gprog, mating_prob=0.7, mutation_prob=0.5, population_size=50, num_generations=2)

    #run_experiment.add_variant("test", gprog=gprog, mating_prob=0.7, mutation_prob=0.5, population_size=50, num_generations=2)
    #ex = run_experiment.get_variant("test")
    #record = ex.run(keep_record=True, display_results=True)

    #print(record)
    #run_experiment.add_variant(gprog=gprog, mating_prob=0.8, mutation_prob=0.8, population_size=50, num_generations=2)
    #run_experiment.browse(close_after=False)

    #records = run_experiment.get_records()
    #print(records)
    #result = records[0].get_result()

    # Note: to make Artemis work
    # change line 3 in persistent_ordered_dict.py to import dill as pickle (TODO: fork)

    # gprog.evolve(0.8, 0.8, 50, 5)

    # records = variants[0].get_records()
    # print(records)


