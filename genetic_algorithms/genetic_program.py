import operator
import os
import logging
import numpy as np
from deap import creator, tools, base
from backtesting.signals import Signal
from backtesting.strategies import SignalStrategy, Strength, TickerStrategy, StrategyDecision
from chart_plotter import *
from custom_deap_algorithms import combined_mutation, eaSimpleCustom
from backtester_ticks import TickDrivenBacktester
from tick_provider import PriceDataframeTickProvider
from abc import ABC, abstractmethod
import dill as pickle

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

        if self.i <= self.history_size:
            # outcomes.append("skipped")
            return StrategyDecision.IGNORE, None
        outcome = self.func([timestamp, self.data.transaction_currency, self.data.counter_currency])

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
    def name(self):
        return self._name

    def __str__(self):
        return self._name


class FitnessFunctionV1(FitnessFunction):
    _name = "ff_v1"

    def compute(self, individual, evaluation, genetic_program):
        max_len = 3 ** genetic_program.tree_depth
        return evaluation.profit_percent + (max_len - len(individual)) / float(max_len) * 20 \
               + evaluation.num_sells * 5,

class BenchmarkDiffFitness(FitnessFunction):
    _name = "ff_benchmarkdiff"

    def compute(self, individual, evaluation, genetic_program):
        return evaluation.profit_percent - evaluation.benchmark_backtest.profit_percent,

class BenchmarkDiffTrades(FitnessFunction):
    _name = "ff_benchmarkdiff_trades"

    def compute(self, individual, evaluation, genetic_program):
        return (evaluation.profit_percent - evaluation.benchmark_backtest.profit_percent)*evaluation.num_profitable_trades,

class BenchmarkLengthControlFitness(FitnessFunction):
    _name = "ff_benchlenctrl"

    def compute(self, individual, evaluation, genetic_program):
        max_len = 3 ** genetic_program.tree_depth
        return (evaluation.profit_percent - evaluation.benchmark_backtest.profit_percent) + \
               (max_len - len(individual)) / float(max_len) * 20,

class BenchmarkLengthControlFitnessV2(FitnessFunction):
    _name = "ff_benchlenctrl_v2"

    def compute(self, individual, evaluation, genetic_program):
        max_len = 3 ** genetic_program.tree_depth
        return (evaluation.profit_percent - evaluation.benchmark_backtest.profit_percent) * \
               (max_len - len(individual)) / float(max_len),

class BenchmarkLengthControlFitnessV3(FitnessFunction):
    _name = "ff_benchlenctrl_v3"

    def compute(self, individual, evaluation, genetic_program):
        max_len = 3 ** genetic_program.tree_depth
        return (evaluation.profit_percent - evaluation.benchmark_backtest.profit_percent) * \
               (1+0.1*(max_len - len(individual)) / float(max_len)),

class GeneticProgram:
    def __init__(self, data_collection, **kwargs):
        self.data_collection = data_collection
        self.function_provider = kwargs.get('function_provider')
        self.grammar = kwargs.get('grammar')
        self.fitness = kwargs.get('fitness_function', FitnessFunctionV1())
        self.tree_depth = kwargs.get('tree_depth', 5)
        self.combined_fitness_operator = kwargs.get('combined_fitness_operator', min)
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
        toolbox.register("evaluate", self.compute_fitness_over_datasets)
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
                                        halloffame=hof, verbose=True)

        if verbose:
            logging.info("Winners in each generation: ")
            for i in range(len(best)):
                logging.info(f"  {i}  {best[i]}")
            logging.info("Hall of fame: ")
            for i in range(len(hof)):
                logging.info(f"  {i}  {hof[i]}")
            draw_tree(hof[0])


        if output_folder != None:  # TODO: clean
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
    def compute_fitness(self, individual, data, super_verbose=False):
        evaluation = self.build_evaluation_object(individual, data)

        if evaluation.num_trades > 1 and super_verbose:
            logging.info(str(individual))
            logging.info(evaluation.get_report())
            draw_tree(individual)

        return self.fitness.compute(individual, evaluation, self)
    
    def compute_fitness_over_datasets(self, individual):
        fitnesses = []
        for i, data in enumerate(self.data_collection):
            fitnesses.append(self.compute_fitness(individual, data)[0])
        return self.combined_fitness_operator(fitnesses),
        

    def build_evaluation_object(self, individual, data, ticker=True):
        if not ticker:
            strategy = GeneticSignalStrategy(individual, data, self,
                                             history_size=self.grammar.longest_function_history_size)
            evaluation = strategy.evaluate(data.transaction_currency, data.counter_currency,
                                           data.start_cash, data.start_crypto,
                                           data.start_time, data.end_time, data.source, 60, verbose=False)

        else:
            strategy = GeneticTickerStrategy(tree=individual,
                                             data=data,
                                             gp_object=self,
                                             history_size=self.grammar.longest_function_history_size)
            tick_provider = PriceDataframeTickProvider(data.price_data)

            # create a new tick based backtester
            evaluation = TickDrivenBacktester(
                tick_provider=tick_provider,
                strategy=strategy,
                transaction_currency=data.transaction_currency,
                counter_currency=data.counter_currency,
                start_cash=data.start_cash,
                start_crypto=data.start_crypto,
                start_time=data.start_time,
                end_time=data.end_time,
                benchmark_backtest=data.build_buy_and_hold_benchmark(
                    num_ticks_to_skip=self.grammar.longest_function_history_size
                ),
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

    @staticmethod
    def compress_individual(individual):
        for i, element in enumerate(individual):

            if individual == 'True':
                return True
            if individual == 'False':
                return False
            if individual == 'identity':
                return 'identity'

            if individual.arity == 0:
                return individual
            elements = individual[i:i+element.arity]
            if str(individual) == 'if_then_else':
                if GeneticProgram.compress_individual(elements[1]) == False:
                    return GeneticProgram.compress_individual(elements[2])
                elif GeneticProgram.compress_individual(elements[1]) == True:
                    return GeneticProgram.compress_individual(elements[1])
                else:
                    return 'if_then_else'

def visit_node(graph, node, labels, parent = None):
    children = [visit_node(graph, child, labels, node)
                for child in graph[node]
                if child != parent]
    return [labels[node], children]

def compress(individual):
    nodes, edges, labels = gp.graph(individual)
    from gp_utils import recompute_tree_graph
    graph = recompute_tree_graph(nodes, edges)
    root = visit_node(graph, 0, labels)
    optimized = optimize(root)
    return wrap_children(optimized)

def wrap_children(optimized):
    node, children = optimized
    if len(children) == 0:
        return node
    s = f'{str(node)}('
    for child in children:
        s += wrap_children(child)
        s += ','
    s = s[:-1]
    s += ')'
    return s

def optimize(node):
    node_type, children = node
    # optimize all children
    for i in range(len(children)):
        children[i] = optimize(children[i])
    if node_type == "if_then_else":
        cond_type = children[0][0]
        if cond_type == True:
            return children[1]
        if cond_type == False:
            return children[2]
    # can't optimize this node
    # (and children are already optimized)
    return node




