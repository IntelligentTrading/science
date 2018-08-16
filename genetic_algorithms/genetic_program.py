from deap import base
from deap import algorithms
from deap import creator
from deap import tools
import operator
import random
import types
from backtesting.signals import Signal
from backtesting.strategies import SignalStrategy, Horizon, Strength, TickerStrategy, StrategyDecision
from chart_plotter import *
from custom_deap_algorithms import combined_mutation, eaSimpleCustom
from gp_data import Data
import os
import dill as pickle
from backtester_ticks import TickDrivenBacktester

HISTORY_SIZE = 200


class GeneticTradingStrategy(TickerStrategy):
    def __init__(self, tree, data, gp_object, history_size=HISTORY_SIZE):
        self.data = data
        self.horizon = data.horizon
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
        # self.build_from_gp_tree(tree)
        self.i = 0

    def process_ticker(self, price_data, signals):
        """
        :param price_data: Pandas row with OHLC data and timestamp.
        :param signals: ITF signals co-ocurring with price tick.
        :return: StrategyDecision.BUY or StrategyDecision.SELL or StrategyDecision.IGNORE
        """
        self.i += 1
        func = self.gp_object.toolbox.compile(expr=self.tree)
        price = price_data.close_price
        timestamp = price_data['timestamp']
        if self.i < self.history_size:
            # outcomes.append("skipped")
            return StrategyDecision.IGNORE, None
        outcome = func([timestamp])

        decision = StrategyDecision.IGNORE
        signal = None
        if outcome == self.gp_object.buy:
            decision = StrategyDecision.BUY
            signal = Signal("Genetic", 1, None, 3, 3, price, 0, timestamp, None, self.transaction_currency,
                            self.counter_currency, self.source, self.resample_period)
        elif outcome == self.gp_object.sell:
            decision = StrategyDecision.SELL
            signal = Signal("Genetic", -1, None, 3, 3, price, 0, timestamp, None, self.transaction_currency,
                            self.counter_currency, self.source, self.resample_period)
        elif not outcome == self.gp_object.ignore:
            print("WARNING: Invalid outcome encountered")

        return decision, signal

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == "Genetic"

    def get_short_summary(self):
        return("Strategy: evolved using genetic programming\nRule set: {}".format(str(self.tree)))



"""
class GeneticTradingStrategy(Strategy):
    def __init__(self, tree, data, gp_object, history_size=HISTORY_SIZE):
        self.data = data
        self.horizon = data.horizon
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
                print("WARNING: Invalid outcome encountered")
            if trend != None:
                signal = Signal("Genetic", trend, self.horizon, 3, 3, price, 0, timestamp, None, self.transaction_currency,
                                self.counter_currency, self.source, self.resample_period)
                self.signals.append(signal)

            outcomes.append(outcome.__name__)
        df = self.data.to_dataframe()
        df['outcomes'] = pd.Series(outcomes, index=df.index)
        self.df_data_and_outcomes = df

        #print(self.signals)
        #print(str(tree))
        #print(outcomes)

    def get_short_summary(self):
        return("Strategy: evolved using genetic programming\nRule set: {}".format(str(self.tree)))


    def get_dataframe_with_outcomes(self):
        return self.df_data_and_outcomes

"""

class GeneticProgram:
    def __init__(self, data, tree_depth=5):
        self.data = data
        self.tree_depth = tree_depth
        self.build_grammar()
        self.build_toolbox()

    def if_then_else(self, input, output1, output2):
        try:
            return output1 if input else output2
        except:
            return output1

    def rsi(self, input):
        timestamp = input[0]
        timestamp_index = np.where(self.data.price_data.index == timestamp)[0]
        return self.data.rsi_data[timestamp_index]


    def sma50(self, input):
        timestamp = input[0]
        timestamp_index = np.where(self.data.price_data.index == timestamp)[0]
        return self.data.sma50_data[timestamp_index]


    def ema50(self, input):
        timestamp = input[0]
        timestamp_index = np.where(self.data.price_data.index == timestamp)[0]
        return self.data.ema50_data[timestamp_index]


    def sma200(self, input):
        timestamp = input[0]
        timestamp_index = np.where(self.data.price_data.index == timestamp)[0]
        return self.data.sma200_data[timestamp_index]


    def ema200(self, input):
        timestamp = input[0]
        timestamp_index = np.where(self.data.price_data.index == timestamp)[0]
        return self.data.ema200_data[timestamp_index]


    def price(self, input):
        timestamp = input[0]
        return self.data.price_data.loc[timestamp,"close_price"]


    def buy(self):
        pass


    def sell(self):
        pass


    def ignore(self):
        pass


    def identity(self, x):
        return x


    def build_grammar(self):
        pset = gp.PrimitiveSetTyped("main", [list], types.FunctionType)
        pset.addPrimitive(operator.lt, [float, float], bool)
        pset.addPrimitive(operator.gt, [float, float], bool)
        pset.addPrimitive(operator.or_, [bool, bool], bool)
        pset.addPrimitive(operator.and_, [bool, bool], bool)
        pset.addPrimitive(self.if_then_else, [bool, types.FunctionType, types.FunctionType], types.FunctionType)
        pset.addPrimitive(self.rsi, [list], float)
        pset.addPrimitive(self.sma50, [list], float)
        pset.addPrimitive(self.ema50, [list], float)
        pset.addPrimitive(self.sma200, [list], float)
        pset.addPrimitive(self.ema200, [list], float)
        pset.addPrimitive(self.price, [list], float)
        pset.addTerminal(False, bool)
        pset.addTerminal(True, bool)
        pset.addTerminal(self.buy, types.FunctionType)
        pset.addTerminal(self.sell, types.FunctionType)
        pset.addTerminal(self.ignore, types.FunctionType)
        pset.addPrimitive(self.identity, [bool], bool, name="identity_bool")
        pset.addPrimitive(self.identity, [list], list, name="identity_list")
        pset.addPrimitive(self.identity, [float], float, name="identity_float")
        pset.addEphemeralConstant("rsi_overbought_threshold", lambda: random.uniform(70, 100), float)
        pset.addEphemeralConstant("rsi_oversold_threshold", lambda: random.uniform(0, 30), float)
        self.pset = pset

    def evaluate_individual(self, individual, super_verbose=False):
        strategy = GeneticTradingStrategy(individual, self.data, self)
        from tick_provider import PriceDataframeTickProvider
        # supply ticks from the ITF DB
        tick_provider = PriceDataframeTickProvider(self.data.price_data)

        # create a new tick based backtester
        evaluation = TickDrivenBacktester(tick_provider=tick_provider,
                                          strategy=strategy,
                                          transaction_currency=transaction_currency,
                                          counter_currency=counter_currency,
                                          start_cash=start_cash,
                                          start_crypto=start_crypto,
                                          start_time=start_time,
                                          end_time=end_time,
                                          benchmark_backtest=None,
                                          time_delay=0,
                                          slippage=0,
                                          verbose=False
                                          )

        if evaluation.num_trades > 1:
            if super_verbose and False:
                orders, _ = strategy.get_orders(1, 0)
                draw_price_chart(self.data.timestamps, self.data.prices, orders)
                print(str(individual))
                print(evaluation.get_report())
                draw_tree(individual)
        max_len = 3 ** self.tree_depth
        return evaluation.profit_percent + (max_len - len(individual)) / float(max_len) * 20 \
               + evaluation.num_sells * 5,


    def build_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=self.tree_depth)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)
        toolbox.register("evaluate", self.evaluate_individual)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", combined_mutation, min_=0, max_=self.tree_depth)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)
        toolbox.register("mutate", combined_mutation, expr=toolbox.expr_mut, pset=self.pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.tree_depth))  # 17))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.tree_depth))  # 17))
        self.toolbox = toolbox

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
            print("Winners in each generation: ")
            for i in range(len(best)):
                print(i, best[i])
            print("Hall of fame: ")
            for i in range(len(hof)):
                print(hof[i])

        if output_folder != None:
            hof_name = self.get_hof_filename(mating_prob, mutation_prob, run_id)
            gen_best_name = self.get_gen_best_filename(mating_prob, mutation_prob, run_id)
            out_path_hof = os.path.join(output_folder, hof_name)
            out_path_gen_best = os.path.join(output_folder, gen_best_name)
            if False: #os.path.exists(out_path_hof) and os.path.exists(out_path_gen_best):
                print("{} already exists, skipping...".format(hof_name))
            else:
                pickle.dump(hof, open(out_path_hof, "wb"))
                pickle.dump(best, open(out_path_gen_best, "wb"))




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

    training_data = Data(start_time, end_time, transaction_currency, counter_currency, resample_period, horizon,
                         start_cash,
                         start_crypto, source)

    validation_data = Data(validation_start_time, validation_end_time, transaction_currency, counter_currency,
                           resample_period, horizon,
                           start_cash, start_crypto, source)

    data = training_data
    gp = GeneticProgram(data)
    gp.evolve(0.8, 0.8, 50, 5)

