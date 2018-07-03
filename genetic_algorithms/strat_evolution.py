from deap import base
from deap import algorithms
from custom_deap_algorithms import eaSimpleCustom
from deap import creator
from deap import tools
import operator
import random
import types
import talib
from backtesting.signals import Signal
from backtesting.strategies import Strategy, Horizon, Strength, BuyAndHoldTimebasedStrategy
from data_sources import get_resampled_prices_in_range
from chart_plotter import *
import pandas as pd
import os
import dill as pickle
import time

SUPER_VERBOSE = False

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

class Data:
    def __init__(self, start_time, end_time, transaction_currency, counter_currency, resample_period, horizon,
                 start_cash, start_crypto, source):
        self.start_time = start_time
        self.end_time = end_time
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.resample_period = resample_period
        self.horizon = horizon
        self.start_cash = start_cash
        self.start_crypto = start_crypto
        self.source = source

        self.price_data = get_resampled_prices_in_range(start_time, end_time, transaction_currency, counter_currency, resample_period)
        self.rsi_data = talib.RSI(np.array(self.price_data.close_price, dtype=float), timeperiod=14)
        self.sma50_data = talib.SMA(np.array(self.price_data.close_price, dtype=float), timeperiod=50)
        self.ema50_data = talib.EMA(np.array(self.price_data.close_price, dtype=float), timeperiod=50)
        self.sma200_data = talib.SMA(np.array(self.price_data.close_price, dtype=float), timeperiod=200)
        self.ema200_data = talib.EMA(np.array(self.price_data.close_price, dtype=float), timeperiod=200)
        self.prices = self.price_data.as_matrix(columns=["close_price"])
        self.timestamps = pd.to_datetime(self.price_data.index.values, unit='s')
        assert len(self.prices) == len(self.timestamps)


training_data = Data(start_time, end_time, transaction_currency, counter_currency, resample_period, horizon, start_cash,
            start_crypto, source)

validation_data = Data(validation_start_time, validation_end_time, transaction_currency, counter_currency, resample_period, horizon,
            start_cash, start_crypto, source)

data = training_data

start_bah = int(data.price_data.iloc[history_size].name)
bah = BuyAndHoldTimebasedStrategy(start_bah, end_time, transaction_currency, counter_currency, source)
print("Buy and hold baseline: {0:0.2f}%".format(bah.evaluate(start_cash, start_crypto,
                                                             start_bah, end_time,
                                                             False, False).get_profit_percent()))


class GeneticTradingStrategy(Strategy):
    def __init__(self, tree, data):
        self.data = data
        self.horizon = data.horizon
        self.horizon = horizon
        self.transaction_currency = data.transaction_currency
        self.counter_currency = data.counter_currency
        self.strength = Strength.any
        self.source = data.source
        self.start_time = data.start_time
        self.end_time = data.end_time
        self.tree = tree
        self.build_from_gp_tree(tree)

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == "Genetic"

    def build_from_gp_tree(self, tree):
        self.signals = []
        func = toolbox.compile(expr=tree)

        outcomes = []
        for i, row in enumerate(data.price_data.itertuples()):
            price = row.close_price
            timestamp = row.Index
            if i < history_size:
                continue
            outcome = func([timestamp])
            trend = None
            if outcome == buy:
                trend = 1
            elif outcome == sell:
                trend = -1
            if trend != None:
                signal = Signal("Genetic", trend, self.horizon, 3, 3, price, 0, timestamp, None, self.transaction_currency,
                                self.counter_currency)
                self.signals.append(signal)

            outcomes.append(outcome)
        #print(self.signals)
        #print(str(tree))
        #print(outcomes)

    def get_short_summary(self):
        return("Strategy: evolved using genetic programming\nRule set: {}".format(str(self.tree)))


def if_then_else(input, output1, output2):
    try:
        return output1 if input else output2
    except:
        return output1


def rsi(input):
    timestamp = input[0]
    timestamp_index = np.where(data.price_data.index == timestamp)[0]
    return data.rsi_data[timestamp_index]


def sma50(input):
    timestamp = input[0]
    timestamp_index = np.where(data.price_data.index == timestamp)[0]
    return data.sma50_data[timestamp_index]


def ema50(input):
    timestamp = input[0]
    timestamp_index = np.where(data.price_data.index == timestamp)[0]
    return data.ema50_data[timestamp_index]


def sma200(input):
    timestamp = input[0]
    timestamp_index = np.where(data.price_data.index == timestamp)[0]
    return data.sma200_data[timestamp_index]


def ema200(input):
    timestamp = input[0]
    timestamp_index = np.where(data.price_data.index == timestamp)[0]
    return data.ema200_data[timestamp_index]


def price(input):
    timestamp = input[0]
    return data.price_data.loc[timestamp,"close_price"]


def buy():
    pass


def sell():
    pass


def ignore():
    pass


def identity(x):
    return x


def evaluate_individual(individual):
    data = training_data
    strategy = GeneticTradingStrategy(individual, data)
    training_evaluation = strategy.evaluate(start_cash, start_crypto, start_time, end_time, False, False)

    if training_evaluation.get_num_trades() > 1:
        if SUPER_VERBOSE:
            orders, _ = strategy.get_orders(start_cash, start_crypto)
            draw_price_chart(data.timestamps, data.prices, orders)
            print(str(individual))
            print(training_evaluation.get_report())
            draw_tree(individual)

    data = validation_data  # this pointer switching is very ugly, but I have to do it like this because of the DEAP library
    strategy = GeneticTradingStrategy(individual, data)
    validation_evaluation = strategy.evaluate(start_cash, start_crypto, start_time, end_time, False, False)

    max_len = 3**TREE_DEPTH
    return training_evaluation.get_profit_percent()+(max_len-len(individual))/float(max_len)*20,


def combined_mutation(individual, expr, pset):
    if random.random() > 0.5:
        return gp.mutInsert(individual, pset)
    else:
        return gp.mutEphemeral(individual, "one")




pset = gp.PrimitiveSetTyped("main", [list], types.FunctionType)
pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(if_then_else, [bool, types.FunctionType, types.FunctionType], types.FunctionType)
pset.addPrimitive(rsi, [list], float)
pset.addPrimitive(sma50, [list], float)
pset.addPrimitive(ema50, [list], float)
pset.addPrimitive(sma200, [list], float)
pset.addPrimitive(ema200, [list], float)
pset.addPrimitive(price, [list], float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)
pset.addTerminal(buy, types.FunctionType)
pset.addTerminal(sell, types.FunctionType)
pset.addTerminal(ignore, types.FunctionType)
pset.addPrimitive(identity, [bool], bool)
pset.addPrimitive(identity, [list], list)
pset.addPrimitive(identity, [float], float)
pset.addEphemeralConstant("rsi_overbought_threshold", lambda: random.uniform(70, 100), float)
pset.addEphemeralConstant("rsi_oversold_threshold", lambda: random.uniform(0, 30), float)

TREE_DEPTH = 5

#expr = gp.genHalfAndHalf(pset, min_=1, max_=5)
#tree = gp.PrimitiveTree(expr)

#old & working
#toolbox = base.Toolbox()

#toolbox.register("compile", gp.compile, pset=pset)

#func = toolbox.compile(expr=tree)
#print(func(1))
#print(str(tree))
#exit(0)
#evalSymbReg(tree)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=TREE_DEPTH)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", combined_mutation, min_=0, max_=TREE_DEPTH)
#toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutate", combined_mutation, expr=toolbox.expr_mut, pset=pset)


toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=TREE_DEPTH)) #17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=TREE_DEPTH))#17))


def go_doge():
    num_runs = 10000
    population_size = 1000
    num_generations = 200
    hof_size = 10
    mating_probs = [0.3, 0.5, 0.7, 0.8, 0.9, 1]
    mutation_probs = [0.3, 0.5, 0.7, 0.8, 0.9, 1]
    output_folder = "doggienauts"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    for run in range(num_runs):
        random.seed(run)
        start = time.time()

        for mating_prob in mating_probs:
            for mutation_prob in mutation_probs:
                pop = toolbox.population(n=population_size)
                hof = tools.HallOfFame(hof_size)

                stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
                stats_size = tools.Statistics(len)
                mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
                mstats.register("avg", np.mean)
                mstats.register("std", np.std)
                mstats.register("min", np.min)
                mstats.register("max", np.max)

                hof_name = "doge_{}_x-{}_m-{}-hof.p".format(run, mating_prob, mutation_prob)
                gen_best_name = "doge_{}_x-{}_m-{}-gen_best.p".format(run, mating_prob, mutation_prob)
                out_path_hof = os.path.join(output_folder, hof_name)
                out_path_gen_best = os.path.join(output_folder, gen_best_name)
                if os.path.exists(out_path_hof) and os.path.exists(out_path_gen_best):
                    print("{} already exists, skipping...".format(hof_name))
                    continue
                pop, log, best = eaSimpleCustom(pop, toolbox, mating_prob, mutation_prob, num_generations, stats=mstats,
                                               halloffame=hof, verbose=True)

                for i in range(len(best)):
                    print(i, best[i])
                for i in range(len(hof)):
                    print(hof[i])
                pickle.dump(hof, open(out_path_hof,"wb"))
                pickle.dump(best, open(out_path_gen_best, "wb"))
        end = time.time()
        print("Time for one run: {} minutes".format((end-start)/60))

    #hof = pickle.load(open(out_path,"rb"))
    #best = hof[0]
    #strat = GeneticTradingStrategy(best, data)
    #orders, _ = strat.get_orders(start_cash, start_crypto)
    #draw_price_chart(data.timestamps, data.prices, orders)





if __name__ == "__main__":
    random.seed(318)
    go_doge()
    exit(0)

    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(10)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 1, 0.3, 50, stats=mstats,
                                       halloffame=hof, verbose=True)
    for i in range(len(hof)):
        print(hof[i])

    best = hof[0]
    strat = GeneticTradingStrategy(best, data)
    orders, _ = strat.get_orders(start_cash, start_crypto)
    draw_price_chart(data.timestamps, data.prices, orders)



    data = validation_data
    strat = GeneticTradingStrategy(best, data)

    evaluation = strat.evaluate(start_cash, start_crypto, validation_start_time, validation_end_time, False, True)
    orders, _ = strat.get_orders(start_cash, start_crypto)
    draw_price_chart(data.timestamps, data.prices, orders)
    draw_tree(best)



