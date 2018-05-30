
from deap import base
from deap import gp
from deap import algorithms
from deap import creator
from deap import tools

import operator
import random
import types
import talib
import numpy as np
from backtesting.signals import Signal
from backtesting.strategies import Strategy, Horizon, Strength
from data_sources import get_resampled_prices_in_range


transaction_currency = "ETH"
counter_currency = "BTC"
end_time = 1526637600
start_time = end_time - 60 * 60 * 24 * 30
horizon = Horizon.short
resample_period = 60
start_cash = 1
start_crypto = 0

price_data = get_resampled_prices_in_range(start_time, end_time, transaction_currency, counter_currency, resample_period)
#price_data = np.random.random(1000)
rsi_data = talib.RSI(np.array(price_data.close_price, dtype=float), timeperiod=14)
sma_data = talib.SMA(np.array(price_data.close_price, dtype=float), timeperiod=50)

class GeneticTradingStrategy(Strategy):
    def __init__(self, tree):
        self.horizon = horizon
        self.transaction_currency = transaction_currency
        self.counter_currency = counter_currency
        self.strength = Strength.any
        self.source = 0
        self.start_time = start_time
        self.end_time = end_time
        self.tree = tree
        self.build_from_gp_tree(tree)

    def belongs_to_this_strategy(self, signal):
        return signal.signal_type == "Genetic"

    def build_from_gp_tree(self, tree):
        self.signals = []
        func = toolbox.compile(expr=tree)
        history_size = 100
        outcomes = []
        for i, row in enumerate(price_data.itertuples()):
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
        return output1  # TODO fix this


def rsi(input):
    timestamp = input[0]
    timestamp_index = np.where(price_data.index == timestamp)[0]
    return rsi_data[timestamp_index]


def sma(input):
    timestamp = input[0]
    timestamp_index = np.where(price_data.index == timestamp)[0]
    return sma_data[timestamp_index]


def price(input):
    timestamp = input[0]
    return price_data.loc[timestamp,"close_price"]


def buy():
    pass


def sell():
    pass


def ignore():
    pass


def identity(x):
    return x


def evaluate_individual(individual):
    strategy = GeneticTradingStrategy(individual)
    evaluation = strategy.evaluate(start_cash, start_crypto, start_time, end_time, False, False)
    if evaluation.get_num_trades() > 1:
        print(str(individual))
        print(evaluation.get_report())

    return evaluation.get_profit_percent(),


def combined_mutation(individual, expr, pset):
    if random.random() > 0:
        #print("Insert")
        return gp.mutInsert(individual, pset)
    else:
        #print("Ephemeral")
        return gp.mutEphemeral(individual, "one")


pset = gp.PrimitiveSetTyped("main", [list], types.FunctionType)
pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
#pset.addPrimitive(operator.or_, [bool, bool], bool)
#pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(if_then_else, [bool, types.FunctionType, types.FunctionType], types.FunctionType)
pset.addPrimitive(rsi, [list], float)
pset.addPrimitive(sma, [list], float)
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

TREE_DEPTH = 7

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


def main():
    random.seed(318)

    pop = toolbox.population(n=50)#0)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 1, 0.3, 80, stats=mstats,
                                       halloffame=hof, verbose=True)
    # print log
    print(hof[0])

    return pop, log, hof


if __name__ == "__main__":
    main()