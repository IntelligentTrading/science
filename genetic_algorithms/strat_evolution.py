import random
from backtesting.strategies import Horizon, BuyAndHoldTimebasedStrategy
from chart_plotter import *
import pandas as pd
import os
import time
from gp_data import Data
import sys

SUPER_VERBOSE = False
start_cash = 1
start_crypto = 0

from genetic_program import GeneticProgram, GeneticTradingStrategy


def evaluate_buy_and_hold(data, history_size, verbose=True):
    start_bah = int(data.price_data.iloc[history_size].name)
    bah = BuyAndHoldTimebasedStrategy(start_bah, data.end_time, data.transaction_currency, data.counter_currency, data.source)
    evaluation = bah.evaluate(start_cash, start_crypto, start_bah, data.end_time, False, False)
    if verbose:
        print("Start time: {} \tEnd time: {}".format(
            pd.to_datetime(start_bah, unit='s'),
            pd.to_datetime(end_time, unit='s')))
        print("Buy and hold baseline: {0:0.2f}%".format(evaluation.get_profit_percent()))
    return evaluation


def go_doge(data, output_folder):
    num_runs = 5#10000
    population_size = 1000
    num_generations = 100
    hof_size = 10
    mating_probs = [0.3, 0.5, 0.7, 0.8, 0.9, 1]
    mutation_probs = [0.3, 0.5, 0.7, 0.8, 0.9, 1]
    mating_probs = [0.8, 0.9]
    mutation_probs = [0.8, 0.9]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    doge = GeneticProgram(data)
    for run in range(num_runs):
        random.seed(run)
        start = time.time()
        for mating_prob in mating_probs:
            for mutation_prob in mutation_probs:
                doge.evolve(mating_prob, mutation_prob, population_size, num_generations, verbose=True,
                            output_folder=output_folder, run_id=run)
        end = time.time()
        print("Time for one run: {} minutes".format((end-start)/60))


def evaluate_dogenauts_wow(doge_folder, evaluation_data):
    seen_individuals = []
    evaluate_buy_and_hold(evaluation_data, history_size)
    start_cash = 1
    start_crypto = 0

    genp = GeneticProgram(evaluation_data)

    for hof_filename in os.listdir(doge_folder):
        if not hof_filename.endswith("best.p"):
            continue
        try:
            run, mating_prob, mutation_prob = GeneticProgram.parse_evolution_filename(hof_filename)
        except:
            print("Error parsing filename {}".format(hof_filename))
            continue
        gen_best_filename = GeneticProgram.get_gen_best_filename(mating_prob, mutation_prob, run)
        print("Evaluating {}...".format(hof_filename))

        # load the individuals in the hall of fame

        hof = genp.load_evolution_file(os.path.join(doge_folder, hof_filename))

        for individual in hof:
            print(str(individual))
            try:
                #start_time = validation_start_time
                #end_time = validation_end_time
                strat = GeneticTradingStrategy(individual, evaluation_data, genp)
                orders, _ = strat.get_orders(start_cash, start_crypto)
                evaluation = strat.evaluate(start_cash, start_crypto, evaluation_data.start_time, evaluation_data.end_time, False, True)
                profit = evaluation.get_profit_percent()
                print("Profit: {0:0.02f}%".format(profit))
                if True: #profit > 0 and not profit in seen_individuals:
                    draw_price_chart(evaluation_data.timestamps, evaluation_data.prices, orders)
                    print(evaluation.get_report())
                    seen_individuals.append(profit)
            except:
                print("Error in evaluating individual")


if __name__ == "__main__":
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
    history_size = 200

    output_folder = sys.argv[1]

    training_data = Data(start_time, end_time, transaction_currency, counter_currency, resample_period, horizon,
                         start_cash,
                         start_crypto, source)


    validation_start_time = 1518825600
    validation_end_time = validation_start_time + 60 * 60 * 24 * 30 * 2
    validation_data = Data(validation_start_time, validation_end_time, "DOGE", counter_currency, resample_period,
                           horizon,
                           start_cash, start_crypto, source)

    #go_doge(training_data, output_folder)
    evaluate_dogenauts_wow(output_folder, validation_data)



