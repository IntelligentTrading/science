import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backtesting.orders import Order, OrderType
import pandas as pd
import networkx as nx
from deap import gp

def price(x):
    return '$%1.2f' % x

def plot_orders(ax, orders):
    for order in orders:
        if order.order_type == OrderType.BUY:
            color = "g"
        else:
            color = "r"
        timestamp = pd.to_datetime(order.timestamp, unit="s")
        price = order.unit_price
        circle = plt.Circle((timestamp, price), 0.02, color=color)
        ax.add_artist(circle)


def draw_price_chart(timestamps, prices, orders):

    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    weeks = mdates.WeekdayLocator()
    days = mdates.DayLocator() # every day
    daysFmt = mdates.DateFormatter('%m/%d')
    monthsFmt = mdates.DateFormatter('%m')

    fig, ax = plt.subplots()
    ax.plot(timestamps, prices)

    #circle1 = plt.Circle((timestamps[100], prices[100]), 0.02, color='r')
    #ax.add_artist(circle1)

    plot_orders(ax, orders)


    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_minor_locator(days)
    ax.xaxis.set_minor_formatter(daysFmt)
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)

    datemin = np.datetime64(timestamps[0])
    datemax = np.datetime64(timestamps[-1])
    ax.set_xlim(datemin, datemax)


    # format the coords message box
    ax.format_xdata = daysFmt
    ax.format_ydata = price
    ax.grid(False)

    plt.ylabel("Price", fontsize=14)



    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()

    plt.show()

def draw_tree(individual):
    nodes, edges, labels = gp.graph(individual)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    pos = nx.spring_layout(g)
    nx.draw(g, pos, node_size=1500, node_color='yellow', font_size=8, font_weight='bold')
    nx.draw_networkx_labels(g, pos, labels)

    plt.tight_layout()
    plt.show()