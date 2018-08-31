import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backtesting.orders import OrderType
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


def draw_chart_from_dataframe(df, data_column_name):
    timestamps = pd.to_datetime(df.index.values, unit='s')
    data = df.as_matrix(columns=[data_column_name])
    draw_price_chart(timestamps, data, None)


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

    if orders != None:
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


### IPython doge DNA visualizations

def create_jupyter_dna_container(container_name):
    from IPython.core.display import HTML, Javascript
    from string import Template
    html_template = Template("""
    <div id="$container">< /div>

    <style>

    .node {stroke: #fff; stroke-width: 1.5px;}
    .link {stroke: #999; stroke-opacity: .6;}

    < /style>
    """)

    html = html_template.substitute({"container": container_name})
    return html

def show_doge_dna(container_name, json_file):
    from string import Template
    from IPython.core.display import HTML, Javascript

    js_template = Template("""
    // We load the latest version of d3.js from the Web.
    require.config({paths: {d3: "https://d3js.org/d3.v3.min"}});


    require(["d3"], function(d3) {

        // Parameter declaration, the height and width of our viz.
        var width = 800,
            height = 600;

        // Colour scale for node colours.
        var color = d3.scale.category10();

        // We create a force-directed dynamic graph layout.
        // D3 has number of layouts - refer to the documentation.
        var force = d3.layout.force()
            .charge(-120)
            .linkDistance(30)
            .size([width, height]);

        // We select the < div> we created earlier and add an 
        // SVG = Scalable Vector Graphics
        var svg = d3.select("#$container").select("svg")
        if (svg.empty()) {
            svg = d3.select("#$container").append("svg")
                        .attr("width", width)
                        .attr("height", height);
        }

        // We load the JSON network file.
        d3.json("$json_file", function(error, graph) {
            // Within this block, the network has been loaded
            // and stored in the 'graph' object.

            // We load the nodes and links into the force-directed
            // graph and initialise the dynamics.
            force.nodes(graph.nodes)
                .links(graph.links)
                .start();

            // We create a < line> SVG element for each link
            // in the graph.
            var link = svg.selectAll(".link")
                .data(graph.links)
                .enter().append("line")
                .attr("class", "link");

            // We create a < circle> SVG element for each node
            // in the graph, and we specify a few attributes.
            var node = svg.selectAll(".node")
                .data(graph.nodes)
                .enter().append("circle")
                .attr("class", "node")
                .attr("r", function(d) {
                    // We colour the node depending on the degree.
                    if(d.id == 0) return 10; else return 5;
                })
                //.attr("r", 5)  // radius
                .style("fill", function(d) {
                    // We colour the node depending on the degree.
                    return color(d.name); 
                })
                .call(force.drag);

            // The label each node its node number from the networkx graph.
            node.append("title")
                .text(function(d) { return d.name; });

            //node.append("circle")
            //    .attr("r", 5)  // radius
            //    .style("fill", function(d) {
                    // We colour the node depending on the degree.
            //        return color(d.name); 
            //    });

            //node.append("text")
            //    .attr("dx", 12)
            //    .attr("dy", ".35em")
            //    .text(function(d) { return d.name });





            // We bind the positions of the SVG elements
            // to the positions of the dynamic force-directed graph,
            // at each time step.
            force.on("tick", function() {
                link.attr("x1", function(d) { return d.source.x; })
                    .attr("y1", function(d) { return d.source.y; })
                    .attr("x2", function(d) { return d.target.x; })
                    .attr("y2", function(d) { return d.target.y; });

                node.attr("cx", function(d) { return d.x; })
                    .attr("cy", function(d) { return d.y; });
            });
        });
    });
    """)
    js = js_template.substitute({"container": container_name, "json_file": json_file})
    #Javascript(js)
    return js


def write_graph_to_json(individual, json_file_name):
    from deap import gp
    import json

    nodes, edges, labels = gp.graph(individual)
    edited_nodes = [{"name": str(v), "id": int(k)} for (k, v) in labels.items()]
    edited_edges = [{"source": item[0], "target": item[1]} for item in edges]
    with open(json_file_name, 'w') as f:
        json.dump({'nodes': edited_nodes, 'links': edited_edges},
                  f, indent=4, )


def networkx_graph(individual):
    from deap import gp
    import matplotlib.pyplot as plt
    import networkx as nx
    from networkx.drawing.nx_agraph import graphviz_layout
    import pylab
    pylab.figure(1, figsize=(18, 12))

    nodes, edges, labels = gp.graph(individual)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()
