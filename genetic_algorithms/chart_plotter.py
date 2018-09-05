import matplotlib.pyplot as plt
from backtesting.orders import OrderType
import pandas as pd
from deap import gp
from string import Template
import networkx as nx




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
    html_template = Template("""
    <div id="$container">< /div>
        <style>
        .node {stroke: #fff; stroke-width: 1.5px;}
        .link {stroke: #999; stroke-opacity: .6;}
        < /style>
    """)

    return html_template.substitute({"container": container_name})


def show_doge_dna(container_name, json_file):

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
    return js_template.substitute({"container": container_name, "json_file": json_file})


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

class DogeDNACanvas:
    def __init__(self, individual, container_name):
        write_graph_to_json(individual, "tmp.json")
        self.container_name = container_name

    def create_container(self):
        return create_jupyter_dna_container(self.container_name)

    def show(self):
        return show_doge_dna(self.container_name, "tmp.json")
