from deap import gp


def recompute_tree_graph(nodes, edges):
	children = {node: [] for node in nodes}
	for a, b in edges:
		children[a].append(b)
	return children


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
        s += str(wrap_children(child))
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
        if cond_type is True:
            return children[1]
        if cond_type is False:
            return children[2]
        if str(children[1]) == str(children[2]):
            return children[1]
    elif node_type == "or_":
        if children[0][0] is True or children[1][0] is True:
            return [True, []]
        if children[0][0] is False:
            return children[1]
        if children[1][0] is False:
            return children[0]
    elif node_type == "and_":
        if children[0][0] is False or children[1][0] is False:
            return [False, []]
        if children[0][0] is True:
            return children[1]
        if children[1][0] is True:
            return children[0]
    elif node_type == "xor":
        if (children[0][0] is True and children[1][0] is False) or (children[0][0] is False and children[1][0] is True):
            return [True, []]
        if (children[0][0] is True and children[1][0] is True) or (children[0][0] is False and children[1][0] is False):
            return [False, []]
        if children[0][0] is False:
            return children[1]
        if children[1][0] is False:
            return children[0]
    elif node_type == "lt":
        try:
            child1 = float(children[0][0])
            child2 = float(children[1][0])
            return [child1 < child2, []]
        except:
            pass
    elif node_type == "gt":
        try:
            child1 = float(children[0][0])
            child2 = float(children[1][0])
            return [child1 > child2, []]
        except:
            pass


    elif node_type in ("identity_list", "identity_bool", "identity_float"):
        return children[0]

    # can't optimize this node
    # (and children are already optimized)
    return node