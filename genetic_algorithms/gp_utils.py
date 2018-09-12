

def recompute_tree_graph(nodes, edges):
	children = {node: [] for node in nodes}
	for a, b in edges:
		children[a].append(b)
	return children

