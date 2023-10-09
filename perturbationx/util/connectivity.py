import numpy as np

__all__ = ["connect_adjacency_components"]


def get_root(node, parents: dict):
    """Get the root of a node in a tree represented by a dictionary of parent-child relationships.

    :param node: The node whose root to find.
    :param parents: A dictionary of parent-child relationships.
    :type parents: dict
    :return: The root of the node.
    """
    if parents[node] == node:
        return node

    parents[node] = get_root(parents[node], parents)
    return parents[node]


def find_connected_components(adj: np.ndarray):
    """Find the connected components of a graph represented by an adjacency matrix.

    :param adj: The adjacency matrix of the graph.
    :type adj: np.ndarray
    :raises ValueError: If the adjacency matrix is not square.
    :return: A dictionary of connected components, where the keys are the roots of the components and the values are
        the nodes in the components.
    :rtype: dict
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Argument 'adj' {} is not a square matrix.".format(str(adj.shape)))

    num_nodes = adj.shape[0]
    parents = {idx: idx for idx in range(num_nodes)}
    for idx, idy in zip(*np.nonzero(adj)):
        if idx < idy:
            root_x = get_root(idx, parents)
            root_y = get_root(idy, parents)

            if root_x != root_y:
                parents[root_y] = root_x

    components = {}
    for idx in range(num_nodes):
        root = get_root(idx, parents)
        if root not in components:
            components[root] = set()
        components[root].add(idx)

    return components


def connect_adjacency_components(adj: np.ndarray, nodes=None, weights=(1.,), seed=None):
    """Connect the components of a graph represented by an adjacency matrix.

    :param adj: The adjacency matrix of the graph.
    :type adj: np.ndarray
    :param nodes: The nodes to connect. If specified, edges will only be added between these nodes (when possible).
    :type nodes: list, optional
    :param weights: The weights to assign to the edges. Each edge will be assigned a random weight from this list.
    :type weights: list, optional
    :param seed: The seed for the random number generator.
    :type seed: int, optional
    :raises ValueError: If the adjacency matrix is not square.
    """
    if adj.ndim != 2:
        raise ValueError("Argument 'adj' {} is not a square matrix.".format(str(adj.shape)))

    components = find_connected_components(adj)

    rng = np.random.default_rng(seed)
    component_roots = rng.permutation(list(components.keys()))
    weights = list(rng.choice(weights, size=len(components) - 1, replace=True))

    for root_x, root_y in zip(component_roots[:-1], component_roots[1:]):
        if nodes is not None:
            component_x = components[root_x].intersection(nodes)
            if len(component_x) == 0:
                component_x = components[root_x]

            component_y = components[root_y].intersection(nodes)
            if len(component_y) == 0:
                component_y = components[root_y]
        else:
            component_x = components[root_x]
            component_y = components[root_y]

        node_x = rng.choice(list(component_x))
        node_y = rng.choice(list(component_y))

        weight = weights.pop()
        adj[node_x, node_y] = weight
        adj[node_y, node_x] = weight

        components[root_y].update(components[root_x])
        del components[root_x]
