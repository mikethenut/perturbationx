import numpy as np

__all__ = ["connect_adjacency_components"]


def get_root(node, parents: dict):
    parent = parents[node]
    return parent if parent == node else get_root(parent, parents)


def find_connected_components(adj: np.ndarray):
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
