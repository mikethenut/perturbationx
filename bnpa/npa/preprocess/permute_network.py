import warnings

import numpy as np


def get_root(node, parents: dict):
    parent = parents[node]
    return parent if parent == node else get_root(parent, parents)


def find_connected_components(adj: np.ndarray):
    num_nodes = len(adj)
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

        node_x = rng.choice(component_x)
        node_y = rng.choice(component_y)

        weight = weights.pop()
        adj[node_x, node_y] = weight
        adj[node_y, node_x] = weight

        components[root_y].update(components[root_x])
        del components[root_x]


def permute_adjacency_k1(adj: np.ndarray, iterations=500, permutation_rate=1., ensure_connectedness=True, seed=None):
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Argument 'adj' {} is not a square matrix.".format(str(adj.shape)))

    rng = np.random.default_rng(seed)
    tril_idx = np.tril_indices(adj.shape[0], -1)
    adj_tril = adj[tril_idx]

    edge_idx = np.nonzero(adj_tril)[0]
    fixed_edges = np.floor(edge_idx.size * (1. - permutation_rate)).astype(int)
    edge_weights = [adj[idx, idy] for idx, idy in zip(*np.nonzero(adj)) if idx < idy]

    permuted = []
    for _ in range(iterations):
        fixed_idx = rng.choice(edge_idx, size=fixed_edges, replace=False)
        permuted_idx = np.ones(len(adj_tril), dtype=bool)
        permuted_idx[fixed_idx] = False
        random_tril = adj_tril.copy()
        random_tril[permuted_idx] = rng.permutation(random_tril[permuted_idx])

        random_adj = np.zeros(adj.shape)
        random_adj[tril_idx] = random_tril
        random_adj += random_adj.transpose()

        if ensure_connectedness:
            connect_adjacency_components(random_adj, weights=edge_weights, seed=seed)
        permuted.append(random_adj)

    return permuted


def permute_adjacency_k2(adj: np.ndarray, iterations=500, permutation_rate=1., ensure_connectedness=True, seed=None):
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Argument 'adj' {} is not a square matrix.".format(str(adj.shape)))

    rng = np.random.default_rng(seed)
    edge_stubs = np.array([[idx, idy] for idx, idy in zip(*np.nonzero(adj)) if idx < idy])

    total_edges = edge_stubs.shape[0]
    permuted_edges = np.ceil(total_edges * permutation_rate).astype(int)
    edge_weights = np.array([adj[idx, idy] for idx, idy in edge_stubs])

    permuted = []
    for _ in range(iterations):
        permuted_idx = rng.choice(total_edges, size=permuted_edges, replace=False)
        permuted_stubs = edge_stubs.copy()
        permuted_stubs[permuted_idx, :] = \
            rng.permutation(permuted_stubs[permuted_idx, :].flatten()).reshape(-1, 2)

        permuted_edge_weights = edge_weights.copy()
        permuted_edge_weights[permuted_idx] = rng.permutation(permuted_edge_weights[permuted_idx])
        permuted_edge_weights = list(permuted_edge_weights)

        random_adj = np.zeros(adj.shape)
        for (src, trg), weight in zip(permuted_stubs, permuted_edge_weights):
            if src != trg:
                random_adj[src, trg] += weight
                random_adj[trg, src] += weight

        if ensure_connectedness:
            connect_adjacency_components(random_adj, weights=edge_weights, seed=seed)
        permuted.append(random_adj)

    return permuted


def permute_adjacency(adj: np.ndarray, permutations=('k2',), iterations=500, permutation_rate=1., seed=None):
    adj_perms = dict()
    for p in set(permutations):
        match p.lower():
            case 'k1':
                adj_perms[p] = permute_adjacency_k1(
                    adj, iterations=iterations, permutation_rate=permutation_rate,
                    ensure_connectedness=True, seed=seed
                )
            case 'k2':
                adj_perms[p] = permute_adjacency_k2(
                    adj, iterations=iterations, permutation_rate=permutation_rate,
                    ensure_connectedness=True, seed=seed
                )
            case 'o':
                # Permutation 'o' is not applied to the laplacian.
                continue
            case _:
                warnings.warn("Permutation %s is unknown and will be skipped." % p)

    return adj_perms


def permute_edge_list(edge_list: np.ndarray, node_list, iterations, method='k1', permutation_rate=1., seed=None):
    if node_list is None:
        node_list = np.unique(edge_list[:, :2])

    permuted_edge_count = np.ceil(edge_list.shape[0] * permutation_rate).astype(int)
    rng = np.random.default_rng(seed)

    permutations = []
    for _ in range(iterations):
        permutation = []
        permutations.append(permutation)

        permuted_edge_idx = rng.choice(edge_list.shape[0], size=permuted_edge_count, replace=False, axis=0)
        permuted_edges = edge_list[permuted_edge_idx, :].copy()
        fixed_edges = np.delete(edge_list, permuted_edge_idx, axis=0)

        match method.lower():
            case 'k1':
                permuted_edges[:, 0] = rng.choice(node_list, size=permuted_edge_count, replace=True)
                permuted_edges[:, 1] = rng.choice(node_list, size=permuted_edge_count, replace=True)
            case 'k2':
                permuted_edges[:, 0] = rng.permutation(permuted_edges[:, 0])
                permuted_edges[:, 1] = rng.permutation(permuted_edges[:, 1])
            case _:
                raise ValueError("Unknown permutation %s." % permutation)

        permuted_edges[:, 2] = rng.permutation(permuted_edges[:, 2])
        permuted_edges = np.concatenate([permuted_edges, fixed_edges], axis=0)

        for i in range(permuted_edges.shape[0]):
            src, trg, rel = permuted_edges[i, :]
            if not any(e[0] == src and e[1] == trg for e in permutation):
                permutation.append((src, trg, rel))

        for i in range(edge_list.shape[0]):
            src, trg, rel = edge_list[i, :]
            if not any(e[0] == src and e[1] == trg for e in permutation):
                permutation.append((src, trg, None))

    return permutations
