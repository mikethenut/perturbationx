import time
import warnings
import itertools

import numpy as np


def get_root(node, parents: dict):
    parent = parents[node]
    return parent if parent == node else get_root(parent, parents)


def connect_adjacency_components(adj: np.ndarray, weights=(1.,), seed=None):
    # Find connected components
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
            components[root] = []
        components[root].append(idx)

    # Randomly connect components
    rng = np.random.default_rng(seed)
    component_roots = rng.permutation(list(components.keys()))
    weights = list(rng.choice(weights, size=len(components) - 1, replace=True))
    for root_x, root_y in zip(component_roots[:-1], component_roots[1:]):
        node_x = rng.choice(components[root_x])
        node_y = rng.choice(components[root_y])

        weight = weights.pop()
        adj[node_x, node_y] = weight
        adj[node_y, node_x] = weight

        components[root_y].extend(components[root_x])
        del components[root_x]


def permute_adjacency_k1(adj: np.ndarray, iterations=500, seed=None):
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Argument 'adj' {} is not a square matrix.".format(str(adj.shape)))

    rng = np.random.default_rng(seed)
    network_size = adj.shape[0]
    tril_idx = np.tril_indices(network_size, -1)
    adj_tril = adj[tril_idx]
    edge_weights = [w for w in adj_tril if w != 0]

    permuted = []
    permuting_time = 0.
    connecting_time = 0.
    while len(permuted) < iterations:
        start_time = time.process_time_ns()
        random_tril = rng.permutation(adj_tril)
        random_adj = np.zeros(adj.shape)
        random_adj[tril_idx] = random_tril
        random_adj += random_adj.transpose()
        permuting_time += time.process_time_ns() - start_time

        start_time = time.process_time_ns()
        connect_adjacency_components(random_adj, edge_weights, seed)
        connecting_time += time.process_time_ns() - start_time
        permuted.append(random_adj)

    print("Permuting time: %.2f s" % (permuting_time / 1e9))
    print("Connecting time: %.2f s" % (connecting_time / 1e9))

    return permuted


def permute_adjacency_k2(adj: np.ndarray, iterations=500, seed=None):
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Argument 'adj' {} is not a square matrix.".format(str(adj.shape)))

    rng = np.random.default_rng(seed)
    edge_counts = np.count_nonzero(adj, axis=0)
    edge_stubs = list(itertools.chain.from_iterable([idx] * edge_counts[idx] for idx in range(adj.shape[0])))
    edge_weights = [adj[idx, idy] for idx in range(adj.shape[0]) for idy in range(idx) if adj[idx, idy] != 0]
    edge_count = len(edge_stubs) // 2

    permuted = []
    permuting_time = 0.
    connecting_time = 0.
    while len(permuted) < iterations:
        start_time = time.process_time_ns()
        rng.shuffle(edge_stubs)
        permuted_edge_weights = list(rng.permutation(edge_weights))
        random_adj = np.zeros(adj.shape)

        for src, trg in zip(edge_stubs[:edge_count], edge_stubs[edge_count:]):
            edge_weight = permuted_edge_weights.pop()
            if src != trg:
                random_adj[src, trg] += edge_weight
                random_adj[trg, src] += edge_weight
        permuting_time += time.process_time_ns() - start_time

        start_time = time.process_time_ns()
        connect_adjacency_components(random_adj, edge_weights, seed)
        connecting_time += time.process_time_ns() - start_time
        permuted.append(random_adj)

    print("Permuting time: %.2f s" % (permuting_time / 1e9))
    print("Connecting time: %.2f s" % (connecting_time / 1e9))

    return permuted


def permute_adjacency(adj: np.ndarray, permutations=('k2',), p_iters=500, seed=None):
    adj_perms = dict()
    for p in set(permutations):
        match p.lower():
            case 'k1':
                adj_perms[p] = permute_adjacency_k1(adj, p_iters, seed)
            case 'k2':
                adj_perms[p] = permute_adjacency_k2(adj, p_iters, seed)
            case 'o':
                # Permutation 'o' is not applied to the laplacian.
                continue
            case _:
                warnings.warn("Permutation %s is unknown and will be skipped." % p)

    return adj_perms
