import warnings
import itertools

import numpy as np


def permute_adjacency_k1(adj: np.ndarray, iterations=500, seed=None):
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Argument 'adj' {} is not a square matrix.".format(str(adj.shape)))

    # WARNING: Some generated matrices might be singular,
    #          as this permutation does not ensure
    #          a connected network adjacency matrix

    rng = np.random.default_rng(seed)
    network_size = adj.shape[0]
    tril_idx = np.tril_indices(network_size, -1)

    permuted = []
    while len(permuted) < iterations:
        random_tril = rng.permutation(adj[tril_idx])
        random_adj = np.zeros(adj.shape)
        random_adj[tril_idx] = random_tril
        random_adj += random_adj.transpose()

        isolated_nodes = [idx for idx, deg in enumerate(np.sum(np.abs(random_adj), axis=0)) if deg == 0]
        trg_nodes = rng.integers(network_size - 1, size=len(isolated_nodes))
        for n, trg in zip(isolated_nodes, trg_nodes):
            if trg >= n:
                trg += 1
            random_adj[n, trg] = 1
            random_adj[trg, n] = 1

        permuted.append(random_adj)

    return permuted


def permute_adjacency_k2(adj: np.ndarray, iterations=500, seed=None):
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Argument 'adj' {} is not a square matrix.".format(str(adj.shape)))

    # WARNING: Some generated matrices might be singular,
    #          as this permutation does not ensure
    #          a connected network adjacency matrix

    rng = np.random.default_rng(seed)
    edge_counts = np.count_nonzero(adj, axis=0)
    edge_stubs = list(itertools.chain.from_iterable([idx] * edge_counts[idx] for idx in range(adj.shape[0])))
    edge_weights = [adj[idx, idy] for idx in range(adj.shape[0]) for idy in range(idx) if adj[idx, idy] != 0]
    edge_count = len(edge_stubs) // 2

    permuted = []
    while len(permuted) < iterations:
        rng.shuffle(edge_stubs)
        permuted_edge_weights = list(rng.permutation(edge_weights))
        random_adj = np.zeros(adj.shape)

        for src, trg in zip(edge_stubs[:edge_count], edge_stubs[edge_count:]):
            edge_weight = permuted_edge_weights.pop()
            if src != trg:
                random_adj[src, trg] += edge_weight
                random_adj[trg, src] += edge_weight

        permuted.append(random_adj)

    return permuted


def permute_adjacency(adj: np.ndarray, permutations=('k',), p_iters=500, seed=None):
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
