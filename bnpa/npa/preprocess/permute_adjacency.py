import warnings

import numpy as np


def permute_adjacency_k(adj: np.ndarray, iterations=500, seed=None):
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        print(adj.shape)
        raise ValueError("Argument 'adj' is not a square matrix.")

    # WARNING: Some generated matrices might be singular,
    #          as this permutation does not ensure
    #          weakly chained diagonal dominance

    generator = np.random.default_rng(seed)
    network_size = adj.shape[0]
    tril_idx = np.tril_indices(network_size, -1)
    permuted = []

    while len(permuted) < iterations:
        random_tril = generator.permutation(adj[tril_idx])
        random_adj = np.zeros((network_size, network_size))
        random_adj[tril_idx] = random_tril
        random_adj += random_adj.transpose()

        isolated_nodes = [idx for idx, deg in enumerate(np.sum(np.abs(random_adj), axis=0)) if deg == 0]
        trg_nodes = generator.integers(network_size - 1, size=len(isolated_nodes))
        for n, trg in zip(isolated_nodes, trg_nodes):
            if trg >= n:
                trg += 1
            random_adj[n, trg] = 1
            random_adj[trg, n] = 1

        permuted.append(random_adj)

    return permuted


def permute_adjacency(adj: np.ndarray, permutations=('k',), p_iters=500, seed=None):
    adj_perms = dict()
    for p in set(permutations):
        match p.lower():
            case 'k':
                adj_perms[p] = permute_adjacency_k(adj, p_iters, seed)
            case 'o':
                # Permutation 'o' is not applied to the laplacian.
                continue
            case _:
                warnings.warn("Permutation %s is unknown and will be skipped." % p)

    return adj_perms
