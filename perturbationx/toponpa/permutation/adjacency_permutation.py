import numpy as np
from scipy.sparse import issparse, lil_array

from perturbationx.util import connect_adjacency_components


def adjacency_permutation_k1(adj, iterations=500, permutation_rate=1.,
                             ensure_connectedness=True, seed=None):
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Argument 'adj' {} is not a square matrix.".format(str(adj.shape)))

    rng = np.random.default_rng(seed)
    tril_idx = np.tril_indices(adj.shape[0], -1)
    adj_tril = adj[tril_idx]
    if issparse(adj_tril):
        adj_tril = adj_tril.todense().flatten()

    edge_idx = np.nonzero(adj_tril)[0]
    fixed_edges = np.floor(edge_idx.size * (1. - permutation_rate)).astype(int)
    edge_weights = [adj[idx, idy] for idx, idy in zip(*np.nonzero(adj)) if idx < idy]

    permuted = []
    for _ in range(iterations):
        fixed_idx = rng.choice(edge_idx, size=fixed_edges, replace=False)
        permuted_idx = np.ones(adj_tril.shape[0], dtype=bool)
        permuted_idx[fixed_idx] = False
        random_tril = adj_tril.copy()
        random_tril[permuted_idx] = rng.permutation(random_tril[permuted_idx])

        if issparse(adj):
            random_adj = lil_array(adj.shape)
        else:
            random_adj = np.zeros(adj.shape)

        random_adj[tril_idx] = random_tril
        random_adj += random_adj.transpose()

        if ensure_connectedness:
            connect_adjacency_components(random_adj, weights=edge_weights, seed=seed)
        permuted.append(random_adj)

    return permuted


def adjacency_permutation_k2(adj: np.ndarray, iterations=500, permutation_rate=1.,
                             ensure_connectedness=True, seed=None):
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
        permuted_stubs[permuted_idx, 0] = rng.permutation(permuted_stubs[permuted_idx, 0])
        permuted_stubs[permuted_idx, 1] = rng.permutation(permuted_stubs[permuted_idx, 1])

        permuted_edge_weights = edge_weights.copy()
        permuted_edge_weights[permuted_idx] = rng.permutation(permuted_edge_weights[permuted_idx])

        if issparse(adj):
            random_adj = lil_array(adj.shape)
        else:
            random_adj = np.zeros(adj.shape)

        src_indices, trg_indices = zip(*permuted_stubs)
        mask = src_indices != trg_indices

        src_indices = np.array(src_indices)[mask]
        trg_indices = np.array(trg_indices)[mask]
        permuted_edge_weights = permuted_edge_weights[mask]

        random_adj[src_indices, trg_indices] = permuted_edge_weights
        random_adj[trg_indices, src_indices] += permuted_edge_weights

        if ensure_connectedness:
            connect_adjacency_components(random_adj, weights=edge_weights, seed=seed)

        permuted.append(random_adj)

    return permuted
