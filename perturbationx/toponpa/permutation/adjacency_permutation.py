import numpy as np
from scipy.sparse import issparse, lil_array

from perturbationx.util import connect_adjacency_components


def adjacency_permutation_k1(adj: np.ndarray, iterations=500, permutation_rate=1.,
                             ensure_connectedness=True, seed=None):
    """Permute the edges of an adjacency matrix using the "K1" method. This method permutes edges randomly.

    :param adj: The adjacency matrix to permute.
    :type adj: np.ndarray
    :param iterations: The number of permutations to generate. Defaults to 500.
    :type iterations: int, optional
    :param permutation_rate: The fraction of edges to permute. Defaults to 1.
    :type permutation_rate: float, optional
    :param ensure_connectedness: Whether to ensure that the permuted adjacency matrix is connected. Defaults to True.
    :type ensure_connectedness: bool, optional
    :param seed: The seed for the random number generator.
    :type seed: int, optional
    :raises ValueError: If the adjacency matrix is not square.
    :return: A list of permuted adjacency matrices.
    :rtype: list
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Argument 'adj' {} is not a square matrix.".format(str(adj.shape)))

    rng = np.random.default_rng(seed)
    tril_idx = np.tril_indices(adj.shape[0], -1)
    adj_tril = adj[tril_idx]
    if issparse(adj_tril):
        adj_tril = adj_tril.todense().flatten()

    edge_idx = np.nonzero(adj_tril)[0]
    edge_weights = adj_tril[edge_idx]
    fixed_edges = np.floor(edge_idx.size * (1. - permutation_rate)).astype(int)

    permuted = []
    for _ in range(iterations):
        fixed_idx = rng.choice(edge_idx, size=fixed_edges, replace=False)
        remaining_idx = [idx for idx in edge_idx if idx not in fixed_idx]
        fixed_weights = adj_tril[fixed_idx]
        remaining_weights = adj_tril[remaining_idx]

        permuted_idx = np.arange(adj_tril.size)
        permuted_idx = np.delete(permuted_idx, fixed_idx)
        permuted_idx = rng.choice(permuted_idx, size=remaining_weights.size, replace=False)

        combined_idx = np.concatenate((fixed_idx, permuted_idx))
        combined_weights = np.concatenate((fixed_weights, remaining_weights))

        if issparse(adj):
            random_adj = lil_array(adj.shape)
        else:
            random_adj = np.zeros(adj.shape)

        src_indices = tril_idx[0][combined_idx]
        trg_indices = tril_idx[1][combined_idx]
        random_adj[src_indices, trg_indices] = combined_weights
        random_adj[trg_indices, src_indices] += combined_weights

        if ensure_connectedness:
            connect_adjacency_components(random_adj, weights=edge_weights, seed=seed)
        permuted.append(random_adj)

    return permuted


def adjacency_permutation_k2(adj: np.ndarray, iterations=500, permutation_rate=1.,
                             ensure_connectedness=True, seed=None):
    """Permute the edges of an adjacency matrix using the "K2" method. This method permutes edges by preserving
    the degree of each node as much as possible.

    :param adj: The adjacency matrix to permute.
    :type adj: np.ndarray
    :param iterations: The number of permutations to generate. Defaults to 500.
    :type iterations: int, optional
    :param permutation_rate: The fraction of edges to permute. Defaults to 1.
    :type permutation_rate: float, optional
    :param ensure_connectedness: Whether to ensure that the permuted adjacency matrix is connected. Defaults to True.
    :type ensure_connectedness: bool, optional
    :param seed: The seed for the random number generator.
    :type seed: int, optional
    :raises ValueError: If the adjacency matrix is not square.
    :return: A list of permuted adjacency matrices.
    :rtype: list
    """
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
