import warnings

import numpy as np
from scipy.sparse import sparray

from .adjacency_permutation import adjacency_permutation_k1, adjacency_permutation_k2

__all__ = ["permute_adjacency", "permute_edge_list"]


def permute_adjacency(adj: np.ndarray | sparray, permutations=('k2',), iterations=500, permutation_rate=1., seed=None):
    """Permute an adjacency matrix.

    :param adj: The adjacency matrix to permute.
    :type adj: np.ndarray | sp.sparray
    :param permutations: The permutations to apply. May contain 'k1' and 'k2' in any order. Defaults to ('k2',).
    :type permutations: list, optional
    :param iterations: The number of permutations to generate. Defaults to 500.
    :type iterations: int, optional
    :param permutation_rate: The fraction of edges to permute. Defaults to 1.
    :type permutation_rate: float, optional
    :param seed: The seed for the random number generator.
    :type seed: int, optional
    :return: A dictionary of lists with permuted adjacency matrices, keyed by the permutation name.
    :rtype: dict
    """
    if type(permutations) is str:
        permutations = [permutations]

    adj_perms = dict()
    for p in set(permutations):
        match p.lower():
            case 'k1':
                adj_perms[p] = adjacency_permutation_k1(
                    adj, iterations=iterations, permutation_rate=permutation_rate,
                    ensure_connectedness=True, seed=seed
                )
            case 'k2':
                adj_perms[p] = adjacency_permutation_k2(
                    adj, iterations=iterations, permutation_rate=permutation_rate,
                    ensure_connectedness=True, seed=seed
                )
            case 'o':
                # Permutation 'o' is not applied to the laplacian.
                continue
            case _:
                warnings.warn("Permutation %s is unknown and will be skipped." % p)

    return adj_perms


def permute_edge_list(edge_list: np.ndarray, node_list=None, iterations=500,
                      method='k1', permutation_rate=1., seed=None):
    """Permute an edge list.

    :param edge_list: The edge list to permute. Must be a 2D array with shape (n_edges, 4). The first two columns
        contain the source and target nodes, the third column contains the edge type, and the fourth column contains
        the confidence weight. Confidence weights are optional.
    :type edge_list: np.ndarray
    :param node_list: The list of nodes to use in the permutation. Only edges that connect nodes in this list
        are permuted. If None, the list is inferred from the edge list.
    :type node_list: list, optional
    :param iterations: The number of permutations to generate. Defaults to 500.
    :type iterations: int, optional
    :param method: The permutation method to use. Defaults to 'k1'. May be 'k1' or 'k2'.
    :type method: str, optional
    :param permutation_rate: The fraction of edges to permute. Defaults to 1. If 'confidence', the confidence weights
        are used to determine the number of edges to permute. For each edge, a random number is drawn from a uniform
        distribution between 0 and 1. If the confidence weight is larger than this number, the edge is permuted.
    :type permutation_rate: float | str, optional
    :param seed: The seed for the random number generator.
    :type seed: int, optional
    :raises ValueError: If the permutation method is unknown.
    :return: A list of permutations. Each permutation is a list of tuples with the source node, target node, and edge
        type. If the edge type is None, the edge is removed.
    """
    if node_list is None:
        node_list = np.unique(edge_list[:, :2])

    if permutation_rate != "confidence":
        permutation_rate = float(permutation_rate)

    permuted_edge_count = np.ceil(edge_list.shape[0] * permutation_rate).astype(int) \
        if type(permutation_rate) is float else None
    confidence_weights = edge_list[:, 3].astype(float)
    rng = np.random.default_rng(seed)

    permutations = []
    for _ in range(iterations):
        permutation = []
        permutations.append(permutation)

        if permutation_rate == "confidence":
            permuted_edge_idx = np.where(rng.uniform(size=edge_list.shape[0]) > confidence_weights)[0]
            permuted_edge_count = permuted_edge_idx.shape[0]
        else:
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
        for i in range(permuted_edges.shape[0]):
            src, trg, rel = permuted_edges[i, :3]
            if src != trg and not any(e[0] == src and e[1] == trg for e in permutation):
                permutation.append((src, trg, rel))

        permuted_edges = np.concatenate([permuted_edges, fixed_edges], axis=0)
        for i in range(edge_list.shape[0]):
            src, trg = edge_list[i, :2]
            if not any(permuted_edges[j, 0] == src and permuted_edges[j, 1] == trg
                       for j in range(permuted_edges.shape[0])):
                permutation.append((src, trg, None))

        if len(permutation) == 0:
            warnings.warn("Edge list permutation '%s' produced empty modification list." % method)

    return permutations
