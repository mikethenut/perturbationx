import warnings

import numpy as np

from .adjacency_permutation import adjacency_permutation_k1, adjacency_permutation_k2

__all__ = ["permute_adjacency", "permute_edge_list"]


def permute_adjacency(adj: np.ndarray, permutations=('k2',), iterations=500, permutation_rate=1., seed=None):
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


def permute_edge_list(edge_list: np.ndarray, node_list, iterations, method='k1', permutation_rate=1., seed=None):
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
