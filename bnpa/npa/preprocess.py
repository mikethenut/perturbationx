import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse

from bnpa.importer.RelationTranslator import RelationTranslator
from typing import Optional


def enumerate_nodes(backbone_edges: dict, downstream_edges: dict):
    backbone_nodes = {n for nodes in backbone_edges.keys() for n in nodes} | \
                     {nodes[0] for nodes in downstream_edges.keys()}
    downstream_nodes = {nodes[1] for nodes in downstream_edges.keys()}
    node_intersection = backbone_nodes & downstream_nodes
    if len(node_intersection) > 0:
        raise ValueError("The same nodes appear in network backbone and downstream: %s." % str(node_intersection))

    backbone_size = len(backbone_nodes)
    node_idx = {node: idx for idx, node in enumerate(backbone_nodes)} | \
               {node: (backbone_size + idx) for idx, node in enumerate(downstream_nodes)}
    return node_idx, backbone_size


def adjacency_matrix(backbone_edges: dict, downstream_edges: dict, node_idx: dict,
                     relation_translator: Optional[RelationTranslator] = None):
    if relation_translator is None:
        relation_translator = RelationTranslator()

    downstream_degree = {src: 0. for src, trg in downstream_edges}
    for (src, trg), rel in downstream_edges.items():
        downstream_degree[src] += abs(relation_translator.translate(rel))

    rows = [node_idx[src] for src, trg in backbone_edges] + [node_idx[src] for src, trg in downstream_edges]
    cols = [node_idx[trg] for src, trg in backbone_edges] + [node_idx[trg] for src, trg in downstream_edges]
    data = [relation_translator.translate(rel) for rel in backbone_edges.values()] + \
           [relation_translator.translate(rel) / downstream_degree[src] for (src, trg), rel in downstream_edges.items()]
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(node_idx), len(node_idx)))


def laplacian_matrices(adjacency: sparse.spmatrix, backbone_size: int):
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Argument adjacency is not a square matrix.")
    if backbone_size < 0 or backbone_size >= adjacency.shape[0]:
        raise ValueError("Argument backbone_size is outside of interval [%d, %d]." % (0, adjacency.shape[0]))

    laplacian = - adjacency - adjacency.transpose()
    degree = abs(laplacian).sum(axis=1).A[:, 0]
    laplacian = sparse.diags(degree) + laplacian
    l3 = laplacian[:backbone_size, :backbone_size].todense().A
    l2 = laplacian[:backbone_size, backbone_size:].todense().A

    backbone_adjacency = adjacency[:backbone_size, :backbone_size]
    q = backbone_adjacency + backbone_adjacency.transpose()
    backbone_degree = abs(q).sum(axis=1).A[:, 0]
    q = sparse.diags(backbone_degree) + q
    q = q.todense().A

    return l3, l2, q


def permute_laplacian_k(laplacian: np.ndarray, permutations=500, seed=None):
    if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
        print(laplacian.shape)
        raise ValueError("Argument laplacian is not a square matrix.")
    # WARNING: Some generated laplacians might be singular

    if seed is None:
        generator = np.random.default_rng()
    else:
        generator = np.random.default_rng(seed)

    network_size = laplacian.shape[0]
    tril_idx = np.tril_indices(network_size, -1)
    excess_degree = np.sum(np.abs(laplacian), axis=0) - 2 * laplacian.diagonal()

    permuted = []
    for p in range(permutations):
        random_tril = generator.permutation(laplacian[tril_idx])
        random_laplacian = np.zeros((network_size, network_size))
        random_laplacian[tril_idx] = random_tril
        random_laplacian += random_laplacian.transpose()

        isolated_nodes = [idx for idx, deg in enumerate(np.sum(np.abs(random_laplacian), axis=0)) if deg == 0]
        trg_nodes = generator.integers(network_size - 1, size=len(isolated_nodes))
        for n, trg in zip(isolated_nodes, trg_nodes):
            if trg >= n:
                trg += 1
            random_laplacian[n, trg] = 1
            random_laplacian[trg, n] = 1

        np.fill_diagonal(random_laplacian, np.sum(np.abs(random_laplacian), axis=0) + excess_degree)

        try:
            la.inv(random_laplacian)
            permuted.append(random_laplacian)
        except la.LinAlgError:
            # Singular backbone generated
            continue

    return permuted


def preprocess_network(backbone_edges, downstream_edges, relation_translator):
    node_idx, bb_size = enumerate_nodes(backbone_edges, downstream_edges)
    adj_mat = adjacency_matrix(backbone_edges, downstream_edges,
                               node_idx, relation_translator)
    l3, l2, q = laplacian_matrices(adj_mat, bb_size)
    l3_permutations = permute_laplacian_k(l3)

    return node_idx, l3, l2, q, l3_permutations
