import warnings
import sys

import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse
import pandas as pd
import networkx as nx

from bnpa.importer.RelationTranslator import RelationTranslator
from typing import Optional


def enumerate_nodes(graph: nx.DiGraph):
    # Select nodes with outgoing edges and targets of core edges as core nodes,
    # select targets of boundary edges as boundary nodes
    core_nodes = {src for src, trg in graph.edges} | \
                 {trg for src, trg in graph.edges if graph[src][trg]["type"] == "core"}
    boundary_nodes = {trg for src, trg in graph.edges if graph[src][trg]["type"] == "boundary"}

    # Check that there isn't overlap between the two sets
    node_intersection = core_nodes & boundary_nodes
    if len(node_intersection) > 0:
        raise ValueError("The same nodes appear in network core and boundary: %s." % str(node_intersection))

    # Infer targets of unknown edges; their type defaults to boundary if they have no outgoing links
    for src, trg in graph.edges:
        if graph[src][trg]["type"] not in ("core", "boundary"):
            if trg in core_nodes:
                graph[src][trg]["type"] = "core"
            else:
                graph[src][trg]["type"] = "boundary"
                boundary_nodes.add(trg)

    # Compute indices and add data to graph instance
    core_size = len(core_nodes)
    node_idx = {node: idx for idx, node in enumerate(core_nodes)} | \
               {node: (core_size + idx) for idx, node in enumerate(boundary_nodes)}
    for n in graph.nodes:
        graph.nodes[n]["idx"] = node_idx[n]
        graph.nodes[n]["type"] = "core" if n in core_nodes else "boundary"

    return graph


def adjacency_matrix(graph: nx.DiGraph, relation_translator: Optional[RelationTranslator] = None):
    rt = relation_translator if relation_translator is not None else RelationTranslator()

    boundary_outdegree = {src: 0. for src, trg in graph.edges if graph.nodes[trg]["type"] == "boundary"}
    for src, trg in graph.edges:
        if graph.nodes[trg]["type"] == "boundary":
            boundary_outdegree[src] += abs(rt.translate(graph[src][trg]["relation"]))

    rows = [graph.nodes[src]["idx"] for src, trg in graph.edges]
    cols = [graph.nodes[trg]["idx"] for src, trg in graph.edges]
    data = [rt.translate(graph[src][trg]["relation"])
            if graph.nodes[trg]["type"] == "core" else
            rt.translate(graph[src][trg]["relation"]) / boundary_outdegree[src]
            for src, trg in graph.edges]
    return sparse.csr_matrix((data, (rows, cols)), shape=(graph.number_of_nodes(), graph.number_of_nodes()))


def laplacian_matrices(graph: nx.DiGraph, adjacency: sparse.spmatrix):
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Argument adjacency is not a square matrix.")
    if graph.number_of_nodes() != adjacency.shape[0]:
        raise ValueError("Argument graph has invalid number of "
                         "vertices (%d)." % graph.number_of_nodes())

    core_size = sum(1 for n in graph.nodes if graph.nodes[n]["type"] == "core")
    laplacian = - adjacency - adjacency.transpose()
    degree = abs(laplacian).sum(axis=1).A[:, 0]
    laplacian = sparse.diags(degree) + laplacian
    lap_c = laplacian[:core_size, :core_size].todense().A
    lap_b = laplacian[:core_size, core_size:].todense().A

    core_adjacency = adjacency[:core_size, :core_size]
    lap_q = core_adjacency + core_adjacency.transpose()
    core_degree = abs(lap_q).sum(axis=1).A[:, 0]
    lap_q = sparse.diags(core_degree) + lap_q
    lap_q = lap_q.todense().A

    return lap_c, lap_b, lap_q


def permute_laplacian_k(laplacian: np.ndarray, iterations=500, seed=None):
    if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
        print(laplacian.shape)
        raise ValueError("Argument laplacian is not a square matrix.")
    # WARNING: Some generated laplacians might be singular and are ignored,
    #          as the permutation does not ensure weakly chained diagonal dominance

    if seed is None:
        generator = np.random.default_rng()
    else:
        generator = np.random.default_rng(seed)

    network_size = laplacian.shape[0]
    tril_idx = np.tril_indices(network_size, -1)
    excess_degree = 2 * laplacian.diagonal() - np.sum(np.abs(laplacian), axis=0)
    permuted = []

    while len(permuted) < iterations:
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


def preprocess_network(graph, relation_translator, permutations='k', p_iters=500, seed=None):
    enumerate_nodes(graph)
    adj_mat = adjacency_matrix(graph, relation_translator)
    lap_c, lap_b, lap_q = laplacian_matrices(graph, adj_mat)
    lps = {'c': lap_c, 'b': lap_b, 'q': lap_q}

    np.set_printoptions(threshold=sys.maxsize)

    lperms = dict()
    for p in set(permutations):
        match p.lower():
            case 'k':
                lperms[p] = permute_laplacian_k(lap_c, p_iters, seed)
            case 'o':
                warnings.warn("Permutation 'o' is a boundary permutation "
                              "and is not applied to the laplacian." % p)
            case _:
                warnings.warn("Permutation %s is unknown and will be skipped." % p)

    return graph, lps, lperms


def reduce_to_common_nodes(lb: np.ndarray, graph: nx.DiGraph, dataset: pd.DataFrame):
    if lb.ndim != 2:
        raise ValueError("Argument lb is not two-dimensional.")
    elif any(col not in dataset.columns for col in ['nodeID', 'logFC', 't']):
        raise ValueError("Dataset does not contain columns 'nodeID', 'logFC' and 't'.")

    core_size = lb.shape[0]
    network_idx = np.array([graph.nodes[node_name]["idx"] - core_size
                            for node_name in dataset['nodeID'].values
                            if node_name in graph.nodes])
    lb_reduced = lb[:, network_idx]

    dataset_reduced = dataset[['nodeID', 'logFC', 't']]
    dataset_reduced = dataset_reduced[dataset['nodeID'].isin(graph.nodes)]

    return lb_reduced, dataset_reduced
