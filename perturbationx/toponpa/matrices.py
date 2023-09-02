import warnings

import networkx as nx
import numpy as np
from scipy.sparse import issparse, lil_array, SparseEfficiencyWarning

__all__ = ["generate_adjacencies", "generate_boundary_laplacian", "generate_core_laplacians"]


def generate_adjacencies(graph: nx.DiGraph, directed=False, sparse=True):
    core_size = sum(1 for n in graph.nodes if graph.nodes[n]["type"] == "core")
    network_size = graph.number_of_nodes()

    rows = [graph.nodes[src]["idx"] for src, trg in graph.edges]
    cols = [graph.nodes[trg]["idx"] for src, trg in graph.edges]
    data = [graph[src][trg]["weight"] for src, trg in graph.edges]

    if sparse:
        adj = lil_array((core_size, network_size))
    else:
        adj = np.zeros((core_size, network_size))

    adj[rows, cols] = data
    adj_c = adj[:core_size, :core_size]
    adj_b = adj[:core_size, core_size:]

    if not directed:
        adj_c += adj_c.transpose()

    return adj_b, adj_c


def generate_boundary_laplacian(adj_b, boundary_edge_minimum=6):
    if adj_b.ndim != 2:
        raise ValueError("Argument adj_b is not two-dimensional.")
    if boundary_edge_minimum < 0:
        raise ValueError("Boundary edge minimum must be non-negative.")

    adj_b = adj_b.copy()

    if issparse(adj_b):
        # Clip edges where boundary outdegree is below threshold
        row_nonzero_counts = np.bincount(adj_b.nonzero()[0], minlength=adj_b.shape[0])
        adj_b[row_nonzero_counts < boundary_edge_minimum, :] = 0

        row_sums = abs(adj_b).sum(axis=1)
        row_sums[row_sums == 0] = 1
        adj_b /= row_sums[:, np.newaxis]
        adj_b = adj_b.tocsr()

    else:
        # Clip edges where boundary outdegree is below threshold
        row_nonzero_counts = np.count_nonzero(adj_b, axis=1)
        adj_b[row_nonzero_counts < boundary_edge_minimum, :] = 0

        row_sums = np.abs(adj_b).sum(axis=1)
        row_sums[row_sums == 0] = 1
        adj_b /= row_sums[:, np.newaxis]

    return adj_b


def generate_core_laplacians(lap_b, adj_c, exact_boundary_outdegree=True):
    core_degrees = np.abs(adj_c).sum(axis=1)
    boundary_outdegrees = np.abs(lap_b).sum(axis=1)

    if not exact_boundary_outdegree:
        # Fix the boundary outdegrees to 1 if they are non-zero
        boundary_outdegrees[boundary_outdegrees != 0] = 1

    if issparse(adj_c):
        lap_c = - adj_c.copy()
        lap_q = adj_c.copy()

        # lap_c is always converted to csr when inverting the sign (-)
        # this causes a SparseEfficiencyWarning when setting the diagonal
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
            lap_c.setdiag(core_degrees + boundary_outdegrees)
            lap_q.setdiag(core_degrees)
            lap_q = lap_q.tocsr()

    else:
        lap_c = np.diag(core_degrees + boundary_outdegrees) - adj_c
        lap_q = np.diag(core_degrees) + adj_c

    return lap_c, lap_q
