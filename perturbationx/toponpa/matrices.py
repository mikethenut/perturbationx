import warnings

import networkx as nx
import numpy as np
from scipy.sparse import issparse, lil_array, csr_array, SparseEfficiencyWarning, sparray

__all__ = ["generate_adjacencies", "generate_boundary_laplacian", "generate_core_laplacians"]


def generate_adjacencies(graph: nx.DiGraph, directed=False, sparse=True):
    """Generate the boundary and core adjacency matrices from a graph.

    :param graph: The graph.
    :type graph: nx.DiGraph
    :param directed: Whether to generate directed adjacency matrices. Defaults to False.
    :type directed: bool, optional
    :param sparse: Whether to generate sparse adjacency matrices. Defaults to True.
    :type sparse: bool, optional
    :return: The boundary and core adjacency matrices.
    :rtype: (np.ndarray, np.ndarray) | (sp.sparray, sp.sparray)
    """
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


def generate_boundary_laplacian(adj_b: np.ndarray | sparray, boundary_edge_minimum=6):
    """Generate the boundary Lb Laplacian from a boundary adjacency matrix.

    :param adj_b: The boundary adjacency matrix.
    :type adj_b: np.ndarray | sp.sparray
    :param boundary_edge_minimum: The minimum number of boundary edges a core node must have to be
                                    included in the Lb Laplacian. Nodes with fewer boundary edges
                                    are removed from the Lb Laplacian. Defaults to 6.
    :type boundary_edge_minimum: int, optional
    :raises ValueError: If the adjacency matrix is misshapen or if the boundary edge minimum is negative.
    :return: The boundary Lb Laplacian.
    :rtype: np.ndarray | sp.sparray
    """

    if adj_b.ndim != 2:
        raise ValueError("Argument adj_b is not two-dimensional.")
    if boundary_edge_minimum < 0:
        raise ValueError("Boundary edge minimum must be non-negative.")

    adj_b = adj_b.copy()

    if issparse(adj_b):
        # Clip edges where boundary outdegree is below threshold
        row_nonzero_counts = np.bincount(adj_b.nonzero()[0], minlength=adj_b.shape[0])
        clipping_mask = row_nonzero_counts < boundary_edge_minimum
        clipping_mask *= row_nonzero_counts > 0

        clip_row_idx = [idx for idx, clip_row in enumerate(clipping_mask) if clip_row] + [0]
        for idx in clip_row_idx:
            adj_b[idx, :] = 0

        row_sums = abs(adj_b).sum(axis=1)
        row_sums[row_sums == 0] = 1
        adj_b /= row_sums[:, np.newaxis]
        adj_b = csr_array(adj_b)

    else:
        # Clip edges where boundary outdegree is below threshold
        row_nonzero_counts = np.count_nonzero(adj_b, axis=1)
        adj_b[row_nonzero_counts < boundary_edge_minimum, :] = 0

        row_sums = np.abs(adj_b).sum(axis=1)
        row_sums[row_sums == 0] = 1
        adj_b /= row_sums[:, np.newaxis]

    return adj_b


def generate_core_laplacians(lap_b: np.ndarray | sparray, adj_c: np.ndarray | sparray, exact_boundary_outdegree=True):
    """Generate the core Laplacians from a boundary Laplacian and core adjacency matrix.

    :param lap_b: The boundary Laplacian.
    :type lap_b: np.ndarray | sp.sparray
    :param adj_c: The core adjacency matrix.
    :type adj_c: np.ndarray | sp.sparray
    :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
                                        is set to 1 for all core nodes with boundary edges. Defaults to True.
    :type exact_boundary_outdegree: bool, optional
    :return: The core Laplacians.
    :rtype: (np.ndarray, np.ndarray) | (sp.sparray, sp.sparray)
    """
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
