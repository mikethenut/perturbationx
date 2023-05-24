import numpy as np
import networkx as nx


def generate_adjacency(graph: nx.DiGraph, directed=False):
    core_size = sum(1 for n in graph.nodes if graph.nodes[n]["type"] == "core")
    network_size = graph.number_of_nodes()
    adj = np.zeros((core_size, network_size))

    rows = [graph.nodes[src]["idx"] for src, trg in graph.edges]
    cols = [graph.nodes[trg]["idx"] for src, trg in graph.edges]
    data = [graph[src][trg]["weight"] for src, trg in graph.edges]
    adj[rows, cols] = data

    adj_c = adj[:core_size, :core_size]
    adj_b = adj[:core_size, core_size:]

    if not directed:
        adj_c = adj_c + adj_c.transpose()

    return adj_b, adj_c


def normalize_rows(adj_b: np.ndarray):
    if adj_b.ndim != 2:
        raise ValueError("Argument adjacency_boundary is not two-dimensional.")

    row_sums = np.abs(adj_b).sum(axis=1)
    row_sums[row_sums == 0] = 1
    return adj_b / row_sums[:, np.newaxis]


def generate_laplacians(adj_b, adj_c, adj_perms):
    lap_b = normalize_rows(adj_b)

    core_degrees = np.abs(adj_c).sum(axis=1)
    boundary_outdegrees = np.abs(lap_b).sum(axis=1)

    lap_c = np.diag(core_degrees + boundary_outdegrees) - adj_c
    lap_q = np.diag(core_degrees) + adj_c

    lap_perms = {}
    for p in adj_perms:
        lap_perms[p] = [adj + np.diag(np.abs(adj).sum(axis=1) + boundary_outdegrees) for adj in adj_perms[p]]

    return lap_b, lap_c, lap_q, lap_perms
