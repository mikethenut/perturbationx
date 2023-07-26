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


def generate_core_laplacians(lap_b, adj_c, adj_perms, legacy=False):
    core_degrees = np.abs(adj_c).sum(axis=1)
    boundary_outdegrees = np.abs(lap_b).sum(axis=1)

    if legacy:  # Fix the boundary outdegrees to 1 if they are non-zero
        boundary_outdegrees[boundary_outdegrees > 0] = 1

    lap_c = np.diag(core_degrees + boundary_outdegrees) - adj_c
    lap_q = np.diag(core_degrees) + adj_c

    lap_perms = {}
    for p in adj_perms:
        lap_perms[p] = [np.diag(np.abs(adj).sum(axis=1) + boundary_outdegrees) - adj
                        for adj in adj_perms[p]]

    return lap_c, lap_q, lap_perms
