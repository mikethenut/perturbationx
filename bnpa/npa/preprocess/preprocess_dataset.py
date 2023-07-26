import logging

import numpy as np
import pandas as pd
import networkx as nx


def format_dataset(dataset: pd.DataFrame):
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Dataset is not a pandas.DataFrame.")
    if any(col not in dataset.columns for col in ["nodeID", "logFC"]):
        raise ValueError("Dataset does not contain columns 'nodeID' and 'logFC'.")

    if "stderr" not in dataset.columns:
        # TODO: Add more options for computing standard error
        if 't' in dataset.columns:
            dataset["stderr"] = np.divide(dataset["logFC"].to_numpy(), dataset['t'].to_numpy())
        else:
            raise ValueError("Dataset does not contain columns 'stderr' or 't'.")

    return dataset[["nodeID", "logFC", "stderr"]]


def normalize_rows(x: np.ndarray):
    if x.ndim != 2:
        raise ValueError("Argument x is not two-dimensional.")

    row_sums = np.abs(x).sum(axis=1)
    row_sums[row_sums == 0] = 1
    return x / row_sums[:, np.newaxis]


def prune_network_dataset(graph: nx.DiGraph, adj_b: np.ndarray, dataset: pd.DataFrame, dataset_id,
                          strict=False, legacy=False, verbose=True):
    if adj_b.ndim != 2:
        raise ValueError("Argument adjacency_boundary is not two-dimensional.")
    core_size = adj_b.shape[0]
    network_idx = np.array([graph.nodes[node_name]["idx"] - core_size
                            for node_name in dataset["nodeID"].values
                            if node_name in graph.nodes])

    if network_idx.ndim == 0:
        raise ValueError("The dataset does not contain any boundary nodes.")

    if strict:
        # TODO: implement strict pruning
        pass

    dataset_pruned = dataset[dataset["nodeID"].isin(graph.nodes)]
    if legacy:
        lap_b = - normalize_rows(adj_b)
        laplacian_boundary_pruned = lap_b[:, network_idx]
    else:
        adj_b_pruned = adj_b[:, network_idx]
        laplacian_boundary_pruned = - normalize_rows(adj_b_pruned)

    # Infer dataset-specific metadata
    outer_boundary_node_count = network_idx.size
    # Count non-zero elements per row
    non_zero_row_count = np.count_nonzero(laplacian_boundary_pruned, axis=1)
    boundary_edge_count = np.sum(non_zero_row_count)
    inner_boundary_node_count = np.count_nonzero(non_zero_row_count)

    if verbose:
        logging.info("boundary nodes matched with dataset: %d" % outer_boundary_node_count)
        logging.info("boundary edges remaining: %d" % boundary_edge_count)
        logging.info("core nodes with boundary edges remaining: %d" % inner_boundary_node_count)

    graph.graph[dataset_id] = {
        "matched_outer_boundary_nodes": int(outer_boundary_node_count),
        "matched_boundary_edges": int(boundary_edge_count),
        "matched_inner_boundary_nodes": int(inner_boundary_node_count)
    }

    return laplacian_boundary_pruned, dataset_pruned
