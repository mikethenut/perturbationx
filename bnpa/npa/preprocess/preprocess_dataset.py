import logging

import numpy as np
import pandas as pd
import networkx as nx

from bnpa.npa.preprocess import network_matrices


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


def remove_opposing_edges(adjacency: np.ndarray, dataset: pd.DataFrame):
    boundary_sign = np.sign(adjacency)
    dataset_sign = np.sign(dataset["logFC"].to_numpy())
    sign_mask = np.logical_and(
        boundary_sign != dataset_sign[np.newaxis, :],
        boundary_sign != 0
    )

    # remove edges with a different sign than the dataset
    adjacency_pruned = adjacency.copy()
    adjacency_pruned[sign_mask] = 0
    return adjacency_pruned


def prune_network_dataset(graph: nx.DiGraph, adj_b: np.ndarray, dataset: pd.DataFrame, dataset_id,
                          missing_value_pruning_mode="remove", opposing_value_pruning_mode="remove",
                          boundary_edge_minimum=6, verbose=True):

    if missing_value_pruning_mode not in ["remove", "nullify"]:
        raise ValueError("Invalid missing value pruning mode. Must be one of 'remove' or 'nullify'.")
    if opposing_value_pruning_mode is not None and opposing_value_pruning_mode not in ["remove", "nullify"]:
        raise ValueError("Invalid opposing value pruning mode. Must be either None or one of 'remove' or 'nullify'.")
    if boundary_edge_minimum < 0:
        raise ValueError("Boundary edge minimum must be non-negative.")
    if adj_b.ndim != 2:
        raise ValueError("Argument adjacency_boundary is not two-dimensional.")

    dataset_pruned = dataset[~dataset["logFC"].isna()]
    dataset_pruned = dataset_pruned[dataset_pruned["nodeID"].isin(graph.nodes)]

    core_size = adj_b.shape[0]
    network_idx = np.array([graph.nodes[node_name]["idx"] - core_size
                            for node_name in dataset_pruned["nodeID"].values
                            if node_name in graph.nodes])

    if network_idx.size == 0:
        raise ValueError("The dataset does not contain any boundary nodes.")

    lap_b = adj_b

    if missing_value_pruning_mode == "remove":
        lap_b = lap_b[:, network_idx]
    if opposing_value_pruning_mode == "remove":
        lap_b = remove_opposing_edges(lap_b, dataset_pruned)

    lap_b = network_matrices.generate_boundary_laplacian(lap_b, boundary_edge_minimum)

    if missing_value_pruning_mode == "nullify":
        lap_b = lap_b[:, network_idx]
    if opposing_value_pruning_mode == "nullify":
        lap_b = remove_opposing_edges(lap_b, dataset_pruned)

    lap_b_pruned = - lap_b

    # Infer dataset-specific metadata
    outer_boundary_node_count = network_idx.size
    # Count non-zero elements per row
    non_zero_row_count = np.count_nonzero(lap_b_pruned, axis=1)
    boundary_edge_count = np.sum(non_zero_row_count)
    inner_boundary_node_count = np.count_nonzero(non_zero_row_count)

    if verbose:
        logging.info("boundary nodes matched with dataset: %d" % outer_boundary_node_count)
        logging.info("boundary edges remaining: %d" % boundary_edge_count)
        logging.info("core nodes with boundary edges remaining: %d" % inner_boundary_node_count)

    graph.graph["dataset_" + dataset_id] = {
        "matched_outer_boundary_nodes": int(outer_boundary_node_count),
        "matched_boundary_edges": int(boundary_edge_count),
        "matched_inner_boundary_nodes": int(inner_boundary_node_count)
    }

    return lap_b_pruned, dataset_pruned
