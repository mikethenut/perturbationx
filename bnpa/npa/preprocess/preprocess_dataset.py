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


def prune_network_dataset(graph: nx.DiGraph, adj_b: np.ndarray, dataset: pd.DataFrame, dataset_id,
                          strict=False, verbose=True):
    if adj_b.ndim != 2:
        raise ValueError("Argument adjacency_boundary is not two-dimensional.")
    core_size = adj_b.shape[0]
    network_idx = np.array([graph.nodes[node_name]["idx"] - core_size
                            for node_name in dataset["nodeID"].values
                            if node_name in graph.nodes])

    if strict:
        # TODO: implement strict pruning
        pass

    if network_idx.ndim == 0:
        raise ValueError("The dataset does not contain any boundary nodes.")

    adjacency_boundary_pruned = adj_b[:, network_idx]
    dataset_pruned = dataset[dataset["nodeID"].isin(graph.nodes)]

    # Infer dataset-specific metadata
    outer_boundary_node_count = network_idx.size
    # Count non-zero elements per row
    non_zero_row_count = np.count_nonzero(adjacency_boundary_pruned, axis=1)
    boundary_edge_count = np.sum(non_zero_row_count)
    inner_boundary_node_count = np.count_nonzero(non_zero_row_count)

    if verbose:
        logging.info("boundary nodes matched with dataset: %d" % outer_boundary_node_count)
        logging.info("boundary edges remaining: %d" % boundary_edge_count)
        logging.info("core nodes with boundary edges remaining: %d" % inner_boundary_node_count)

    dataset_id_underscored = dataset_id.replace(" ", "_")
    graph.graph[dataset_id_underscored + "_outer_boundary_nodes"] = outer_boundary_node_count
    graph.graph[dataset_id_underscored + "_boundary_edges"] = boundary_edge_count
    graph.graph[dataset_id_underscored + "_inner_boundary_nodes"] = inner_boundary_node_count

    return adjacency_boundary_pruned, dataset_pruned
