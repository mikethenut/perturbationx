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


def prune_network_dataset(graph: nx.DiGraph, adj_b: np.ndarray, dataset: pd.DataFrame, strict=False, verbose=True):
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

    if verbose:
        # TODO: add metadata to info
        logging.info("boundary nodes matched with dataset: %d" % network_idx.size)

    adjacency_boundary_pruned = adj_b[:, network_idx]
    dataset_pruned = dataset[dataset["nodeID"].isin(graph.nodes)]
    return adjacency_boundary_pruned, dataset_pruned
