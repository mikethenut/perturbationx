import logging
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import issparse, sparray

from perturbationx.io import RelationTranslator
from .preprocess_network import infer_node_type, enumerate_nodes, remove_invalid_graph_elements, \
    infer_edge_attributes, infer_metadata
from ..matrices import generate_boundary_laplacian

__all__ = ["format_dataset", "prune_network_dataset", "infer_graph_attributes"]


def format_dataset(dataset: pd.DataFrame, computing_statistics=True):
    """Format a dataset for use with toponpa.

    :param dataset: The dataset to format. Must contain columns 'nodeID' and 'logFC'. If computing_statistics is True,
        the dataset must also contain a column 'stderr' or 't'.
    :type dataset: pd.DataFrame
    :param computing_statistics: Whether statistics will be computed from the dataset. Defaults to True.
    :type computing_statistics: bool, optional
    :raises ValueError: If the dataset is not a pandas.DataFrame, or if it does not contain columns 'nodeID' and
        'logFC', or if computing_statistics is True and the dataset does not contain a column 'stderr' or 't'.
    :return: The formatted dataset.
    :rtype: pd.DataFrame
    """
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Dataset is not a pandas.DataFrame.")
    if any(col not in dataset.columns for col in ["nodeID", "logFC"]):
        raise ValueError("Dataset does not contain columns 'nodeID' and 'logFC'.")

    if computing_statistics and "stderr" not in dataset.columns:
        if 't' in dataset.columns:
            dataset["stderr"] = np.divide(dataset["logFC"].to_numpy(), dataset['t'].to_numpy())
        else:
            raise ValueError("Dataset does not contain columns 'stderr' or 't'.")

    reduced_dataset = dataset[["nodeID", "logFC", "stderr"]] \
        if computing_statistics \
        else dataset[["nodeID", "logFC"]]

    return reduced_dataset


def remove_opposing_edges(adjacency: np.ndarray | sparray, dataset: pd.DataFrame, minimum_amplitude=1.):
    """Remove edges that causally oppose the values in the dataset.

    :param adjacency: The boundary adjacency matrix to prune.
    :type adjacency: np.ndarray | sp.sparray
    :param dataset: The dataset to use for pruning.
    :type dataset: pd.DataFrame
    :param minimum_amplitude: The minimum amplitude of the dataset values to consider. Values with an absolute value
        smaller than this threshold are ignored. Defaults to 1.
    :return: The pruned adjacency matrix.
    :rtype: np.ndarray
    """
    dataset_sign = np.sign(dataset["logFC"].to_numpy())
    dataset_mask = np.logical_and(
        np.abs(dataset["logFC"].to_numpy()) >= minimum_amplitude,
        dataset["logFC"].to_numpy() != 0.
    )

    # mask edges with a different sign than the dataset
    adjacency_mask = adjacency * dataset_sign[np.newaxis, :] < 0.
    adjacency_mask *= dataset_mask[np.newaxis, :]

    # remove edges with a different sign than the dataset
    adjacency_pruned = adjacency.copy()
    adjacency_pruned[adjacency_mask] = 0
    return adjacency_pruned


def prune_network_dataset(graph: nx.DiGraph, adj_b: np.ndarray | sparray, dataset: pd.DataFrame, dataset_id: str,
                          missing_value_pruning_mode="nullify", opposing_value_pruning_mode=None,
                          opposing_value_minimum_amplitude=1., boundary_edge_minimum=6, verbose=True):
    """Prune a network and dataset to match each other.

    :param graph: The network to prune.
    :type graph: nx.DiGraph
    :param adj_b: The boundary adjacency matrix to prune.
    :type adj_b: np.ndarray | sp.sparray
    :param dataset: The dataset to use for pruning.
    :type dataset: pd.DataFrame
    :param dataset_id: The name of the dataset.
    :type dataset_id: str
    :param missing_value_pruning_mode: The mode to use for pruning nodes with missing values. Must be one of 'remove'
        or 'nullify'. Defaults to 'nullify'.
    :type missing_value_pruning_mode: str, optional
    :param opposing_value_pruning_mode: The mode to use for pruning edges with opposing values. Must be one of 'remove',
        'nullify', or 'none'. Defaults to None.
    :type opposing_value_pruning_mode: str, optional
    :param opposing_value_minimum_amplitude: The minimum amplitude of the dataset values to consider. Values with an
        absolute value smaller than this threshold are ignored. Defaults to 1.
    :type opposing_value_minimum_amplitude: float, optional
    :param boundary_edge_minimum: The minimum number of boundary edges a core node must have to be included
        in the pruned network. If a core node has fewer boundary edges after 'remove' pruning, all of its edges are
        removed. This parameter is ignored if 'nullify' pruning is used. Defaults to 6.
    :type boundary_edge_minimum: int, optional
    :param verbose: Whether to log network statistics.
    :type verbose: bool, optional
    :raises ValueError: If the missing value pruning mode is invalid, or if the opposing value pruning mode is invalid,
        or if the boundary edge minimum is negative, or if the adjacency matrix is not two-dimensional,
        or if the dataset does not contain any boundary nodes.
    :return: The pruned boundary adjacency matrix and the pruned dataset.
    :rtype: tuple
    """
    if missing_value_pruning_mode not in ["remove", "nullify"]:
        raise ValueError("Invalid missing value pruning mode. Must be one of 'remove' or 'nullify'.")
    if opposing_value_pruning_mode is not None and opposing_value_pruning_mode not in ["remove", "nullify", "None"]:
        raise ValueError("Invalid opposing value pruning mode. Must be either None "
                         "or one of 'remove', 'nullify' or 'none.")
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
        lap_b = remove_opposing_edges(
            lap_b, dataset_pruned, minimum_amplitude=opposing_value_minimum_amplitude)

    lap_b = generate_boundary_laplacian(lap_b, boundary_edge_minimum)

    if missing_value_pruning_mode == "nullify":
        lap_b = lap_b[:, network_idx]
    if opposing_value_pruning_mode == "nullify":
        lap_b = remove_opposing_edges(
            lap_b, dataset_pruned, minimum_amplitude=opposing_value_minimum_amplitude)

    lap_b_pruned = - lap_b

    # Infer dataset-specific metadata
    outer_boundary_node_count = network_idx.size
    # Count non-zero elements per row
    if issparse(lap_b_pruned):
        non_zero_row_count = np.bincount(lap_b_pruned.nonzero()[0], minlength=lap_b_pruned.shape[0])
    else:
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


def infer_graph_attributes(graph: nx.DiGraph, relation_translator: Optional[RelationTranslator] = None, verbose=True):
    """Infer attributes of a network and add them to the graph instance.

    :param graph: The network to process.
    :type graph: nx.DiGraph
    :param relation_translator: The relation translator to use. If None, a new instance will be created.
    :type relation_translator: RelationTranslator, optional
    :param verbose: Whether to log network statistics.
    :type verbose: bool, optional
    :return: The processed network.
    :rtype: nx.DiGraph
    """
    # Quietly remove nodes without edges
    graph.remove_nodes_from(list(nx.isolates(graph)))

    # Partition core and boundary nodes
    boundary_nodes, core_nodes = infer_node_type(graph)
    if len(core_nodes) == 0:
        raise ValueError("The network does not contain any core nodes.")
    if len(boundary_nodes) == 0:
        raise ValueError("The network does not contain any boundary nodes.")

    # Compute node type and indices, add data to graph instance
    enumerate_nodes(graph, boundary_nodes, core_nodes)

    remove_invalid_graph_elements(graph)

    # Compute edge weight and interaction type
    infer_edge_attributes(graph, relation_translator)

    # Add stats to metadata
    infer_metadata(graph, verbose)

    return graph
