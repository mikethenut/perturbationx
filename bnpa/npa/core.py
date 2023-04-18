import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse

import pandas as pd


def reduce_to_common_nodes(l2: np.ndarray, node_idx: dict, dataset: pd.DataFrame):
    if l2.ndim != 2:
        raise ValueError("Argument l2 is not two-dimensional.")
    elif any(col not in dataset.columns for col in ['nodeID', 'logFC', 't']):
        raise ValueError("Dataset does not contain columns 'nodeID', 'logFC' and 't'.")

    backbone_size = l2.shape[0]
    network_idx = np.array([node_idx[node_name] - backbone_size for node_name in dataset['nodeID'].values
                           if node_name in node_idx])
    l2_reduced = l2[:, network_idx]

    dataset_reduced = dataset[['nodeID', 'logFC', 't']]
    dataset_reduced = dataset_reduced[dataset['nodeID'].isin(node_idx)]

    return l2_reduced, dataset_reduced


def value_diffusion(l3: np.ndarray, l2: np.ndarray, fold_change: np.ndarray):
    if l2.ndim != 2:
        raise ValueError("Argument l2 is not two-dimensional.")
    elif l3.ndim != 2 or l3.shape[0] != l3.shape[1]:
        raise ValueError("Argument l3 is not a square matrix.")
    elif fold_change.ndim != 1:
        raise ValueError("Argument fold_change is not one-dimensional.")
    elif l3.shape[0] != l2.shape[0]:
        raise ValueError("Dimensions of l2 and l3 do not match.")
    elif l2.shape[1] != fold_change.shape[0]:
        raise ValueError("Dimensions of l2 and fold_change do not match.")

    diffusion_matrix = np.matmul(la.inv(l3), l2)
    return - np.matmul(diffusion_matrix, fold_change)


def perturbation_amplitude(q: np.ndarray, backbone_values: np.ndarray, backbone_edge_count: int):
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("Argument q is not a square matrix.")
    elif backbone_values.ndim != 1:
        raise ValueError("Argument backbone_values is not one-dimensional.")
    elif q.shape[0] != backbone_values.shape[0]:
        raise ValueError("Dimensions of q and backbone_values do not match.")

    return np.matmul(q.dot(backbone_values), backbone_values) / backbone_edge_count


def perturbation_amplitude_contributions(q: sparse.spmatrix, backbone_values: np.ndarray, backbone_edge_count: int):
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("Argument q is not a square matrix.")
    elif backbone_values.ndim != 1:
        raise ValueError("Argument backbone_values is not one-dimensional.")
    elif q.shape[0] != backbone_values.shape[0]:
        raise ValueError("Dimensions of q and backbone_values do not match.")

    temp = q.dot(backbone_values)
    node_contributions_abs = np.multiply(temp, backbone_values)
    total_perturbation = np.sum(node_contributions_abs)

    node_contributions_rel = np.abs(np.divide(node_contributions_abs, total_perturbation))
    node_contributions_rel[node_contributions_rel == -0.] = 0.
    return total_perturbation / backbone_edge_count, node_contributions_rel
