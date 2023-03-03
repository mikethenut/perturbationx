import numpy as np
import scipy.sparse as sparse


def value_diffusion(diffusion_matrix: np.ndarray, fold_change: np.ndarray):
    if diffusion_matrix.ndim != 2:
        raise ValueError("Argument diffusion_matrix is not two-dimensional.")
    elif fold_change.ndim != 1:
        raise ValueError("Argument fold_change is not one-dimensional.")
    elif diffusion_matrix.shape[1] != fold_change.shape[0]:
        raise ValueError("Dimensions of diffusion_matrix and fold_change do not match.")

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

    node_contributions_rel = np.divide(node_contributions_abs, total_perturbation)
    return total_perturbation / backbone_edge_count, node_contributions_rel
