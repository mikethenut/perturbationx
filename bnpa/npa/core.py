import numpy as np
import scipy.sparse as sparse


def npa_value_diffusion(diffusion_matrix: np.ndarray, network_node_name: dict,
                        fold_change: np.ndarray, dataset_node_name: dict):

    if diffusion_matrix.ndim != 2:
        raise ValueError("Argument transformation_matrix is not of correct shape.")
    elif fold_change.ndim != 1:
        raise ValueError("Argument fold_change is not of correct shape.")

    backbone_node_count = diffusion_matrix.shape[0]
    dataset_node_idx = {v: k for k, v in dataset_node_name.items()}

    network_idx = np.array([node_idx - backbone_node_count for node_idx, node_name in sorted(network_node_name.items())
                            if node_name in dataset_node_idx])
    dataset_idx = np.array([dataset_node_idx[node_name] for node_idx, node_name in sorted(network_node_name.items())
                            if node_name in dataset_node_idx])

    transformation_matrix = diffusion_matrix[:, network_idx]
    fold_change = fold_change[dataset_idx, ]

    return - np.matmul(transformation_matrix, fold_change)


def network_perturbation_amplitude(backbone_edge_count: int, laplacian_backbone_signless: sparse.spmatrix,
                                   backbone_values: np.ndarray):

    temp = np.matmul(laplacian_backbone_signless.todense().A, backbone_values)
    res = np.matmul(backbone_values, temp)
    return res / backbone_edge_count
