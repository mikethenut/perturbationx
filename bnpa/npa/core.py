import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse


def value_inference(lb: np.ndarray, lc: np.ndarray, boundary_coefficients: np.ndarray):
    if lb.ndim != 2:
        raise ValueError("Argument lb is not two-dimensional.")
    elif lc.ndim != 2 or lc.shape[0] != lc.shape[1]:
        raise ValueError("Argument lc is not a square matrix.")
    elif lc.shape[0] != lb.shape[0]:
        raise ValueError("Dimensions of lb and lc do not match.")
    elif lb.shape[1] != boundary_coefficients.shape[0]:
        raise ValueError("Dimensions of lb and boundary_coefficients do not match.")

    inference_matrix = - np.matmul(la.inv(lc), lb)
    return np.matmul(inference_matrix, boundary_coefficients)


def perturbation_amplitude(lq: np.ndarray, core_coefficients: np.ndarray, core_edge_count: int):
    if lq.ndim != 2 or lq.shape[0] != lq.shape[1]:
        raise ValueError("Argument lq is not a square matrix.")
    elif lq.shape[0] != core_coefficients.shape[0]:
        raise ValueError("Dimensions of lq and core_coefficients do not match.")

    total_amplitude = np.matmul(core_coefficients.transpose(), lq.dot(core_coefficients))
    if core_coefficients.ndim > 1 and core_coefficients.shape[1] > 1:
        total_amplitude = np.diag(total_amplitude)
    return total_amplitude / core_edge_count


def perturbation_amplitude_contributions(lq: sparse.spmatrix, core_coefficients: np.ndarray, core_edge_count: int):
    if lq.ndim != 2 or lq.shape[0] != lq.shape[1]:
        raise ValueError("Argument lq is not a square matrix.")
    elif lq.shape[0] != core_coefficients.shape[0]:
        raise ValueError("Dimensions of lq and core_coefficients do not match.")

    node_contributions = np.multiply(lq.dot(core_coefficients), core_coefficients)
    total_perturbation = node_contributions.sum(axis=0)

    relative_contributions = np.divide(node_contributions, total_perturbation)
    relative_contributions[relative_contributions == -0.] = 0.
    return total_perturbation / core_edge_count, relative_contributions.T
