import numpy as np
import numpy.linalg as la
from scipy.sparse import issparse

__all__ = ["coefficient_inference", "perturbation_amplitude", "perturbation_amplitude_contributions"]


def coefficient_inference(lap_b, lap_c, boundary_coefficients: np.ndarray):
    if lap_b.ndim != 2:
        raise ValueError("Argument lap_b is not two-dimensional.")
    elif lap_c.ndim != 2 or lap_c.shape[0] != lap_c.shape[1]:
        raise ValueError("Argument lap_c not a square matrix.")
    elif lap_c.shape[0] != lap_b.shape[0]:
        raise ValueError("Dimensions of lap_b and lap_c do not match.")
    elif lap_b.shape[1] != boundary_coefficients.shape[0]:
        raise ValueError("Dimensions of lap_b and boundary_coefficients do not match.")

    if issparse(lap_c):
        lap_c = lap_c.todense()

    edge_constraints = - lap_b @ boundary_coefficients
    return la.inv(lap_c) @ edge_constraints


def perturbation_amplitude(lap_q: np.ndarray, core_coefficients: np.ndarray, core_edge_count: int):
    if lap_q.ndim != 2 or lap_q.shape[0] != lap_q.shape[1]:
        raise ValueError("Argument lap_q is not a square matrix.")
    elif lap_q.shape[0] != core_coefficients.shape[0]:
        raise ValueError("Dimensions of lap_q and core_coefficients do not match.")

    total_amplitude = core_coefficients.T @ lap_q @ core_coefficients
    if core_coefficients.ndim > 1 and core_coefficients.shape[1] > 1:
        total_amplitude = np.diag(total_amplitude)
    return total_amplitude / core_edge_count


def perturbation_amplitude_contributions(lap_q: np.ndarray, core_coefficients: np.ndarray, core_edge_count: int):
    if lap_q.ndim != 2 or lap_q.shape[0] != lap_q.shape[1]:
        raise ValueError("Argument lap_q is not a square matrix.")
    elif lap_q.shape[0] != core_coefficients.shape[0]:
        raise ValueError("Dimensions of lap_q and core_coefficients do not match.")

    node_contributions = np.multiply(lap_q @ core_coefficients, core_coefficients)
    total_perturbation = node_contributions.sum(axis=0)

    relative_contributions = np.divide(node_contributions, total_perturbation)
    relative_contributions[relative_contributions == -0.] = 0.
    return total_perturbation / core_edge_count, relative_contributions.T
