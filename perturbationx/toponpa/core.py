import numpy as np
import numpy.linalg as la
from scipy.sparse import issparse, sparray

__all__ = ["coefficient_inference", "perturbation_amplitude", "perturbation_amplitude_contributions"]


def coefficient_inference(lap_b: np.ndarray | sparray, lap_c: np.ndarray | sparray, boundary_coefficients: np.ndarray):
    """Infer core coefficients from boundary coefficients and Laplacian matrices.

    :param lap_b: The Lb boundary Laplacian.
    :type lap_b: np.ndarray | sp.sparray
    :param lap_c: The Lc core Laplacian.
    :type lap_c: np.ndarray | sp.sparray
    :param boundary_coefficients: The boundary coefficients.
    :type boundary_coefficients: np.ndarray
    :raises ValueError: If the Laplacian matrices are misshapen or if the matrix dimensions do not match.
    :return: The inferred core coefficients.
    :rtype: np.ndarray
    """
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
    return la.solve(lap_c, edge_constraints)


def perturbation_amplitude(lap_q: np.ndarray | sparray, core_coefficients: np.ndarray, core_edge_count: int):
    """Compute the perturbation amplitude from the core Laplacian and core coefficients.

    :param lap_q: The Q core Laplacian.
    :type lap_q: np.ndarray | sp.sparray
    :param core_coefficients: The core coefficients.
    :type core_coefficients: np.ndarray
    :param core_edge_count: The number of edges in the core network.
    :type core_edge_count: int
    :raises ValueError: If the Laplacian matrix is misshapen or if the matrix dimensions do not match.
    :return: The perturbation amplitude.
    :rtype: np.ndarray
    """
    if lap_q.ndim != 2 or lap_q.shape[0] != lap_q.shape[1]:
        raise ValueError("Argument lap_q is not a square matrix.")
    elif lap_q.shape[0] != core_coefficients.shape[0]:
        raise ValueError("Dimensions of lap_q and core_coefficients do not match.")

    total_amplitude = core_coefficients.T @ lap_q @ core_coefficients
    if core_coefficients.ndim > 1 and core_coefficients.shape[1] > 1:
        total_amplitude = np.diag(total_amplitude)
    return total_amplitude / core_edge_count


def perturbation_amplitude_contributions(lap_q: np.ndarray | sparray, core_coefficients: np.ndarray,
                                         core_edge_count: int):
    """Compute the perturbation amplitude and relative contributions from the core Laplacian and core coefficients.

    :param lap_q: The Q core Laplacian.
    :type lap_q: np.ndarray | sp.sparray
    :param core_coefficients: The core coefficients.
    :type core_coefficients: np.ndarray
    :param core_edge_count: The number of edges in the core network.
    :type core_edge_count: int
    :raises ValueError: If the Laplacian matrix is misshapen or if the matrix dimensions do not match.
    :return: The perturbation amplitude and relative contributions.
    :rtype: (np.ndarray, np.ndarray)
    """
    if lap_q.ndim != 2 or lap_q.shape[0] != lap_q.shape[1]:
        raise ValueError("Argument lap_q is not a square matrix.")
    elif lap_q.shape[0] != core_coefficients.shape[0]:
        raise ValueError("Dimensions of lap_q and core_coefficients do not match.")

    node_contributions = np.multiply(lap_q @ core_coefficients, core_coefficients)
    total_perturbation = node_contributions.sum(axis=0)

    relative_contributions = np.divide(node_contributions, total_perturbation)
    relative_contributions[relative_contributions == -0.] = 0.
    return total_perturbation / core_edge_count, relative_contributions.T
