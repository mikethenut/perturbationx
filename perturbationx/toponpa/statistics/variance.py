import numpy as np
import numpy.linalg as la
from scipy.sparse import issparse, lil_array, sparray


def core_covariance_matrix(lap_b: np.ndarray | sparray, lap_c: np.ndarray | sparray, stderr: np.ndarray):
    """Compute the covariance matrix of the core node coefficients.

    :param lap_b: The Lb boundary Laplacian.
    :type lap_b: np.ndarray | sp.sparray
    :param lap_c: The Lc core Laplacian.
    :type lap_c: np.ndarray | sp.sparray
    :param stderr: The standard error of the boundary coefficients.
    :type stderr: np.ndarray
    :return: The covariance matrix of the core node coefficients.
    :rtype: np.ndarray
    """
    if issparse(lap_c):
        lap_c = lap_c.todense()
    lap_c_inv = la.inv(lap_c)

    err_diag = lil_array((stderr.shape[0], stderr.shape[0]))
    err_diag.setdiag(stderr ** 2)
    tmp = lap_b @ err_diag @ lap_b.T
    res = lap_c_inv @ tmp @ lap_c_inv.T
    return res


def perturbation_variance(lap_q: np.ndarray | sparray, core_coefficients: np.ndarray,
                          core_covariance: np.ndarray, core_edge_count: int):
    """Compute the variance of the perturbation score.

    :param lap_q: The Q core Laplacian.
    :type lap_q: np.ndarray | sp.sparray
    :param core_coefficients: The core node coefficients.
    :type core_coefficients: np.ndarray
    :param core_covariance: The covariance matrix of the core node coefficients.
    :type core_covariance: np.ndarray
    :param core_edge_count: The number of edges in the core network.
    :type core_edge_count: int
    :return: The variance of the perturbation score.
    :rtype: float
    """
    temp = lap_q @ core_covariance
    unscaled_variance = 2 * np.trace(temp @ temp) + \
                        4 * core_coefficients @ temp @ lap_q @ core_coefficients
    return unscaled_variance / core_edge_count ** 2
