import numpy as np
import numpy.linalg as la
from scipy.sparse import issparse, lil_array


def core_covariance_matrix(lap_b, lap_c, stderr: np.ndarray):
    if issparse(lap_c):
        lap_c = lap_c.todense()
    lap_c_inv = la.inv(lap_c)

    err_diag = lil_array((stderr.shape[0], stderr.shape[0]))
    err_diag.setdiag(stderr ** 2)
    tmp = lap_b @ err_diag @ lap_b.T
    res = lap_c_inv @ tmp @ lap_c_inv.T
    return res


def perturbation_variance(lap_q: np.ndarray, core_coefficients: np.ndarray,
                          core_covariance: np.ndarray, core_edge_count: int):
    temp = lap_q @ core_covariance
    unscaled_variance = 2 * np.trace(temp @ temp) + \
                        4 * core_coefficients @ temp @ lap_q @ core_coefficients
    return unscaled_variance / core_edge_count ** 2
