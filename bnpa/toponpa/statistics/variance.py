import numpy as np
import numpy.linalg as la


def core_covariance_matrix(lb: np.ndarray, lc: np.ndarray, stderr: np.ndarray):
    inference_matrix = np.matmul(la.inv(lc), lb)
    repeated = np.repeat([stderr], inference_matrix.shape[0], axis=0)
    temp = np.multiply(inference_matrix, repeated)
    return np.matmul(temp, temp.T)


def perturbation_variance(lq: np.ndarray, core_coefficients: np.ndarray,
                          core_covariance: np.ndarray, core_edge_count: int):
    temp = np.matmul(lq, core_covariance)
    unscaled_variance = 2 * np.trace(np.matmul(temp, temp)) + \
                        4 * np.matmul(np.matmul(core_coefficients, temp),
                                      np.matmul(lq, core_coefficients))
    return unscaled_variance / core_edge_count ** 2
