import numpy as np

from scipy.stats import norm
import numpy.linalg as la


def core_covariance_matrix(lb: np.ndarray, lc: np.ndarray, stderr: np.ndarray):
    inference_matrix = np.matmul(la.inv(lc), lb)
    repeated = np.repeat([stderr], inference_matrix.shape[0], axis=0)
    temp = np.multiply(inference_matrix, repeated)
    return np.matmul(temp, temp.T)


def amplitude_variance(lq: np.ndarray, core_coefficients: np.ndarray,
                       core_covariance: np.ndarray, core_edge_count: int):
    temp = np.matmul(lq, core_covariance)
    unscaled_variance = 2 * np.trace(np.matmul(temp, temp)) + \
                        4 * np.matmul(np.matmul(core_coefficients, temp),
                                      np.matmul(lq, core_coefficients))
    return unscaled_variance / core_edge_count ** 2


def compute_variances(lap: dict, stderr: np.ndarray, core_coefficients: np.ndarray, core_edge_count: int):
    core_covariance = core_covariance_matrix(lap['b'], lap['c'], stderr)
    node_variance = np.diag(core_covariance)
    npa_variance = amplitude_variance(lap['q'], core_coefficients, core_covariance, core_edge_count)
    return npa_variance, node_variance


def confidence_interval(values, variances, alpha=0.95):
    percentile_threshold = norm.ppf((1. + alpha) / 2.)
    std_dev = np.sqrt(variances)

    ci_lower = values - percentile_threshold * std_dev
    ci_upper = values + percentile_threshold * std_dev
    p_values = np.subtract(1., norm.cdf(np.divide(np.abs(values), std_dev)))
    return ci_lower, ci_upper, p_values
