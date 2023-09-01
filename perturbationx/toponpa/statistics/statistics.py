import warnings

import numpy as np
from scipy.stats import norm

from .variance import core_covariance_matrix, perturbation_variance
from .permutation_tests import test_boundary_permutations, test_core_permutations

__all__ = ["compute_variances", "confidence_interval", "test_permutations", "p_value"]


def compute_variances(lap_b: np.ndarray, lap_c: np.ndarray, lap_q: np.ndarray,
                      stderr: np.ndarray, core_coefficients: np.ndarray, core_edge_count: int):
    core_covariance = core_covariance_matrix(lap_b, lap_c, stderr)
    node_variance = np.diag(core_covariance)
    npa_variance = perturbation_variance(lap_q, core_coefficients, core_covariance, core_edge_count)
    return npa_variance, node_variance


def confidence_interval(values, variances, alpha=0.95):
    percentile_threshold = norm.ppf((1. + alpha) / 2.)
    std_dev = np.sqrt(variances)

    ci_lower = values - percentile_threshold * std_dev
    ci_upper = values + percentile_threshold * std_dev
    p_values = np.subtract(1., norm.cdf(np.divide(np.abs(values), std_dev)))
    return ci_lower, ci_upper, p_values


def test_permutations(lap_b: np.ndarray, lap_c: np.ndarray, lap_q: np.ndarray, lperms: dict,
                      core_edge_count: int, boundary_coefficients: np.ndarray,
                      permutations=('o', 'k2'), permutation_rate=1., iterations=500, seed=None):
    distributions = dict()
    for p in set(permutations):
        match p.lower():
            case 'o':
                distributions[p] = test_boundary_permutations(
                    lap_b, lap_c, lap_q, boundary_coefficients, core_edge_count,
                    permutation_rate=permutation_rate, iterations=iterations, seed=seed
                )
            case 'k1' | 'k2':
                distributions[p] = test_core_permutations(
                    lap_b, lperms[p], lap_q, boundary_coefficients, core_edge_count
                )
            case _:
                warnings.warn("Permutation %s is unknown and will be skipped." % p)
                continue

    return distributions


def p_value(value, distribution):
    return sum(sample > value for sample in distribution) / len(distribution)
