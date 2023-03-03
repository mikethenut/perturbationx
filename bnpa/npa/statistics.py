from bnpa.exporter.network import write_matrix_alt
import math
import warnings

import numpy as np
import numpy.random
from scipy.stats import norm
import scipy.linalg as la

from bnpa.npa.core import value_diffusion, perturbation_amplitude


def backbone_covariance_matrix(diffusion_matrix: np.ndarray, fold_change: np.ndarray, t_statistic: np.ndarray):
    standard_error = np.divide(fold_change, t_statistic)
    repeated = np.repeat([standard_error], diffusion_matrix.shape[0], axis=0)
    temp = np.multiply(diffusion_matrix, repeated)
    return np.matmul(temp, np.transpose(temp))


def backbone_confidence_interval(backbone_values: np.ndarray, backbone_covariance: np.ndarray, alpha=0.95):
    percentile_threshold = norm.ppf((1. + alpha) / 2.)

    variances = np.diag(backbone_covariance)
    standard_deviations = np.sqrt(variances)
    confidence_intervals = [np.subtract(backbone_values, percentile_threshold * standard_deviations),
                            np.add(backbone_values, percentile_threshold * standard_deviations)]
    p_values = np.subtract(1., norm.cdf(np.divide(np.abs(backbone_values), standard_deviations)))

    return variances, confidence_intervals, p_values


def perturbation_confidence_interval(q: np.ndarray, backbone_values: np.ndarray, backbone_covariance: np.ndarray,
                                     backbone_edge_count: int, alpha=0.95):

    percentile_threshold = norm.ppf((1. + alpha) / 2.)
    perturbation = perturbation_amplitude(q, backbone_values, backbone_edge_count)

    temp = np.matmul(q, backbone_covariance)
    unscaled_variance = 2 * np.trace(np.matmul(temp, temp)) + \
                        4 * np.matmul(np.matmul(backbone_values, temp),
                                      np.matmul(q, backbone_values))

    variance = unscaled_variance / backbone_edge_count**2
    standard_deviation = math.sqrt(variance)
    confidence_interval = [perturbation - percentile_threshold * standard_deviation,
                           perturbation + percentile_threshold * standard_deviation]

    return variance, confidence_interval


def permutation_test_o(diffusion_matrix: np.ndarray, q: np.ndarray, fold_change: np.ndarray,
                       backbone_edge_count: int, permutations=500, seed=None):
    if seed is None:
        generator = np.random.default_rng()
    else:
        generator = np.random.default_rng(seed)

    true_perturbation = perturbation_amplitude(q, value_diffusion(diffusion_matrix, fold_change), backbone_edge_count)
    distribution = []
    for p in range(permutations):
        permuted_downstream = generator.permutation(fold_change)

        backbone_values = value_diffusion(diffusion_matrix, permuted_downstream)
        sample_perturbation = perturbation_amplitude(q, backbone_values, backbone_edge_count)
        distribution.append(sample_perturbation)

    p_value = sum(sample > true_perturbation for sample in distribution) / permutations
    return p_value, distribution


def permutation_test_k(l3_permutations, l2: np.ndarray, q: np.ndarray, fold_change: np.ndarray,
                       backbone_edge_count: int, true_perturbation: float):

    distribution = []
    temp = l2.dot(fold_change)
    for l3 in l3_permutations:
        try:
            l3_inv = la.inv(l3)
        except la.LinAlgError:
            warnings.warn("Singular backbone in permutation test K.")
            write_matrix_alt(l3)
            continue

        backbone_values = np.matmul(l3_inv, temp)
        sample_perturbation = perturbation_amplitude(q, backbone_values, backbone_edge_count)
        distribution.append(sample_perturbation)

    p_value = sum(sample > true_perturbation for sample in distribution) / len(distribution)
    return p_value, distribution
