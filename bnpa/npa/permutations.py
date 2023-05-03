import warnings

import numpy as np
import numpy.random
import numpy.linalg as la


def permutation_test_o(lb: np.ndarray, lc: np.ndarray, lq: np.ndarray, boundary_coefficients: np.ndarray,
                       core_edge_count: int, p_iters=500, seed=None):
    if seed is None:
        generator = np.random.default_rng()
    else:
        generator = np.random.default_rng(seed)

    inference_matrix = - np.matmul(la.inv(lc), lb)
    distribution = []

    for p in range(p_iters):
        permuted_boundary = generator.permutation(boundary_coefficients)
        permuted_core = np.matmul(inference_matrix, permuted_boundary)
        sample_perturbation = np.matmul(permuted_core.transpose(), lq.dot(permuted_core)) / core_edge_count
        distribution.append(sample_perturbation)

    return distribution


def permutation_test_k(lb: np.ndarray, lc_permutations, lq: np.ndarray,
                       boundary_coefficients: np.ndarray, core_edge_count: int):

    edge_constraints = - lb.dot(boundary_coefficients)
    distribution = []

    for lc in lc_permutations:
        backbone_values = np.matmul(la.inv(lc), edge_constraints)
        sample_perturbation = np.matmul(lq.dot(backbone_values), backbone_values) / core_edge_count
        distribution.append(sample_perturbation)

    return distribution


def compute_permutations(lap: dict, lperms: dict, core_edge_count: int, boundary_coefficients: np.ndarray,
                         permutations=('o', 'k'), p_iters=500, seed=None):

    distributions = dict()
    for p in set(permutations):
        match p.lower():
            case 'o':
                distributions[p] = permutation_test_o(lap['b'], lap['c'], lap['q'], boundary_coefficients,
                                                      core_edge_count, p_iters, seed)
            case 'k':
                distributions[p] = permutation_test_k(lap['b'], lperms['k'], lap['q'],
                                                      boundary_coefficients, core_edge_count)
            case _:
                warnings.warn("Permutation %s is unknown and will be skipped." % p)
                continue

    return distributions


def p_value(value, distribution):
    return sum(sample > value for sample in distribution) / len(distribution)
