import warnings

import numpy as np
import numpy.random
import numpy.linalg as la


def permutation_test_o(lap_b: np.ndarray, lap_c: np.ndarray, lap_q: np.ndarray, boundary_coefficients: np.ndarray,
                       core_edge_count: int, p_iters=500, seed=None):
    if seed is None:
        generator = np.random.default_rng()
    else:
        generator = np.random.default_rng(seed)

    inference_matrix = - np.matmul(la.inv(lap_c), lap_b)
    distribution = []

    for p in range(p_iters):
        permuted_boundary = generator.permutation(boundary_coefficients)
        permuted_core = np.matmul(inference_matrix, permuted_boundary)
        sample_perturbation = np.matmul(permuted_core.transpose(), lap_q.dot(permuted_core)) / core_edge_count
        distribution.append(sample_perturbation)

    return distribution


def permutation_test_k(lap_b: np.ndarray, lc_permutations, lap_q: np.ndarray,
                       boundary_coefficients: np.ndarray, core_edge_count: int):
    edge_constraints = - lap_b.dot(boundary_coefficients)
    distribution = []

    for lc in lc_permutations:
        try:
            backbone_values = np.matmul(la.inv(lc), edge_constraints)
            sample_perturbation = np.matmul(lap_q.dot(backbone_values), backbone_values) / core_edge_count
            distribution.append(sample_perturbation)
        except la.LinAlgError:
            # Singular backbone generated
            continue

    return distribution


def permutation_tests(lap_b: np.ndarray, lap_c: np.ndarray, lap_q: np.ndarray, lperms: dict,
                      core_edge_count: int, boundary_coefficients: np.ndarray,
                      permutations=('o', 'k'), p_iters=500, seed=None):
    distributions = dict()
    for p in set(permutations):
        match p.lower():
            case 'o':
                distributions[p] = permutation_test_o(lap_b, lap_c, lap_q, boundary_coefficients,
                                                      core_edge_count, p_iters, seed)
            case 'k':
                distributions[p] = permutation_test_k(lap_b, lperms['k'], lap_q,
                                                      boundary_coefficients, core_edge_count)
            case _:
                warnings.warn("Permutation %s is unknown and will be skipped." % p)
                continue

    return distributions
