import warnings

import numpy as np
import numpy.random
import numpy.linalg as la


def permutation_test_o(lap_b: np.ndarray, lap_c: np.ndarray, lap_q: np.ndarray, boundary_coefficients: np.ndarray,
                       core_edge_count: int, permutation_rate=1., iterations=500, seed=None):
    rng = np.random.default_rng(seed)
    inference_matrix = - np.matmul(la.inv(lap_c), lap_b)
    permutation_count = np.ceil(permutation_rate * boundary_coefficients.shape[0]).astype(int)

    distribution = []
    for p in range(iterations):
        permuted_idx = rng.choice(boundary_coefficients.shape[0], size=permutation_count, replace=False)
        permuted_boundary = boundary_coefficients.copy()
        permuted_boundary[permuted_idx] = \
            rng.permutation(permuted_boundary[permuted_idx])

        permuted_core = np.matmul(inference_matrix, permuted_boundary)
        sample_perturbation = np.matmul(permuted_core.transpose(), lap_q.dot(permuted_core)) / core_edge_count
        distribution.append(sample_perturbation)

    return distribution


def permutation_test_k(lap_b: np.ndarray, lc_permutations, lap_q: np.ndarray,
                       boundary_coefficients: np.ndarray, core_edge_count: int):
    edge_constraints = - lap_b.dot(boundary_coefficients)
    distribution = []

    for lc in lc_permutations:
        backbone_values = np.matmul(la.inv(lc), edge_constraints)
        sample_perturbation = np.matmul(lap_q.dot(backbone_values), backbone_values) / core_edge_count
        distribution.append(sample_perturbation)

    return distribution


def permutation_tests(lap_b: np.ndarray, lap_c: np.ndarray, lap_q: np.ndarray, lperms: dict,
                      core_edge_count: int, boundary_coefficients: np.ndarray,
                      permutations=('o', 'k2'), permutation_rate=1., iterations=500, seed=None):
    distributions = dict()
    for p in set(permutations):
        match p.lower():
            case 'o':
                distributions[p] = permutation_test_o(
                    lap_b, lap_c, lap_q, boundary_coefficients, core_edge_count,
                    permutation_rate=permutation_rate, iterations=iterations, seed=seed
                )
            case 'k1' | 'k2':
                distributions[p] = permutation_test_k(
                    lap_b, lperms[p], lap_q, boundary_coefficients, core_edge_count
                )
            case _:
                warnings.warn("Permutation %s is unknown and will be skipped." % p)
                continue

    return distributions
