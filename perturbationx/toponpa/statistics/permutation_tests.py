import numpy as np
import numpy.linalg as la

from perturbationx.toponpa import generate_core_laplacians


def test_boundary_permutations(lap_b: np.ndarray, lap_c: np.ndarray, lap_q: np.ndarray,
                               boundary_coefficients: np.ndarray, core_edge_count: int,
                               permutation_rate=1., iterations=500, seed=None):
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


def test_core_permutations(adj_perms, boundary_coefficients: np.ndarray, lap_b: np.ndarray,
                           core_edge_count: int, exact_boundary_outdegree=True,
                           lap_q=None, full_permutations=True):
    if not full_permutations and lap_q is None:
        raise ValueError("Parameter lap_q must be provided if full_permutations is False.")

    edge_constraints = - lap_b.dot(boundary_coefficients)
    distribution = []

    for adj_c_perm in adj_perms:
        lap_c_perm, lap_q_perm = generate_core_laplacians(lap_b, adj_c_perm, exact_boundary_outdegree)
        if full_permutations:
            lap_q = lap_q_perm

        core_coefficients = np.matmul(la.inv(lap_c_perm), edge_constraints)
        sample_perturbation = np.matmul(lap_q.dot(core_coefficients), core_coefficients)
        distribution.append(sample_perturbation / core_edge_count)

    return distribution
