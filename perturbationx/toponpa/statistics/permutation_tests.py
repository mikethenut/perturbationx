import numpy as np
import numpy.linalg as la
from scipy.sparse import issparse

from perturbationx.toponpa import generate_core_laplacians


def test_boundary_permutations(lap_b, lap_c, lap_q, boundary_coefficients: np.ndarray,
                               core_edge_count: int, permutation_rate=1., iterations=500, seed=None):
    if issparse(lap_c):
        lap_c = lap_c.todense()
    lap_c_inv = la.inv(lap_c)
    inference_matrix = - lap_c_inv @ lap_b if not issparse(lap_b) else None

    rng = np.random.default_rng(seed)
    permutation_count = np.ceil(permutation_rate * boundary_coefficients.shape[0]).astype(int)
    distribution = []
    for p in range(iterations):
        permuted_idx = rng.choice(boundary_coefficients.shape[0], size=permutation_count, replace=False)
        permuted_boundary = boundary_coefficients.copy()
        permuted_boundary[permuted_idx] = \
            rng.permutation(permuted_boundary[permuted_idx])

        if issparse(lap_b):  # We do not precompute the inference matrix for sparse matrices
            permuted_core = - lap_c_inv @ (lap_b @ permuted_boundary)
        else:
            permuted_core = inference_matrix @ permuted_boundary

        sample_perturbation = permuted_core.T @ lap_q @ permuted_core
        distribution.append(sample_perturbation / core_edge_count)

    return distribution


def test_core_permutations(adj_perms, boundary_coefficients: np.ndarray, lap_b: np.ndarray,
                           core_edge_count: int, exact_boundary_outdegree=True,
                           lap_q=None, full_permutations=True):
    if not full_permutations and lap_q is None:
        raise ValueError("Parameter lap_q must be provided if full_permutations is False.")

    edge_constraints = - lap_b @ boundary_coefficients
    distribution = []

    for adj_c_perm in adj_perms:
        lap_c_perm, lap_q_perm = generate_core_laplacians(lap_b, adj_c_perm, exact_boundary_outdegree)
        if full_permutations:
            lap_q = lap_q_perm

        lap_c_inv = la.inv(lap_c_perm.todense()) if issparse(lap_c_perm) else la.inv(lap_c_perm)
        core_coefficients = lap_c_inv @ edge_constraints
        sample_perturbation = core_coefficients.T @ lap_q @ core_coefficients
        distribution.append(sample_perturbation / core_edge_count)

    return distribution
