import numpy as np
import numpy.linalg as la
from scipy.sparse import issparse, sparray

from perturbationx.toponpa import generate_core_laplacians


def test_boundary_permutations(lap_b: np.ndarray | sparray, lap_c: np.ndarray | sparray, lap_q: np.ndarray | sparray,
                               boundary_coefficients: np.ndarray, core_edge_count: int,
                               permutation_rate=1., iterations=500, seed=None):
    """Test the null hypothesis that the ordering of boundary coefficients does not affect the perturbation score.

    :param lap_b: The Lb boundary Laplacian.
    :type lap_b: np.ndarray | sp.sparray
    :param lap_c: The Lc core Laplacian.
    :type lap_c: np.ndarray | sp.sparray
    :param lap_q: The Q core Laplacian.
    :type lap_q: np.ndarray | sp.sparray
    :param boundary_coefficients: The boundary coefficients.
    :type boundary_coefficients: np.ndarray
    :param core_edge_count: The number of edges in the core network.
    :type core_edge_count: int
    :param permutation_rate: The fraction of boundary coefficients to permute. Defaults to 1.
    :type permutation_rate: float, optional
    :param iterations: The number of permutations to perform. Defaults to 500.
    :type iterations: int, optional
    :param seed: The seed for the random number generator. Defaults to None.
    :type seed: int, optional
    :return: The distribution of perturbation scores under the null hypothesis.
    :rtype: list
    """
    if issparse(lap_c):
        lap_c = lap_c.todense()
    inference_matrix = - la.solve(lap_c, lap_b) if not issparse(lap_b) else None

    rng = np.random.default_rng(seed)
    permutation_count = np.ceil(permutation_rate * boundary_coefficients.shape[0]).astype(int)
    distribution = []

    for p in range(iterations):
        permuted_idx = rng.choice(boundary_coefficients.shape[0], size=permutation_count, replace=False)
        permuted_boundary = boundary_coefficients.copy()
        permuted_boundary[permuted_idx] = \
            rng.permutation(permuted_boundary[permuted_idx])

        if issparse(lap_b):  # We do not precompute the inference matrix for sparse matrices
            permuted_core = - la.solve(lap_c, lap_b @ permuted_boundary)
        else:
            permuted_core = inference_matrix @ permuted_boundary

        sample_perturbation = permuted_core.T @ lap_q @ permuted_core
        distribution.append(sample_perturbation / core_edge_count)

    return distribution


def test_core_permutations(adj_perms: dict, boundary_coefficients: np.ndarray, lap_b: np.ndarray | sparray,
                           core_edge_count: int, exact_boundary_outdegree=True,
                           lap_q=None, full_permutations=True):
    """Test the null hypothesis that the distribution of core edges does not affect the perturbation score.

    :param adj_perms: The adjacency matrices of the core network permutations. The keys are the permutation names,
                        the values are the adjacency matrices. The adjacency matrices may be sparse or dense.
    :type adj_perms: dict
    :param boundary_coefficients: The boundary coefficients.
    :type boundary_coefficients: np.ndarray
    :param lap_b: The Lb boundary Laplacian.
    :type lap_b: np.ndarray | sp.sparray
    :param core_edge_count: The number of edges in the core network.
    :type core_edge_count: int
    :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
                                        is set to 1 for all core nodes with boundary edges. Defaults to True.
    :type exact_boundary_outdegree: bool, optional
    :param lap_q: The Q core Laplacian. Required if full_permutations is False. Defaults to None.
    :type lap_q: np.ndarray | sp.sparray, optional
    :param full_permutations: Whether to use the full permutation matrix for each core permutation. Partial
                                permutations sample core coefficients, while full permutations sample perturbation
                                scores. Defaults to True.
    :type full_permutations: bool, optional
    :raises ValueError: If full_permutations is False and lap_q is None.
    :return: The distribution of perturbation scores under the null hypothesis.
    :rtype: list
    """
    if not full_permutations and lap_q is None:
        raise ValueError("Parameter lap_q must be provided if full_permutations is False.")

    edge_constraints = - lap_b @ boundary_coefficients
    distribution = []

    for adj_c_perm in adj_perms:
        lap_c_perm, lap_q_perm = generate_core_laplacians(lap_b, adj_c_perm, exact_boundary_outdegree)

        if issparse(lap_c_perm):
            lap_c_perm = lap_c_perm.todense()
        core_coefficients = la.solve(lap_c_perm, edge_constraints)

        if full_permutations:
            lap_q = lap_q_perm
        sample_perturbation = core_coefficients.T @ lap_q @ core_coefficients

        distribution.append(sample_perturbation / core_edge_count)

    return distribution
