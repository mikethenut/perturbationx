:py:mod:`perturbationx.toponpa.statistics.permutation_tests`
============================================================

.. py:module:: perturbationx.toponpa.statistics.permutation_tests


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.toponpa.statistics.permutation_tests.test_boundary_permutations
   perturbationx.toponpa.statistics.permutation_tests.test_core_permutations



.. py:function:: test_boundary_permutations(lap_b: numpy.ndarray | scipy.sparse.sparray, lap_c: numpy.ndarray | scipy.sparse.sparray, lap_q: numpy.ndarray | scipy.sparse.sparray, boundary_coefficients: numpy.ndarray, core_edge_count: int, permutation_rate=1.0, iterations=500, seed=None)

   Test the null hypothesis that the ordering of boundary coefficients does not affect the perturbation score.

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


.. py:function:: test_core_permutations(adj_perms: dict, boundary_coefficients: numpy.ndarray, lap_b: numpy.ndarray | scipy.sparse.sparray, core_edge_count: int, exact_boundary_outdegree=True, lap_q=None, full_permutations=True)

   Test the null hypothesis that the distribution of core edges does not affect the perturbation score.

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


