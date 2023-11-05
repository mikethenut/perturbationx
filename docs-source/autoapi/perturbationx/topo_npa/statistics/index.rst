:py:mod:`perturbationx.topo_npa.statistics`
===========================================

.. py:module:: perturbationx.topo_npa.statistics


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   permutation_tests/index.rst
   statistics/index.rst
   variance/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.toponpa.statistics.compute_variances
   perturbationx.toponpa.statistics.confidence_interval
   perturbationx.toponpa.statistics.test_permutations
   perturbationx.toponpa.statistics.p_value



.. py:function:: compute_variances(lap_b: numpy.ndarray | scipy.sparse.sparray, lap_c: numpy.ndarray | scipy.sparse.sparray, lap_q: numpy.ndarray | scipy.sparse.sparray, stderr: numpy.ndarray, core_coefficients: numpy.ndarray, core_edge_count: int)

   Compute the variance of the perturbation score and core node coefficients.

   :param lap_b: The Lb boundary Laplacian.
   :type lap_b: np.ndarray | sp.sparray
   :param lap_c: The Lc core Laplacian.
   :type lap_c: np.ndarray | sp.sparray
   :param lap_q: The Q core Laplacian.
   :type lap_q: np.ndarray | sp.sparray
   :param stderr: The standard error of the boundary coefficients.
   :type stderr: np.ndarray
   :param core_coefficients: The core node coefficients.
   :type core_coefficients: np.ndarray
   :param core_edge_count: The number of edges in the core network.
   :type core_edge_count: int
   :return: The variance of the perturbation score and core node coefficients.
   :rtype: (np.ndarray, np.ndarray)


.. py:function:: confidence_interval(values: numpy.ndarray, variances: numpy.ndarray, alpha=0.95)

   Compute the confidence intervals for the given significance level.

   :param values: The mean values for which to compute the confidence intervals.
   :type values: np.ndarray
   :param variances: The variances of the values.
   :type variances: np.ndarray
   :param alpha: The confidence level. Defaults to 0.95.
   :type alpha: float, optional
   :return: The lower and upper confidence intervals and the p-values.
   :rtype: (np.ndarray, np.ndarray, np.ndarray)


.. py:function:: test_permutations(lap_b: numpy.ndarray | scipy.sparse.sparray, lap_c: numpy.ndarray | scipy.sparse.sparray, lap_q: numpy.ndarray | scipy.sparse.sparray, adj_perms: dict, core_edge_count: int, boundary_coefficients: numpy.ndarray, permutations=('o', 'k2'), full_core_permutation=True, exact_boundary_outdegree=True, permutation_rate=1.0, iterations=500, seed=None)

   Test the null hypothesis that the ordering of boundary coefficients and the distribution of core edges does not
   affect the perturbation score. This is a convenience function that calls test_boundary_permutations and
   test_core_permutations.

   :param lap_b: The Lb boundary Laplacian.
   :type lap_b: np.ndarray | sp.sparray
   :param lap_c: The Lc core Laplacian.
   :type lap_c: np.ndarray | sp.sparray
   :param lap_q: The Q core Laplacian.
   :type lap_q: np.ndarray | sp.sparray
   :param adj_perms: The adjacency matrices of the core network permutations. The keys are the permutation names,
                       the values are the adjacency matrices. The adjacency matrices may be sparse or dense.
   :type adj_perms: dict
   :param core_edge_count: The number of edges in the core network.
   :type core_edge_count: int
   :param boundary_coefficients: The boundary coefficients.
   :type boundary_coefficients: np.ndarray
   :param permutations: The permutations to test. May contain 'o', 'k1', and 'k2' in any order.
                           Defaults to ('o', 'k2'). For 'k1' and 'k2', the adjacency matrices to test must be
                           provided in adj_perms.
   :type permutations: list, optional
   :param full_core_permutation: Whether to use the full permutation matrix for each core permutation. Partial
                                   permutations sample core coefficients, while full permutations sample perturbation
                                   scores. Defaults to True.
   :type full_core_permutation: bool, optional
   :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
                                       is set to 1 for all core nodes with boundary edges. Defaults to True.
   :type exact_boundary_outdegree: bool, optional
   :param permutation_rate: The fraction of boundary coefficients to permute. Defaults to 1.
   :type permutation_rate: float, optional
   :param iterations: The number of boundary permutations to perform. Defaults to 500.
   :type iterations: int, optional
   :param seed: The seed for the random number generator. Defaults to None.
   :type seed: int, optional
   :return: The distributions of perturbation scores under the null hypothesis.
   :rtype: dict


.. py:function:: p_value(value: float, distribution: list)

   Compute the p-value for the given value in the given distribution.

   :param value: The value for which to compute the p-value.
   :type value: float
   :param distribution: The distribution.
   :type distribution: list
   :return: The p-value.
   :rtype: float


