:py:mod:`perturbationx.topo_npa.permutation.adjacency_permutation`
==================================================================

.. py:module:: perturbationx.topo_npa.permutation.adjacency_permutation


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.toponpa.permutation.adjacency_permutation.adjacency_permutation_k1
   perturbationx.toponpa.permutation.adjacency_permutation.adjacency_permutation_k2



.. py:function:: adjacency_permutation_k1(adj: numpy.ndarray | scipy.sparse.sparray, iterations=500, permutation_rate=1.0, ensure_connectedness=True, seed=None)

   Permute the edges of an adjacency matrix using the "K1" method. This method permutes edges randomly.

   :param adj: The adjacency matrix to permute.
   :type adj: np.ndarray | sp.sparray
   :param iterations: The number of permutations to generate. Defaults to 500.
   :type iterations: int, optional
   :param permutation_rate: The fraction of edges to permute. Defaults to 1.
   :type permutation_rate: float, optional
   :param ensure_connectedness: Whether to ensure that the permuted adjacency matrix is connected. Defaults to True.
   :type ensure_connectedness: bool, optional
   :param seed: The seed for the random number generator.
   :type seed: int, optional
   :raises ValueError: If the adjacency matrix is not square.
   :return: A list of permuted adjacency matrices.
   :rtype: list


.. py:function:: adjacency_permutation_k2(adj: numpy.ndarray | scipy.sparse.sparray, iterations=500, permutation_rate=1.0, ensure_connectedness=True, seed=None)

   Permute the edges of an adjacency matrix using the "K2" method. This method permutes edges by preserving
   the degree of each node as much as possible.

   :param adj: The adjacency matrix to permute.
   :type adj: np.ndarray | sp.sparray
   :param iterations: The number of permutations to generate. Defaults to 500.
   :type iterations: int, optional
   :param permutation_rate: The fraction of edges to permute. Defaults to 1.
   :type permutation_rate: float, optional
   :param ensure_connectedness: Whether to ensure that the permuted adjacency matrix is connected. Defaults to True.
   :type ensure_connectedness: bool, optional
   :param seed: The seed for the random number generator.
   :type seed: int, optional
   :raises ValueError: If the adjacency matrix is not square.
   :return: A list of permuted adjacency matrices.
   :rtype: list


