:py:mod:`perturbationx.topo_npa.permutation`
============================================

.. py:module:: perturbationx.topo_npa.permutation


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   adjacency_permutation/index.rst
   permutation/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.toponpa.permutation.permute_adjacency
   perturbationx.toponpa.permutation.permute_edge_list



.. py:function:: permute_adjacency(adj: numpy.ndarray | scipy.sparse.sparray, permutations=('k2', ), iterations=500, permutation_rate=1.0, seed=None)

   Permute an adjacency matrix.

   :param adj: The adjacency matrix to permute.
   :type adj: np.ndarray | sp.sparray
   :param permutations: The permutations to apply. May contain 'k1' and 'k2' in any order. Defaults to ('k2',).
   :type permutations: list, optional
   :param iterations: The number of permutations to generate. Defaults to 500.
   :type iterations: int, optional
   :param permutation_rate: The fraction of edges to permute. Defaults to 1.
   :type permutation_rate: float, optional
   :param seed: The seed for the random number generator.
   :type seed: int, optional
   :raises ValueError: If the adjacency matrix is not square.
   :return: A dictionary of lists with permuted adjacency matrices, keyed by the permutation name.
   :rtype: dict


.. py:function:: permute_edge_list(edge_list: numpy.ndarray, node_list=None, iterations=500, method='k1', permutation_rate=1.0, seed=None)

   Permute an edge list.

   :param edge_list: The edge list to permute. Must be a 2D array with shape (n_edges, 4). The first two columns
       contain the source and target nodes, the third column contains the edge type, and the fourth column contains
       the confidence weight. Confidence weights are optional.
   :type edge_list: np.ndarray
   :param node_list: The list of nodes to use in the permutation. Only edges that connect nodes in this list
       are permuted. If None, the list is inferred from the edge list.
   :type node_list: list, optional
   :param iterations: The number of permutations to generate. Defaults to 500.
   :type iterations: int, optional
   :param method: The permutation method to use. Defaults to 'k1'. May be 'k1' or 'k2'.
   :type method: str, optional
   :param permutation_rate: The fraction of edges to permute. Defaults to 1. If 'confidence', the confidence weights
       are used to determine the number of edges to permute. For each edge, a random number is drawn from a uniform
       distribution between 0 and 1. If the confidence weight is larger than this number, the edge is permuted.
   :type permutation_rate: float | str, optional
   :param seed: The seed for the random number generator.
   :type seed: int, optional
   :raises ValueError: If the permutation method is unknown.
   :return: A list of permutations. Each permutation is a list of tuples with the source node, target node, and edge
       type. If the edge type is None, the edge is removed.


