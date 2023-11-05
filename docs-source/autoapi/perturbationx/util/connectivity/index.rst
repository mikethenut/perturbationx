:py:mod:`perturbationx.util.connectivity`
=========================================

.. py:module:: perturbationx.util.connectivity


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.util.connectivity.connect_adjacency_components



.. py:function:: connect_adjacency_components(adj: numpy.ndarray, nodes=None, weights=(1.0, ), seed=None)

   Connect the components of a graph represented by an adjacency matrix.

   :param adj: The adjacency matrix of the graph.
   :type adj: np.ndarray
   :param nodes: The nodes to connect. If specified, edges will only be added between these nodes (when possible).
   :type nodes: list, optional
   :param weights: The weights to assign to the edges. Each edge will be assigned a random weight from this list.
   :type weights: list, optional
   :param seed: The seed for the random number generator.
   :type seed: int, optional
   :raises ValueError: If the adjacency matrix is not square.


