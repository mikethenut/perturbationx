:py:mod:`perturbationx.util.neighbors`
======================================

.. py:module:: perturbationx.util.neighbors


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.util.neighbors.get_neighborhood_components



.. py:function:: get_neighborhood_components(graph: networkx.DiGraph, source_nodes: set, max_distance=0, neighborhood_type='union')

   Get the neighborhood of a set of nodes in a graph.

   :param graph: The graph to search for neighbors in.
   :type graph: nx.DiGraph
   :param source_nodes: The nodes to find neighbors of.
   :type source_nodes: set
   :param max_distance: The maximum distance from leading nodes that neighbors can be. If 0, no neighbors
           will be included. If 1, only the neighbors of the nodes will be included. If 2, the neighbors of the
           neighbors of the nodes will be included, and so on. Defaults to 0.
   :type max_distance: int, optional
   :param neighborhood_type: The type of neighborhood to find. Can be one of "union" or "intersection".
           If "union", all nodes within the maximum distance from any leading node are returned. If "intersection",
           only nodes within the maximum distance from all leading nodes are returned. Defaults to "union".
   :type neighborhood_type: str, optional
   :raises ValueError: If max_distance is less than 0 or neighborhood_type is not "union" or "intersection".
   :return: The nodes and edges in the neighborhood. They are returned as a pair of sets.
   :rtype: (set, set)


