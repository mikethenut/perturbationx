:py:mod:`perturbationx.util.shortest_paths`
===========================================

.. py:module:: perturbationx.util.shortest_paths


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.util.shortest_paths.get_shortest_path_components



.. py:function:: get_shortest_path_components(graph: networkx.DiGraph, endpoints: set, directed=False, length_tolerance=0)

   Find paths between nodes in a graph and return the nodes and edges in those paths.

   :param graph: The graph to search for paths in.
   :type graph: nx.DiGraph
   :param endpoints: The nodes to find paths between.
   :type endpoints: set
   :param directed: Whether to search for directed or undirected shortest paths.
   :type directed: bool, optional
   :param length_tolerance: The tolerance for the length of the shortest paths. If 0, only the shortest paths
           are returned. If length_tolerance is an integer, it is interpreted as an absolute length. If
           length_tolerance is a float, it is interpreted as a percentage of the length of the shortest path.
           Defaults to 0.
   :type length_tolerance: int | float, optional
   :raises ValueError: If length_tolerance is not a number or is negative.
   :return: The nodes and edges in the paths. They are returned as a pair of lists.
   :rtype: (list, list)


