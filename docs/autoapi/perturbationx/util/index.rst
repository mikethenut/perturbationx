:py:mod:`perturbationx.util`
============================

.. py:module:: perturbationx.util


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   connectivity/index.rst
   neighbors/index.rst
   shortest_paths/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.util.connect_adjacency_components
   perturbationx.util.get_neighborhood_components
   perturbationx.util.get_shortest_path_components



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


