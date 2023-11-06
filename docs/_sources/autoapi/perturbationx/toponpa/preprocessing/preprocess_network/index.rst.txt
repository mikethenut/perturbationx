:py:mod:`perturbationx.toponpa.preprocessing.preprocess_network`
================================================================

.. py:module:: perturbationx.toponpa.preprocessing.preprocess_network


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.toponpa.preprocessing.preprocess_network.infer_node_type
   perturbationx.toponpa.preprocessing.preprocess_network.enumerate_nodes
   perturbationx.toponpa.preprocessing.preprocess_network.remove_invalid_graph_elements
   perturbationx.toponpa.preprocessing.preprocess_network.infer_edge_attributes
   perturbationx.toponpa.preprocessing.preprocess_network.infer_metadata



.. py:function:: infer_node_type(graph: networkx.DiGraph)

   Infer the type of each node in the network (core or boundary).

   :param graph: The network to process.
   :type graph: nx.DiGraph
   :raises ValueError: If the same node appears in both the core and boundary sets.
   :return: A tuple with the sets of boundary and core nodes.
   :rtype: (set, set)


.. py:function:: enumerate_nodes(graph: networkx.DiGraph, boundary_nodes: set, core_nodes: set)

   Assign an index and type to each node in the network.

   :param graph: The network to process.
   :type graph: nx.DiGraph
   :param boundary_nodes: The set of boundary nodes.
   :type boundary_nodes: set
   :param core_nodes: The set of core nodes.
   :type core_nodes: set


.. py:function:: remove_invalid_graph_elements(graph: networkx.DiGraph)

   Remove invalid elements from the graph. This function removes self-loops and opposing edges, and ensures that the
   core graph is weakly connected.

   :param graph: The network to process.
   :type graph: nx.DiGraph


.. py:function:: infer_edge_attributes(graph: networkx.DiGraph, relation_translator: Optional[perturbationx.io.RelationTranslator] = None)

   Infer the attributes of each edge in the network.

   :param graph: The network to process.
   :type graph: nx.DiGraph
   :param relation_translator: The relation translator to use. If None, a new instance will be created.
   :type relation_translator: perturbationx.RelationTranslator, optional


.. py:function:: infer_metadata(graph: networkx.DiGraph, verbose=True)

   Infer metadata about the network and add it to the graph instance.

   :param graph: The network to process.
   :type graph: nx.DiGraph
   :param verbose: Whether to log network statistics.
   :type verbose: bool, optional


