:py:mod:`perturbationx`
=======================

.. py:module:: perturbationx


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   io/index.rst
   resources/index.rst
   result/index.rst
   toponpa/index.rst
   util/index.rst
   vis/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   CausalNetwork/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   perturbationx.CausalNetwork
   perturbationx.RelationTranslator
   perturbationx.NPAResultBuilder
   perturbationx.NPAResult
   perturbationx.NPAResultDisplay




.. py:class:: CausalNetwork(graph: Optional[networkx.DiGraph] = None, relation_translator: Optional[perturbationx.io.RelationTranslator] = None, inplace=False)
   :no-index:


   Class for storing and evaluating causal networks. The network is stored as a networkx.DiGraph. The class
   provides methods for reading and writing networks from and to various file formats, as well as for evaluating
   the network using the TopoNPA algorithm. The class also provides methods for rewiring and wiring edges in the
   network, as well as for evaluating the rewired and wired networks.

   :param graph: The network graph. Defaults to None (empty graph).
   :type graph: networkx.DiGraph, optional
   :param relation_translator: The relation translator. Defaults to None (default relation translator).
   :type relation_translator: perturbationx.RelationTranslator, optional
   :param inplace: Whether to initialize the network in place. If False, the network is copied. Defaults to False.
   :type inplace: bool, optional
   :raises TypeError: If the relation translator is not a RelationTranslator.

   .. py:attribute:: __allowed_edge_types
      :value: ('core', 'boundary', 'infer')

      

   .. py:method:: from_networkx(graph: networkx.DiGraph, relation_translator: Optional[perturbationx.io.RelationTranslator] = None, inplace=False)
      :classmethod:

      Construct a new CausalNetwork from a networkx.DiGraph.

      :param graph: The network graph.
      :type graph: networkx.DiGraph
      :param relation_translator: The relation translator. Defaults to None (default relation translator).
      :type relation_translator: perturbationx.RelationTranslator, optional
      :param inplace: Whether to initialize the network in place. If False, the network is copied. Defaults to False.
      :type inplace: bool, optional
      :raises TypeError: If the relation translator is not a RelationTranslator.


   .. py:method:: from_dsv(filepath: str, edge_type='infer', delimiter='\t', header_cols=DEFAULT_DATA_COLS, relation_translator=None)
      :classmethod:

      Construct a new CausalNetwork from a delimited file.

      :param filepath: The path to the file.
      :type filepath: str
      :param edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
      :type edge_type: str, optional
      :param delimiter: The delimiter to use. Defaults to '   '.
      :type delimiter: str, optional
      :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
          "type"].
      :type header_cols: list, optional
      :param relation_translator: The relation translator. Defaults to None (default relation translator).
      :type relation_translator: perturbationx.RelationTranslator, optional
      :raises TypeError: If the relation translator is not a RelationTranslator.
      :return: The constructed CausalNetwork.
      :rtype: CausalNetwork


   .. py:method:: from_tsv(filepath: str, edge_type='infer', header_cols=DEFAULT_DATA_COLS, relation_translator=None)
      :classmethod:

      Construct a new CausalNetwork from a tab-delimited file.

      :param filepath: The path to the file.
      :type filepath: str
      :param edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
      :type edge_type: str, optional
      :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
          "type"].
      :type header_cols: list, optional
      :param relation_translator: The relation translator. Defaults to None (default relation translator).
      :type relation_translator: perturbationx.RelationTranslator, optional
      :raises TypeError: If the relation translator is not a RelationTranslator.
      :return: The constructed CausalNetwork.
      :rtype: CausalNetwork


   .. py:method:: from_csv(filepath: str, edge_type='infer', header_cols=DEFAULT_DATA_COLS, relation_translator=None)
      :classmethod:

      Construct a new CausalNetwork from a comma-delimited file.

      :param filepath: The path to the file.
      :type filepath: str
      :param edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
      :type edge_type: str, optional
      :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
          "type"].
      :type header_cols: list, optional
      :param relation_translator: The relation translator. Defaults to None (default relation translator).
      :type relation_translator: perturbationx.RelationTranslator, optional
      :raises TypeError: If the relation translator is not a RelationTranslator.
      :return: The constructed CausalNetwork.
      :rtype: CausalNetwork


   .. py:method:: from_cyjs_json(filepath: str, relation_translator=None)
      :classmethod:

      Construct a new CausalNetwork from a Cytoscape.js JSON file.

      :param filepath: The path to the file.
      :type filepath: str
      :param relation_translator: The relation translator. Defaults to None (default relation translator).
      :type relation_translator: perturbationx.RelationTranslator, optional
      :raises TypeError: If the relation translator is not a RelationTranslator.
      :return: The constructed CausalNetwork.
      :rtype: CausalNetwork


   .. py:method:: from_pandas(df: pandas.DataFrame, default_edge_type='infer', header_cols=DEFAULT_DATA_COLS, relation_translator=None)
      :classmethod:

      Construct a new CausalNetwork from a pandas DataFrame.

      :param df: The DataFrame to parse.
      :type df: pd.DataFrame
      :param default_edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
      :type default_edge_type: str, optional
      :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
          "type"].
      :type header_cols: list, optional
      :param relation_translator: The relation translator. Defaults to None (default relation translator).
      :type relation_translator: perturbationx.RelationTranslator, optional
      :raises TypeError: If the relation translator is not a RelationTranslator.
      :return: The constructed CausalNetwork.
      :rtype: CausalNetwork


   .. py:method:: add_edges_from_dsv(filepath: str, edge_type='infer', delimiter='\t', header_cols=DEFAULT_DATA_COLS)

      Add edges from a delimited file.

      :param filepath: The path to the file.
      :type filepath: str
      :param edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
      :type edge_type: str, optional
      :param delimiter: The delimiter to use. Defaults to '   '.
      :type delimiter: str, optional
      :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
          "type"].
      :type header_cols: list, optional


   .. py:method:: add_edges_from_tsv(filepath: str, edge_type='infer', header_cols=DEFAULT_DATA_COLS)

      Add edges from a tab-delimited file.

      :param filepath: The path to the file.
      :type filepath: str
      :param edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
      :type edge_type: str, optional
      :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
          "type"].
      :type header_cols: list, optional


   .. py:method:: add_edges_from_csv(filepath: str, edge_type='infer', header_cols=DEFAULT_DATA_COLS)

      Add edges from a comma-delimited file.

      :param filepath: The path to the file.
      :type filepath: str
      :param edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
      :type edge_type: str, optional
      :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
          "type"].
      :type header_cols: list, optional


   .. py:method:: add_edges_from_pandas(df: pandas.DataFrame, default_edge_type='infer', header_cols=DEFAULT_DATA_COLS)

      Add edges from a pandas DataFrame.

      :param df: The DataFrame to parse.
      :type df: pd.DataFrame
      :param default_edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
      :type default_edge_type: str, optional
      :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
          "type"].
      :type header_cols: list, optional


   .. py:method:: initialize_metadata()

      Initialize the metadata dictionary with default values. These values are "Untitled network" for the title
      and "Untitled collection" for the collection. The metadata dictionary is created if it does not exist.


   .. py:method:: copy()

      Return a copy of the CausalNetwork.

      :return: The copy.
      :rtype: CausalNetwork


   .. py:method:: number_of_nodes(typ=None)

      Return the number of nodes in the network.

      :param typ: The node type to count. If None, all nodes are counted. Defaults to None.
      :type typ: str, optional
      :return: The number of nodes.
      :rtype: int


   .. py:method:: nodes(typ=None, data=True)

      Return a list of nodes in the network.

      :param typ: The node type to return. If None, all nodes are returned. Defaults to None.
      :type typ: str, optional
      :param data: Whether to return the node data. Defaults to True.
      :type data: bool, optional
      :return: The list of nodes.
      :rtype: list


   .. py:method:: number_of_edges(typ=None)

      Return the number of edges in the network.

      :param typ: The edge type to count. If None, all edges are counted. Defaults to None.
      :type typ: str, optional
      :return: The number of edges.
      :rtype: int


   .. py:method:: edges(typ=None, data=True)

      Return a list of edges in the network.

      :param typ: The edge type to return. If None, all edges are returned. Defaults to None.
      :type typ: str, optional
      :param data: Whether to return the edge data. Defaults to True.
      :type data: bool, optional
      :return: The list of edges.
      :rtype: list


   .. py:method:: add_edge(src: str, trg: str, rel: str, typ='infer', confidence=None)

      Add an edge to the network. If the edge already exists, it is modified. If the nodes do not exist, they are
      created.

      :param src: The source node.
      :type src: str
      :param trg: The target node.
      :type trg: str
      :param rel: The causal relation of the edge.
      :type rel: str
      :param typ: The type of the edge. Allowed values are "core", "boundary" and "infer". Defaults to "infer".
      :type typ: str, optional
      :param confidence: The confidence of the edge. Defaults to None, which sets the confidence to 1.0.
      :type confidence: float, optional


   .. py:method:: modify_edge(src: str, trg: str, rel=None, typ=None, confidence=None)

      Modify an edge in the network. If the edge does not exist, a KeyError is raised.

      :param src: The source node.
      :type src: str
      :param trg: The target node.
      :type trg: str
      :param rel: The new causal relation of the edge. Defaults to None, which does not modify the relation.
      :type rel: str, optional
      :param typ: The new type of the edge. Allowed values are "core", "boundary" and "infer". Defaults to None,
          which does not modify the type.
      :type typ: str, optional
      :param confidence: The new confidence of the edge. Defaults to None, which does not modify the confidence.
      :type confidence: float, optional
      :raises KeyError: If the edge does not exist.


   .. py:method:: remove_edge(src: str, trg: str)

      Remove an edge from the network. If the edge does not exist, a KeyError is raised.

      :param src: The source node.
      :type src: str
      :param trg: The target node.
      :type trg: str
      :raises KeyError: If the edge does not exist.


   .. py:method:: modify_network(edge_list)

      Modify the network using a list of edges. The list of edges is a list of tuples of the form
      (source, target, relation, type). If the relation is None, the edge is removed. If the type is None, the type
      is not modified. If the type is not one of the allowed types, it is ignored or replaced with "infer".

      :param edge_list: The list of edges.
      :type edge_list: list


   .. py:method:: rewire_edges(nodes: list, iterations: int, datasets: list, method='k1', p_rate=1.0, missing_value_pruning_mode='nullify', opposing_value_pruning_mode=None, opposing_value_minimum_amplitude=1.0, boundary_edge_minimum=6, exact_boundary_outdegree=True, sparse=True, seed=None, verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS)

      Rewire edges in the network. The rewiring is performed on the given nodes, and the given number of
      iterations. Datasets can be provided to evaluate the rewired networks. If no datasets are provided, the
      rewired networks are returned as a list of modifications. Otherwise, the NPAs of the rewired networks are
      computed and returned.

      :param nodes: The nodes to rewire.
      :type nodes: list
      :param iterations: The number of rewiring iterations.
      :type iterations: int
      :param datasets: The datasets to use for evaluation. If None, the rewired networks are returned as a list of
          modifications. Otherwise, the NPAs of the rewired networks are computed and returned.
      :type datasets: list
      :param method: The rewiring method to use. May be "k1" or "k2". Defaults to "k1".
      :type method: str, optional
      :param p_rate: The fraction of edges to rewire. Defaults to 1.0.
      :type p_rate: float, optional
      :param missing_value_pruning_mode: The mode to use for pruning nodes with missing values. Must be one of
          'remove' or 'nullify'. Defaults to 'nullify'.
      :type missing_value_pruning_mode: str, optional
      :param opposing_value_pruning_mode: The mode to use for pruning edges with opposing values. Must be one of
          'remove', 'nullify', or 'none'. Defaults to None.
      :type opposing_value_pruning_mode: str, optional
      :param opposing_value_minimum_amplitude: The minimum amplitude of the dataset values to consider. Values with
          an absolute value smaller than this threshold are ignored. Defaults to 1.
      :type opposing_value_minimum_amplitude: float, optional
      :param boundary_edge_minimum: The minimum number of boundary edges a core node must have to be included
          in the pruned network. If a core node has fewer boundary edges after 'remove' pruning, all of its edges are
          removed. This parameter is ignored if 'nullify' pruning is used. Defaults to 6.
      :type boundary_edge_minimum: int, optional
      :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
          is set to 1 for all core nodes with boundary edges. Defaults to True.
      :type exact_boundary_outdegree: bool, optional
      :param sparse: Whether to use sparse computation. Defaults to True.
      :type sparse: bool, optional
      :param seed: The seed to use for the random number generator. Defaults to None.
      :type seed: int, optional
      :param verbose: Whether to log progress and network statistics. Defaults to True.
      :type verbose: bool, optional
      :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to "stream": sys.stdout,
          "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
      :type logging_kwargs: dict, optional
      :raises ValueError: If the permutation method is unknown. If the same node appears in both
          the core and boundary network.
      :return: The list of modifications, or a list of tuples of the form (modification, npa), where modification
          is the modification and npa is a dictionary of the form {dataset_id: npa}.
      :rtype: list


   .. py:method:: wire_edges(nodes: list, iterations: int, datasets: list, number_of_edges: int, edge_relations: list, missing_value_pruning_mode='nullify', opposing_value_pruning_mode=None, opposing_value_minimum_amplitude=1.0, boundary_edge_minimum=6, exact_boundary_outdegree=True, sparse=True, seed=None, verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS)

      Wire edges in the network. The wiring is performed on the given nodes, and the given number of
      iterations. Datasets can be provided to evaluate the wired networks. If no datasets are provided, the
      wired networks are returned as a list of modifications. Otherwise, the NPAs of the wired networks are
      computed and returned.

      :param nodes: The nodes to wire.
      :type nodes: list
      :param iterations: The number of wiring iterations.
      :type iterations: int
      :param datasets: The datasets to use for evaluation. If None, the wired networks are returned as a list of
          modifications. Otherwise, the NPAs of the wired networks are computed and returned.
      :type datasets: list
      :param number_of_edges: The number of edges to wire.
      :type number_of_edges: int
      :param edge_relations: The relations to use for wiring.
      :type edge_relations: list
      :param missing_value_pruning_mode: The mode to use for pruning nodes with missing values. Must be one of
          'remove' or 'nullify'. Defaults to 'nullify'.
      :type missing_value_pruning_mode: str, optional
      :param opposing_value_pruning_mode: The mode to use for pruning edges with opposing values. Must be one of
          'remove', 'nullify', or 'none'. Defaults to None.
      :type opposing_value_pruning_mode: str, optional
      :param opposing_value_minimum_amplitude: The minimum amplitude of the dataset values to consider. Values with
          an absolute value smaller than this threshold are ignored. Defaults to 1.
      :type opposing_value_minimum_amplitude: float, optional
      :param boundary_edge_minimum: The minimum number of boundary edges a core node must have to be included
          in the pruned network. If a core node has fewer boundary edges after 'remove' pruning, all of its edges are
          removed. This parameter is ignored if 'nullify' pruning is used. Defaults to 6.
      :type boundary_edge_minimum: int, optional
      :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
          is set to 1 for all core nodes with boundary edges. Defaults to True.
      :type exact_boundary_outdegree: bool, optional
      :param sparse: Whether to use sparse computation. Defaults to True.
      :type sparse: bool, optional
      :param seed: The seed to use for the random number generator. Defaults to None.
      :type seed: int, optional
      :param verbose: Whether to log progress and network statistics. Defaults to True.
      :type verbose: bool, optional
      :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to "stream": sys.stdout,
          "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
      :type logging_kwargs: dict, optional
      :raises ValueError: If the same node appears in both the core and boundary network.
      :return: The list of modifications, or a list of tuples of the form (modification, npa), where modification
          is the modification and npa is a dictionary of the form {dataset_id: npa}.
      :rtype: list


   .. py:method:: evaluate_modifications(modifications: list, nodes, datasets, missing_value_pruning_mode='nullify', opposing_value_pruning_mode=None, opposing_value_minimum_amplitude=1.0, boundary_edge_minimum=6, exact_boundary_outdegree=True, sparse=True, seed=None, verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS)

      Evaluate modifications of the network. The modifications are evaluated using the given datasets. The
      modifications are returned as a list of tuples of the form (modification, npa), where modification is the
      modification and npa is a dictionary of the form {dataset_id: npa}. Modifications can be generated using
      the rewire_edges or wire_edges methods.

      :param modifications: The modifications to evaluate. The modifications are a list of lists of tuples of the
          form (source, target, relation, type).
      :type modifications: list
      :param nodes: The nodes used in the modifications.
      :type nodes: list
      :param datasets: The datasets to use for evaluation.
      :type datasets: list
      :param missing_value_pruning_mode: The mode to use for pruning nodes with missing values. Must be one of
          'remove' or 'nullify'. Defaults to 'nullify'.
      :type missing_value_pruning_mode: str, optional
      :param opposing_value_pruning_mode: The mode to use for pruning edges with opposing values. Must be one of
          'remove', 'nullify', or 'none'. Defaults to None.
      :type opposing_value_pruning_mode: str, optional
      :param opposing_value_minimum_amplitude: The minimum amplitude of the dataset values to consider. Values with
          an absolute value smaller than this threshold are ignored. Defaults to 1.
      :type opposing_value_minimum_amplitude: float, optional
      :param boundary_edge_minimum: The minimum number of boundary edges a core node must have to be included
          in the pruned network. If a core node has fewer boundary edges after 'remove' pruning, all of its edges are
          removed. This parameter is ignored if 'nullify' pruning is used. Defaults to 6.
      :type boundary_edge_minimum: int, optional
      :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
          is set to 1 for all core nodes with boundary edges. Defaults to True.
      :type exact_boundary_outdegree: bool, optional
      :param sparse: Whether to use sparse computation. Defaults to True.
      :type sparse: bool, optional
      :param seed: The seed to use for the random number generator. Defaults to None.
      :type seed: int, optional
      :param verbose: Whether to log progress and network statistics. Defaults to True.
      :type verbose: bool, optional
      :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to "stream": sys.stdout,
          "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
      :type logging_kwargs: dict, optional
      :raises ValueError: If the same node appears in both the core and boundary network.
      :return: The list of modifications, or a list of tuples of the form (modification, npa), where modification
          is the modification and npa is a dictionary of the form {dataset_id: npa}.
      :rtype: list


   .. py:method:: infer_graph_attributes(inplace=False, verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS)

      Infer graph attributes required for NPA computation. The attributes are inferred from the relations in the
      network. The attributes are added to the graph as node and edge attributes. If inplace is True, the attributes
      are added to the graph and the graph is returned. Otherwise, the attributes are added to a copy of the graph
      and the copy is returned.

      :param inplace: Whether to add the attributes to the graph inplace. Defaults to False.
      :type inplace: bool, optional
      :param verbose: Whether to log progress and network statistics. Defaults to True.
      :type verbose: bool, optional
      :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to  "stream": sys.stdout,
          "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
      :type logging_kwargs: dict, optional
      :raises ValueError: If the same node appears in both the core and boundary network.
      :return: The graph with the inferred attributes.
      :rtype: CausalNetwork


   .. py:method:: get_adjacencies(sparse=False, verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS)

      Get the adjacency matrices of the network.

      :param sparse: Whether to use sparse matrices. Defaults to False.
      :type sparse: bool, optional
      :param verbose: Whether to log progress and network statistics. Defaults to True.
      :type verbose: bool, optional
      :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to  "stream": sys.stdout,
          "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
      :type logging_kwargs: dict, optional
      :raises ValueError: If the same node appears in both the core and boundary network.
      :return: The boundary and core adjacency matrices and the node ordering.
      :rtype: (np.ndarray | sp.sparray, np.ndarray | sp.sparray, list)


   .. py:method:: get_laplacians(boundary_outdegree_minimum=6, exact_boundary_outdegree=True, sparse=False, verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS)

      Get the Laplacian matrices of the network.

      :param boundary_outdegree_minimum: The minimum number of boundary edges a core node must have to be included
          in the boundary network. Defaults to 6.
      :type boundary_outdegree_minimum: int, optional
      :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
          is set to 1 for all core nodes with boundary edges. Defaults to True.
      :type exact_boundary_outdegree: bool, optional
      :param sparse: Whether to use sparse matrices. Defaults to False.
      :type sparse: bool, optional
      :param verbose: Whether to log progress and network statistics. Defaults to True.
      :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to  "stream": sys.stdout,
          "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
      :raises ValueError: If the same node appears in both the core and boundary network.
      :return: The Lb boundary Laplacian, Lc core Laplacian and Q core Laplacian and the node ordering.
      :rtype: (np.ndarray | sp.sparray, np.ndarray | sp.sparray, np.ndarray | sp.sparray, list)


   .. py:method:: toponpa(datasets: dict, missing_value_pruning_mode='nullify', opposing_value_pruning_mode=None, opposing_value_minimum_amplitude=1.0, boundary_edge_minimum=6, exact_boundary_outdegree=True, compute_statistics=True, alpha=0.95, permutations=('o', 'k2'), full_core_permutation=True, p_iters=500, p_rate=1.0, sparse=True, seed=None, verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS)

      Compute the Network Perturbation Amplitude (NPA) for a given network and datasets.

      :param datasets: The datasets to use. The keys are the dataset IDs and the values are the datasets, which are
          pandas DataFrames.
      :type datasets: dict
      :param missing_value_pruning_mode: The mode to use for pruning nodes with missing values. Must be one of
          'remove' or 'nullify'. Defaults to 'nullify'.
      :type missing_value_pruning_mode: str, optional
      :param opposing_value_pruning_mode: The mode to use for pruning edges with opposing values. Must be one of
          'remove', 'nullify', or 'none'. Defaults to None.
      :type opposing_value_pruning_mode: str, optional
      :param opposing_value_minimum_amplitude: The minimum amplitude of the dataset values to consider. Values with an
          absolute value smaller than this threshold are ignored. Defaults to 1.
      :type opposing_value_minimum_amplitude: float, optional
      :param boundary_edge_minimum: The minimum number of boundary edges a core node must have to be included in
          the pruned network. If a core node has fewer boundary edges after 'remove' pruning, all of its edges are
          removed. This parameter is ignored if 'nullify' pruning is used. Defaults to 6.
      :type boundary_edge_minimum: int, optional
      :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
          is set to 1 for all core nodes with boundary edges. Defaults to True.
      :type exact_boundary_outdegree: bool, optional
      :param compute_statistics: Whether to compute variances and confidence intervals. Defaults to True.
      :type compute_statistics: bool, optional
      :param alpha: The confidence level for the confidence intervals. Defaults to 0.95.
      :type alpha: float, optional
      :param permutations: The permutations to test. May contain 'o', 'k1', and 'k2' in any order.
          Defaults to ('o', 'k2').
      :type permutations: list, optional
      :param full_core_permutation: Whether to use the full permutation matrix for each core permutation. Partial
          permutations sample core coefficients, while full permutations sample perturbation scores. Defaults to True.
      :type full_core_permutation: bool, optional
      :param p_iters: The number of permutations to perform. Defaults to 500.
      :type p_iters: int, optional
      :param p_rate: The fraction of boundary coefficients to permute. Defaults to 1.
      :type p_rate: float, optional
      :param sparse: Whether to use sparse computation. Defaults to True.
      :type sparse: bool, optional
      :param seed: The seed for the random number generator. Defaults to None.
      :type seed: int, optional
      :param verbose: Whether to log progress and network statistics. Defaults to True.
      :type verbose: bool, optional
      :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to  "stream": sys.stdout,
          "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
      :type logging_kwargs: dict, optional
      :raises ValueError: If the same node appears in both the core and boundary network.
      :return: The NPA result.
      :rtype: perturbationx.NPAResult


   .. py:method:: to_networkx()

      Return a copy of the network as a NetworkX graph.

      :return: The NetworkX graph.
      :rtype: nx.DiGraph


   .. py:method:: to_edge_list(edge_type='all', data_cols=DEFAULT_DATA_COLS)

      Convert the network to a list of edges.

      :param edge_type: List of edge types to include. If "all", all edges will be included. Defaults to "all".
      :type edge_type: list | str, optional
      :param data_cols: data_cols: List of data columns to include. Defaults to ["subject", "object",
          "relation", "type"].
      :type data_cols: list, optional
      :return: A list of edges.
      :rtype: list


   .. py:method:: to_dsv(filepath, edge_type='all', delimiter='\t', data_cols=DEFAULT_DATA_COLS, header=None)

      Write the network to a delimited file.

      :param filepath: The path to write the file to.
      :type filepath: str
      :param edge_type: List of edge types to include. If "all", all edges will be included. Defaults to "all".
      :type edge_type: list | str, optional
      :param delimiter: The delimiter to use in the DSV file. Defaults to "   ".
      :type delimiter: str, optional
      :param data_cols: List of data columns to include. Columns not in ["subject", "object", "relation", "type"] will
          be ignored. Defaults to ["subject", "object", "relation", "type"].
      :type data_cols: list, optional
      :param header: List of header values to use. Must be given in the same order as data_cols. Defaults to None.
      :type header: list, optional
      :raises ValueError: If the length of the header list does not match the length of the data_cols list.


   .. py:method:: to_tsv(filepath, edge_type='all', data_cols=DEFAULT_DATA_COLS, header=None)

      Write the network to a tab-separated file.

      :param filepath: The path to write the file to.
      :type filepath: str
      :param edge_type: List of edge types to include. If "all", all edges will be included. Defaults to "all".
      :type edge_type: list | str, optional
      :param data_cols: List of data columns to include. Columns not in ["subject", "object", "relation", "type"] will
          be ignored. Defaults to ["subject", "object", "relation", "type"].
      :type data_cols: list, optional
      :param header: List of header values to use. Must be given in the same order as data_cols. Defaults to None.
      :type header: list, optional
      :raises ValueError: If the length of the header list does not match the length of the data_cols list.


   .. py:method:: to_csv(filepath, edge_type='all', data_cols=DEFAULT_DATA_COLS, header=None)

      Write the network to a comma-separated file.

      :param filepath: The path to write the file to.
      :type filepath: str
      :param edge_type: List of edge types to include. If "all", all edges will be included. Defaults to "all".
      :type edge_type: list | str, optional
      :param data_cols: List of data columns to include. Columns not in ["subject", "object", "relation", "type"] will
          be ignored. Defaults to ["subject", "object", "relation", "type"].
      :type data_cols: list, optional
      :param header: List of header values to use. Must be given in the same order as data_cols. Defaults to None.
      :type header: list, optional
      :raises ValueError: If the length of the header list does not match the length of the data_cols list.


   .. py:method:: to_cyjs_json(filepath, indent=4)

      Write the network to a Cytoscape.js JSON file.

      :param filepath: The path to write the file to.
      :type filepath: str
      :param indent: The indentation to use in the JSON file. Defaults to 4.
      :type indent: int, optional



.. py:class:: RelationTranslator(mappings=None, allow_numeric=True)


   Class for translating relations to numeric values. By default, relations "1", "increases", "-\>",
   "directlyIncreases", and "=\>" are mapped to 1.0, while relations "-1", "decreases", "-\|", "directlyDecreases",
   and "=\|" are mapped to -1.0. Relations that cannot be mapped to a numeric value will be parsed as 0.0.

   :param mappings: Dictionary of additional relation to numeric value mappings. It extends and overrides the default
       mappings.
   :type mappings: dict, optional
   :param allow_numeric: If True, relations will be parsed as numeric values if they cannot be found in the mappings
       dictionary. Defaults to True.
   :type allow_numeric: bool, optional

   .. py:method:: add_mapping(relation, maps_to)

      Add a new mapping from a relation to a numeric value.

      :param relation: The relation to map.
      :type relation: str
      :param maps_to: The numeric value to map to.
      :type maps_to: float


   .. py:method:: remove_mapping(relation)

      Remove a mapping from a relation to a numeric value.

      :param relation: The relation to remove the mapping for.
      :type relation: str


   .. py:method:: copy()

      Create a copy of this RelationTranslator.

      :return: A copy of this RelationTranslator.
      :rtype: RelationTranslator


   .. py:method:: translate(relation)

      Translate a relation to a numeric value.

      :param relation: The relation to translate.
      :type relation: str
      :return: The numeric value that the relation maps to.
      :rtype: float



.. py:class:: NPAResultBuilder(graph: networkx.DiGraph, datasets: list)


   Class for building NPAResult objects. Node attributes can be only passed for nodes in the graph with type "core".
   Each core node should have an attribute "idx" that is a unique integer index for that node. Node attributes should
   be passed as a list of values that is ordered by the node indices.

   :param graph: The graph used to generate the result.
   :type graph: networkx.DiGraph
   :param datasets: The datasets used to generate the result.
   :type datasets: list

   .. py:method:: new_builder(graph: networkx.DiGraph, datasets: list)
      :classmethod:

      Construct a new NPAResultBuilder object.
              


   .. py:method:: set_global_attributes(dataset_id: str, attributes: list, values: list)

      Set global attributes for a dataset. Attributes and values should be ordered in the same way.

      :param dataset_id: Dataset to set attributes for.
      :type dataset_id: str
      :param attributes: List of attribute names.
      :type attributes: list
      :param values: List of attribute values.
      :type values: list


   .. py:method:: set_node_attributes(dataset_id: str, attributes: list, values: list)

      Set node attributes for a dataset. Attributes and values should be ordered in the same way.
      Values should be passed as a nested list that is ordered by the node indices.

      :param dataset_id: Dataset to set attributes for.
      :type dataset_id: str
      :param attributes: List of attribute names.
      :type attributes: list
      :param values: List of attribute values.
      :type values: list


   .. py:method:: set_distribution(dataset_id: str, distribution: str, values: list, reference=None)

      Set a distribution for a dataset.

      :param dataset_id: Dataset to set distribution for.
      :type dataset_id: str
      :param distribution: Name of distribution.
      :type distribution: str
      :param values: List of values.
      :type values: list
      :param reference: Reference value for distribution. Defaults to None.
      :type reference: float, optional


   .. py:method:: build(metadata=None)

      Construct an NPAResult object.

      :param metadata: Metadata to include in the result. Defaults to None.
      :type metadata: dict, optional
      :return: NPAResult object.
      :rtype: perturbationx.NPAResult



.. py:class:: NPAResult(graph: networkx.DiGraph, datasets: list, global_info: pandas.DataFrame, node_info: pandas.DataFrame, distributions: dict, metadata=None)


   Class for storing and accessing the results of a Network Perturbation Analysis (NPA). It is recommended
   to build an NPAResult object using NPAResultBuilder to ensure correct formatting. Metadata is prefixed with
   "network\_" to avoid conflicts, unless the metadata key already starts with "network" or "dataset". By default,
   the following metadata is added: datetime_utc, python_implementation, python_version, system_name,
   system_release, system_version, network_title, network_collection, perturbationx_version, numpy_version,
   networkx_version, pandas_version, scipy_version, matplotlib_version, seaborn_version, and py4cytoscape_version.


   :param graph: The network graph.
   :type graph: networkx.DiGraph
   :param datasets: The datasets used for the analysis.
   :type datasets: list
   :param global_info: The global information for each dataset.
   :type global_info: pandas.DataFrame
   :param node_info: The node information for each dataset.
   :type node_info: pandas.DataFrame
   :param distributions: The distributions for each permutation.
   :type distributions: dict
   :param metadata: Additional metadata to store with the result.
   :type metadata: dict, optional

   .. py:method:: metadata()

      Get the metadata for this result.

      :return: The metadata for this result.
      :rtype: dict


   .. py:method:: datasets()

      Get the datasets used for this result.

      :return: The datasets used for this result.
      :rtype: list


   .. py:method:: node_attributes()

      Get the node attributes for this result.

      :return: The node attributes for this result.
      :rtype: list


   .. py:method:: distributions()

      Get the distributions for this result.

      :return: The distributions for this result.
      :rtype: list


   .. py:method:: global_info()

      Get the global information for this result.

      :return: The global information for this result.
      :rtype: pandas.DataFrame


   .. py:method:: node_info(accessor: str)

      Get the node information for this result.

      :param accessor: The dataset or node attribute to get the information for.
      :type accessor: str
      :return: The node information for this result.
      :rtype: pandas.DataFrame


   .. py:method:: get_distribution(distribution: str, dataset: str, include_reference=False)

      Get the distribution for a permutation.

      :param distribution: The permutation to get the distribution for.
      :type distribution: str
      :param dataset: The dataset to get the distribution for.
      :type dataset: str
      :param include_reference: If True, the reference value will be included in the distribution. Defaults to False.
      :type include_reference: bool, optional
      :return: The distribution for the permutation. If include_reference is True, a tuple of the distribution and
          the reference value will be returned.
      :rtype: list | (list, float)


   .. py:method:: plot_distribution(distribution: str, datasets=None, show=True)

      Plot the distribution for a permutation.

      :param distribution: The permutation to plot the distribution for.
      :type distribution: str
      :param datasets: The datasets to plot the distribution for. If None, all datasets will be plotted.
      :type datasets: list, optional
      :param show: If True, the plot will be shown. Defaults to True.
      :type show: bool, optional
      :return: The axes of the plot.
      :rtype: matplotlib.axes.Axes


   .. py:method:: get_leading_nodes(dataset: str, cutoff=0.8, attr='contribution', abs_value=True)

      Get the leading nodes for a dataset. The leading nodes are the nodes that contribute the most
      to a selected attribute, up to a certain cutoff.

      :param dataset: The dataset to get the leading nodes for.
      :type dataset: str
      :param cutoff: The cutoff for the cumulative distribution. Defaults to 0.8.
      :type cutoff: float, optional
      :param attr: The node attribute to get the leading nodes for. Defaults to "contribution".
      :type attr: str, optional
      :param abs_value: If True, the absolute value of the attribute will be used. Defaults to True.
      :type abs_value: bool, optional
      :return: The leading nodes for the dataset.
      :rtype: set


   .. py:method:: get_node_subgraph(nodes, include_shortest_paths='none', path_length_tolerance=0, include_neighbors=0, neighborhood_type='union')

      Get the subgraph for a set of nodes. The subgraph can include the shortest paths between the nodes,
      the neighborhood of the nodes, or both.

      :param nodes: The nodes to get the subgraph for.
      :type nodes: set
      :param include_shortest_paths: If "directed", the directed shortest paths between the nodes will be included.
          If "undirected", the undirected shortest paths between the nodes will be included. If "none",
          no shortest paths will be included. Defaults to "none".
      :type include_shortest_paths: str, optional
      :param path_length_tolerance: The tolerance for the length of the shortest paths. If 0, only the shortest paths
          are returned. If length_tolerance is an integer, it is interpreted as an absolute length. If
          length_tolerance is a float, it is interpreted as a percentage of the length of the shortest path.
          Defaults to 0.
      :type path_length_tolerance: int | float, optional
      :param include_neighbors: The maximum distance from leading nodes that neighbors can be. If 0, no neighbors
          will be included. If 1, only the neighbors of the nodes will be included. If 2, the neighbors of the
          neighbors of the nodes will be included, and so on. Defaults to 0.
      :type include_neighbors: int, optional
      :param neighborhood_type: The type of neighborhood to include. Can be one of "union" or "intersection".
          If "union", all nodes within the maximum distance from any leading node are returned. If "intersection",
          only nodes within the maximum distance from all leading nodes are returned. Defaults to "union".
      :type neighborhood_type: str, optional
      :raises ValueError: If include_shortest_paths is not "directed", "undirected", or "none".
          If max_distance is less than 0 or neighborhood_type is not "union" or "intersection".
          If length_tolerance is not a number or is negative.
      :return: The nodes and edges in the subgraph. They are returned as a pair of lists.
      :rtype: (list, list)


   .. py:method:: display_network(display_boundary=False, style=DEFAULT_STYLE, cytoscape_url=DEFAULT_BASE_URL)

      Display the network in Cytoscape.

      :param display_boundary: If True, boundary nodes will be displayed. Defaults to False.
      :type display_boundary: bool, optional
      :param style: The style to apply to the network. Defaults to DEFAULT_STYLE ("perturbationx-default").
      :type style: str, optional
      :param cytoscape_url: The URL of the Cytoscape instance to display the network in. Defaults to
          DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
      :type cytoscape_url: str, optional
      :return: The display object.
      :rtype: perturbationx.NPAResultDisplay


   .. py:method:: to_networkx()

      Retrieve the NetworkX graph for this result.

      :return: The NetworkX graph.
      :rtype: networkx.DiGraph


   .. py:method:: to_dict()

      Convert this result to a dictionary.

      :return: The result as a dictionary. Top-level keys are "metadata" and dataset names. For each dataset, the
          top-level keys are "global_info", "node_info", and "distributions".
      :rtype: dict


   .. py:method:: to_json(filepath: str, indent=4)

      Save this result to a JSON file. The format is the same as the output of to_dict().

      :param filepath: The path to save the result to.
      :type filepath: str
      :param indent: The indentation to use. Defaults to 4.
      :type indent: int, optional



.. py:class:: NPAResultDisplay(graph: networkx.Graph, results: perturbationx.result.NPAResult, network_style: str, network_suid: int, cytoscape_url=DEFAULT_BASE_URL)


   Class to display results from NPA analysis.

   :param graph: NetworkX graph object
   :type graph: nx.Graph
   :param results: NPA results object
   :type results: NPAResults
   :param network_style: Name of the network style to use
   :type network_style: str
   :param network_suid: SUID of the network to display results in.
   :type network_suid: int
   :param cytoscape_url: Cytoscape URL. Defaults to DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
   :type cytoscape_url: str, optional

   .. py:method:: reset_display(display_boundary: Optional[bool] = None, reset_color=False, reset_highlight=False, reset_visibility=False)

      Reset the display of the network.

      :param display_boundary: Whether to display the boundary of the network. Defaults to None, which does not
          change the current setting.
      :type display_boundary: bool, optional
      :param reset_color: Whether to reset the color of the nodes. Defaults to False.
      :type reset_color: bool, optional
      :param reset_highlight: Whether to reset the highlight of the nodes and edges. Defaults to False.
      :type reset_highlight: bool, optional
      :param reset_visibility: Whether to reset the visibility of the nodes and edges. Defaults to False.
      :type reset_visibility: bool, optional
      :raises CyError: If a CyREST error occurs.
      :return: SUID of the network.
      :rtype: int


   .. py:method:: color_nodes(dataset: str, attribute: str, gradient=DEFAULT_GRADIENT, default_color=DEFAULT_NODE_COLOR)

      Color nodes by a given attribute.

      :param dataset: The dataset to color nodes by.
      :type dataset: str
      :param attribute: The attribute to color nodes by.
      :type attribute: str
      :param gradient: The gradient to use. Defaults to DEFAULT_GRADIENT ("#2B80EF", "#EF3B2C").
      :type gradient: (str, str), optional
      :param default_color: The default color to use. Defaults to DEFAULT_NODE_COLOR ("#FEE391").
      :type default_color: str, optional
      :raises CyError: If a CyREST error occurs.
      :return: SUID of the network.
      :rtype: int


   .. py:method:: highlight_leading_nodes(dataset: str, cutoff=0.8, attr='contribution', abs_value=True, include_shortest_paths='none', path_length_tolerance=0, include_neighbors=0, neighborhood_type='union')

      Highlight leading nodes.

      :param dataset: The dataset to highlight leading nodes for.
      :type dataset: str
      :param cutoff: The cutoff to use when determining leading nodes. Defaults to 0.8.
      :type cutoff: float, optional
      :param attr: The attribute to use when determining leading nodes. Defaults to "contribution".
      :type attr: str, optional
      :param abs_value: Whether to use the absolute value of the attribute. Defaults to True.
      :type abs_value: bool, optional
      :param include_shortest_paths: If "directed", the directed shortest paths between the nodes will be included.
          If "undirected", the undirected shortest paths between the nodes will be included. If "none",
          no shortest paths will be included. Defaults to "none".
      :type include_shortest_paths: str, optional
      :param path_length_tolerance: The tolerance for the length of the shortest paths. If 0, only the shortest paths
          are returned. If length_tolerance is an integer, it is interpreted as an absolute length. If
          length_tolerance is a float, it is interpreted as a percentage of the length of the shortest path.
          Defaults to 0.
      :type path_length_tolerance: int | float, optional
      :param include_neighbors: The maximum distance from leading nodes that neighbors can be. If 0, no neighbors
          will be included. If 1, only the neighbors of the nodes will be included. If 2, the neighbors of the
          neighbors of the nodes will be included, and so on. Defaults to 0.
      :type include_neighbors: int, optional
      :param neighborhood_type: The type of neighborhood to include. Can be one of "union" or "intersection".
          If "union", all nodes within the maximum distance from any leading node are returned. If "intersection",
          only nodes within the maximum distance from all leading nodes are returned. Defaults to "union".
      :type neighborhood_type: str, optional
      :raises ValueError: If include_shortest_paths is not "directed", "undirected", or "none".
          If max_distance is less than 0 or neighborhood_type is not "union" or "intersection".
          If length_tolerance is not a number or is negative.
      :raises CyError: If a CyREST error occurs.
      :return: SUID of the network.
      :rtype: int


   .. py:method:: extract_leading_nodes(dataset: str, cutoff=0.8, attr='contribution', abs_value=True, inplace=True, include_shortest_paths='none', path_length_tolerance=0, include_neighbors=0, neighborhood_type='union')

      Extract leading nodes.

      :param dataset: The dataset to extract leading nodes for.
              :type dataset: str
      :param cutoff: The cutoff to use when determining leading nodes. Defaults to 0.8.
      :type cutoff: float, optional
      :param attr: The attribute to use when determining leading nodes. Defaults to "contribution".
      :type attr: str, optional
      :param abs_value: Whether to use the absolute value of the attribute. Defaults to True.
      :type abs_value: bool, optional
      :param inplace: Whether to extract the leading nodes in-place. Defaults to True. If True, the network will be
          modified by hiding all nodes and edges that are not leading nodes. If False, a new network will be created
          with only the leading nodes.
      :type inplace: bool, optional
      :param include_shortest_paths: If "directed", the directed shortest paths between the nodes will be included.
          If "undirected", the undirected shortest paths between the nodes will be included. If "none",
          no shortest paths will be included. Defaults to "none".
      :type include_shortest_paths: str, optional
      :param path_length_tolerance: The tolerance for the length of the shortest paths. If 0, only the shortest paths
          are returned. If length_tolerance is an integer, it is interpreted as an absolute length. If
          length_tolerance is a float, it is interpreted as a percentage of the length of the shortest path.
          Defaults to 0.
      :type path_length_tolerance: int | float, optional
      :param include_neighbors: The maximum distance from leading nodes that neighbors can be. If 0, no neighbors
          will be included. If 1, only the neighbors of the nodes will be included. If 2, the neighbors of the
          neighbors of the nodes will be included, and so on. Defaults to 0.
      :type include_neighbors: int, optional
      :param neighborhood_type: The type of neighborhood to include. Can be one of "union" or "intersection".
          If "union", all nodes within the maximum distance from any leading node are returned. If "intersection",
          only nodes within the maximum distance from all leading nodes are returned. Defaults to "union".
      :type neighborhood_type: str, optional
      :raises ValueError: If include_shortest_paths is not "directed", "undirected", or "none".
          If max_distance is less than 0 or neighborhood_type is not "union" or "intersection".
          If length_tolerance is not a number or is negative.
      :raises CyError: If a CyREST error occurs.
      :return: SUID of the network.
      :rtype: int


   .. py:method:: get_results()

      Retrieve the results object for this display.

      :return: The results object.
      :rtype: NPAResults



