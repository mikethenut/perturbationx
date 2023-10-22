:py:mod:`perturbationx.toponpa.preprocessing.preprocessing`
===========================================================

.. py:module:: perturbationx.toponpa.preprocessing.preprocessing


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.toponpa.preprocessing.preprocessing.format_dataset
   perturbationx.toponpa.preprocessing.preprocessing.prune_network_dataset
   perturbationx.toponpa.preprocessing.preprocessing.infer_graph_attributes



.. py:function:: format_dataset(dataset: pandas.DataFrame, computing_statistics=True)

   Format a dataset for use with toponpa.

   :param dataset: The dataset to format. Must contain columns 'nodeID' and 'logFC'. If computing_statistics is True,
       the dataset must also contain a column 'stderr' or 't'.
   :type dataset: pd.DataFrame
   :param computing_statistics: Whether statistics will be computed from the dataset. Defaults to True.
   :type computing_statistics: bool, optional
   :raises ValueError: If the dataset is not a pandas.DataFrame, or if it does not contain columns 'nodeID' and
       'logFC', or if computing_statistics is True and the dataset does not contain a column 'stderr' or 't'.
   :return: The formatted dataset.
   :rtype: pd.DataFrame


.. py:function:: prune_network_dataset(graph: networkx.DiGraph, adj_b: numpy.ndarray | scipy.sparse.sparray, dataset: pandas.DataFrame, dataset_id: str, missing_value_pruning_mode='nullify', opposing_value_pruning_mode=None, opposing_value_minimum_amplitude=1.0, boundary_edge_minimum=6, verbose=True)

   Prune a network and dataset to match each other.

   :param graph: The network to prune.
   :type graph: nx.DiGraph
   :param adj_b: The boundary adjacency matrix to prune.
   :type adj_b: np.ndarray | sp.sparray
   :param dataset: The dataset to use for pruning.
   :type dataset: pd.DataFrame
   :param dataset_id: The name of the dataset.
   :type dataset_id: str
   :param missing_value_pruning_mode: The mode to use for pruning nodes with missing values. Must be one of 'remove'
                                       or 'nullify'. Defaults to 'nullify'.
   :type missing_value_pruning_mode: str, optional
   :param opposing_value_pruning_mode: The mode to use for pruning edges with opposing values. Must be one of 'remove',
                                       'nullify', or 'none'. Defaults to None.
   :type opposing_value_pruning_mode: str, optional
   :param opposing_value_minimum_amplitude: The minimum amplitude of the dataset values to consider. Values with an
                                               absolute value smaller than this threshold are ignored. Defaults to 1.
   :type opposing_value_minimum_amplitude: float, optional
   :param boundary_edge_minimum: The minimum number of boundary edges a core node must have to be included
                                   in the pruned network. If a core node has fewer boundary edges after 'remove'
                                   pruning, all of its edges are removed. This parameter is ignored if 'nullify'
                                   pruning is used. Defaults to 6.
   :type boundary_edge_minimum: int, optional
   :param verbose: Whether to log network statistics.
   :type verbose: bool, optional
   :raises ValueError: If the missing value pruning mode is invalid, or if the opposing value pruning mode is invalid,
                           or if the boundary edge minimum is negative, or if the adjacency matrix is not
                           two-dimensional, or if the dataset does not contain any boundary nodes.
   :return: The pruned boundary adjacency matrix and the pruned dataset.
   :rtype: (np.ndarray | sp.sparray, pd.DataFrame)


.. py:function:: infer_graph_attributes(graph: networkx.DiGraph, relation_translator: Optional[perturbationx.io.RelationTranslator] = None, verbose=True)

   Infer attributes of a network and add them to the graph instance.

   :param graph: The network to process.
   :type graph: nx.DiGraph
   :param relation_translator: The relation translator to use. If None, a new instance will be created.
   :type relation_translator: perturbationx.RelationTranslator, optional
   :param verbose: Whether to log network statistics.
   :type verbose: bool, optional
   :raises ValueError: If the same node appears in both the core and boundary network.
   :return: The processed network.
   :rtype: nx.DiGraph


