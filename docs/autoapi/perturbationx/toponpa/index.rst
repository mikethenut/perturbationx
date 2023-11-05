:py:mod:`perturbationx.toponpa`
===============================

.. py:module:: perturbationx.toponpa


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   permutation/index.rst
   preprocessing/index.rst
   statistics/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   core/index.rst
   matrices/index.rst
   toponpa/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.toponpa.permute_adjacency
   perturbationx.toponpa.permute_edge_list
   perturbationx.toponpa.format_dataset
   perturbationx.toponpa.prune_network_dataset
   perturbationx.toponpa.infer_graph_attributes
   perturbationx.toponpa.compute_variances
   perturbationx.toponpa.confidence_interval
   perturbationx.toponpa.test_permutations
   perturbationx.toponpa.p_value
   perturbationx.toponpa.coefficient_inference
   perturbationx.toponpa.perturbation_amplitude
   perturbationx.toponpa.perturbation_amplitude_contributions
   perturbationx.toponpa.generate_adjacencies
   perturbationx.toponpa.generate_boundary_laplacian
   perturbationx.toponpa.generate_core_laplacians
   perturbationx.toponpa.toponpa
   perturbationx.toponpa.evaluate_modifications



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


.. py:function:: coefficient_inference(lap_b: numpy.ndarray | scipy.sparse.sparray, lap_c: numpy.ndarray | scipy.sparse.sparray, boundary_coefficients: numpy.ndarray)

   Infer core coefficients from boundary coefficients and Laplacian matrices.

   :param lap_b: The Lb boundary Laplacian.
   :type lap_b: np.ndarray | sp.sparray
   :param lap_c: The Lc core Laplacian.
   :type lap_c: np.ndarray | sp.sparray
   :param boundary_coefficients: The boundary coefficients.
   :type boundary_coefficients: np.ndarray
   :raises ValueError: If the Laplacian matrices are misshapen or if the matrix dimensions do not match.
   :return: The inferred core coefficients.
   :rtype: np.ndarray


.. py:function:: perturbation_amplitude(lap_q: numpy.ndarray | scipy.sparse.sparray, core_coefficients: numpy.ndarray, core_edge_count: int)

   Compute the perturbation amplitude from the core Laplacian and core coefficients.

   :param lap_q: The Q core Laplacian.
   :type lap_q: np.ndarray | sp.sparray
   :param core_coefficients: The core coefficients.
   :type core_coefficients: np.ndarray
   :param core_edge_count: The number of edges in the core network.
   :type core_edge_count: int
   :raises ValueError: If the Laplacian matrix is misshapen or if the matrix dimensions do not match.
   :return: The perturbation amplitude.
   :rtype: np.ndarray


.. py:function:: perturbation_amplitude_contributions(lap_q: numpy.ndarray | scipy.sparse.sparray, core_coefficients: numpy.ndarray, core_edge_count: int)

   Compute the perturbation amplitude and relative contributions from the core Laplacian and core coefficients.

   :param lap_q: The Q core Laplacian.
   :type lap_q: np.ndarray | sp.sparray
   :param core_coefficients: The core coefficients.
   :type core_coefficients: np.ndarray
   :param core_edge_count: The number of edges in the core network.
   :type core_edge_count: int
   :raises ValueError: If the Laplacian matrix is misshapen or if the matrix dimensions do not match.
   :return: The perturbation amplitude and relative contributions.
   :rtype: (np.ndarray, np.ndarray)


.. py:function:: generate_adjacencies(graph: networkx.DiGraph, directed=False, sparse=True)

   Generate the boundary and core adjacency matrices from a graph.

   :param graph: The graph.
   :type graph: nx.DiGraph
   :param directed: Whether to generate directed adjacency matrices. Defaults to False.
   :type directed: bool, optional
   :param sparse: Whether to generate sparse adjacency matrices. Defaults to True.
   :type sparse: bool, optional
   :return: The boundary and core adjacency matrices.
   :rtype: (np.ndarray, np.ndarray) | (sp.sparray, sp.sparray)


.. py:function:: generate_boundary_laplacian(adj_b: numpy.ndarray | scipy.sparse.sparray, boundary_edge_minimum=6)

   Generate the boundary Lb Laplacian from a boundary adjacency matrix.

   :param adj_b: The boundary adjacency matrix.
   :type adj_b: np.ndarray | sp.sparray
   :param boundary_edge_minimum: The minimum number of boundary edges a core node must have to be
                                   included in the Lb Laplacian. Nodes with fewer boundary edges
                                   are removed from the Lb Laplacian. Defaults to 6.
   :type boundary_edge_minimum: int, optional
   :raises ValueError: If the adjacency matrix is misshapen or if the boundary edge minimum is negative.
   :return: The boundary Lb Laplacian.
   :rtype: np.ndarray | sp.sparray


.. py:function:: generate_core_laplacians(lap_b: numpy.ndarray | scipy.sparse.sparray, adj_c: numpy.ndarray | scipy.sparse.sparray, exact_boundary_outdegree=True)

   Generate the core Laplacians from a boundary Laplacian and core adjacency matrix.

   :param lap_b: The boundary Laplacian.
   :type lap_b: np.ndarray | sp.sparray
   :param adj_c: The core adjacency matrix.
   :type adj_c: np.ndarray | sp.sparray
   :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
                                       is set to 1 for all core nodes with boundary edges. Defaults to True.
   :type exact_boundary_outdegree: bool, optional
   :return: The core Laplacians.
   :rtype: (np.ndarray, np.ndarray) | (sp.sparray, sp.sparray)


.. py:function:: toponpa(graph: networkx.DiGraph, relation_translator: perturbationx.io.RelationTranslator, datasets: dict, missing_value_pruning_mode='nullify', opposing_value_pruning_mode=None, opposing_value_minimum_amplitude=1.0, boundary_edge_minimum=6, exact_boundary_outdegree=True, compute_statistics=True, alpha=0.95, permutations=('o', 'k2'), full_core_permutation=True, p_iters=500, p_rate=1.0, sparse=True, seed=None, verbose=True)
   :no-index:

   Compute the Network Perturbation Amplitude (NPA) for a given network and datasets.

   :param graph: The network graph.
   :type graph: nx.DiGraph
   :param relation_translator: The relation translator.
   :type relation_translator: perturbationx.RelationTranslator
   :param datasets: The datasets to use. The keys are the dataset IDs and the values are the datasets, which are
                       pandas DataFrames.
   :type datasets: dict
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
                                   permutations sample core coefficients, while full permutations sample perturbation
                                   scores. Defaults to True.
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
   :raises ValueError: If the same node appears in both the core and boundary network.
   :return: The NPA result.
   :rtype: perturbationx.NPAResult


.. py:function:: evaluate_modifications(graph: networkx.DiGraph, relation_translator: perturbationx.io.RelationTranslator, modifications: list, nodes: list, datasets: dict, missing_value_pruning_mode='nullify', opposing_value_pruning_mode=None, opposing_value_minimum_amplitude=1.0, boundary_edge_minimum=6, exact_boundary_outdegree=True, sparse=True, seed=None, verbose=True)

   Evaluate the generated network modifications.

   :param graph: The network graph.
   :type graph: nx.DiGraph
   :param relation_translator: The relation translator.
   :type relation_translator: perturbationx.RelationTranslator
   :param modifications: The list of modifications. Each modification is a list of tuples of the form
                           (source, target, relation, confidence).
   :type modifications: list
   :param nodes: The nodes that were modified.
   :type nodes: list
   :param datasets: The datasets to use. The keys are the dataset IDs and the values are the datasets, which are
                       pandas DataFrames.
   :type datasets: dict
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
   :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
                                       is set to 1 for all core nodes with boundary edges. Defaults to True.
   :type exact_boundary_outdegree: bool, optional
   :param sparse: Whether to use sparse computation. Defaults to True.
   :type sparse: bool, optional
   :param seed: The seed for the random number generator. Defaults to None.
   :type seed: int, optional
   :param verbose: Whether to log progress and network statistics. Defaults to True.
   :type verbose: bool, optional
   :raises ValueError: If the same node appears in both the core and boundary network.
   :return: List of tuples of the form (modification, npa), where modification is the modification and npa is a
               dictionary of the form {dataset_id: npa}.
   :rtype: list


