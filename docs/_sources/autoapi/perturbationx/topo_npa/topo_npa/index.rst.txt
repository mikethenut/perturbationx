:py:mod:`perturbationx.topo_npa.topo_npa`
=========================================

.. py:module:: perturbationx.topo_npa.topo_npa


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.toponpa.toponpa.toponpa
   perturbationx.toponpa.toponpa.evaluate_modifications



.. py:function:: toponpa(graph: networkx.DiGraph, relation_translator: perturbationx.io.RelationTranslator, datasets: dict, missing_value_pruning_mode='nullify', opposing_value_pruning_mode=None, opposing_value_minimum_amplitude=1.0, boundary_edge_minimum=6, exact_boundary_outdegree=True, compute_statistics=True, alpha=0.95, permutations=('o', 'k2'), full_core_permutation=True, p_iters=500, p_rate=1.0, sparse=True, seed=None, verbose=True)

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


