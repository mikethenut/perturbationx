import logging

import networkx as nx

import perturbationx.util as util
from perturbationx.io import RelationTranslator
from perturbationx.result import NPAResultBuilder
from . import preprocessing, matrices, permutation, core, statistics

__all__ = ["toponpa", "evaluate_modifications"]


def toponpa(graph: nx.DiGraph, relation_translator: RelationTranslator, datasets: dict,
            missing_value_pruning_mode="nullify", opposing_value_pruning_mode=None, opposing_value_minimum_amplitude=1.,
            boundary_edge_minimum=6, exact_boundary_outdegree=True, compute_statistics=True,
            alpha=0.95, permutations=('o', 'k2'), full_core_permutation=True, p_iters=500,
            p_rate=1., sparse=True, seed=None, verbose=True):
    """Compute the Network Perturbation Amplitude (NPA) for a given network and datasets.

    :param graph: The network graph.
    :type graph: nx.DiGraph
    :param relation_translator: The relation translator.
    :type relation_translator: RelationTranslator
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
    :rtype: NPAResult
    """
    if verbose:
        logging.info("PREPROCESSING NETWORK")

    # Preprocess the datasets
    for dataset_id in datasets:
        datasets[dataset_id] = preprocessing.format_dataset(
            datasets[dataset_id], computing_statistics=compute_statistics
        )

    # Preprocess the graph
    preprocessing.infer_graph_attributes(graph, relation_translator, verbose)
    core_edge_count = sum(1 for src, trg in graph.edges if graph[src][trg]["type"] == "core")
    adj_b, adj_c = matrices.generate_adjacencies(graph, sparse=sparse)
    adj_perms = None if permutations is None \
        else permutation.permute_adjacency(
            adj_c, permutations=permutations, iterations=p_iters,
            permutation_rate=p_rate, seed=seed
        )

    result_builder = NPAResultBuilder.new_builder(graph, list(datasets.keys()))
    for dataset_id in datasets:
        dataset = datasets[dataset_id]
        if verbose:
            logging.info("COMPUTING NPA FOR DATASET '%s'" % dataset_id)

        # Prepare data
        lap_b, dataset = preprocessing.prune_network_dataset(
            graph, adj_b, dataset, dataset_id,
            missing_value_pruning_mode=missing_value_pruning_mode,
            opposing_value_pruning_mode=opposing_value_pruning_mode,
            opposing_value_minimum_amplitude=opposing_value_minimum_amplitude,
            boundary_edge_minimum=boundary_edge_minimum,
            verbose=verbose
        )
        lap_c, lap_q = matrices.generate_core_laplacians(
            lap_b, adj_c, exact_boundary_outdegree=exact_boundary_outdegree
        )

        # Compute NPA
        core_coefficients = core.coefficient_inference(lap_b, lap_c, dataset["logFC"].to_numpy())
        npa, node_contributions = core.perturbation_amplitude_contributions(
            lap_q, core_coefficients, core_edge_count
        )
        result_builder.set_global_attributes(dataset_id, ["NPA"], [npa])
        result_builder.set_node_attributes(dataset_id, ["contribution"], [node_contributions])
        result_builder.set_node_attributes(dataset_id, ["coefficient"], [core_coefficients])

        # Compute variances and confidence intervals
        if compute_statistics:
            npa_var, node_var = statistics.compute_variances(
                lap_b, lap_c, lap_q, dataset["stderr"].to_numpy(), core_coefficients, core_edge_count)
            npa_ci_lower, npa_ci_upper, _ = statistics.confidence_interval(npa, npa_var, alpha)
            result_builder.set_global_attributes(
                dataset_id, ["variance", "ci_lower", "ci_upper"], [npa_var, npa_ci_lower, npa_ci_upper]
            )
            node_ci_lower, node_ci_upper, node_p_value = \
                statistics.confidence_interval(core_coefficients, node_var, alpha)
            result_builder.set_node_attributes(
                dataset_id, ["variance", "ci_lower", "ci_upper", "p_value"],
                [node_var, node_ci_lower, node_ci_upper, node_p_value]
            )

        # Compute permutation test statistics
        if permutations is not None:
            distributions = statistics.test_permutations(
                lap_b=lap_b, lap_c=lap_c, lap_q=lap_q, adj_perms=adj_perms,
                core_edge_count=core_edge_count, boundary_coefficients=dataset["logFC"].to_numpy(),
                permutations=permutations, full_core_permutation=full_core_permutation,
                exact_boundary_outdegree=exact_boundary_outdegree, permutation_rate=p_rate,
                iterations=p_iters, seed=seed
            )

            for p in distributions:
                pv = statistics.p_value(npa, distributions[p])
                result_builder.set_global_attributes(dataset_id, ["%s_value" % p], [pv])
                result_builder.set_distribution(dataset_id, p, distributions[p], npa)

    args_metadata = {
        "missing_value_pruning_mode": missing_value_pruning_mode,
        "opposing_value_pruning_mode": opposing_value_pruning_mode,
        "opposing_value_minimum_amplitude": opposing_value_minimum_amplitude,
        "boundary_edge_minimum": boundary_edge_minimum,
        "exact_boundary_outdegree": exact_boundary_outdegree,
        "alpha": alpha,
        "full_core_permutation": full_core_permutation,
        "permutation_iterations": p_iters,
        "permutation_rate": p_rate,
        "seed": seed
    }
    return result_builder.build(args_metadata)


def evaluate_modifications(graph: nx.DiGraph, relation_translator: RelationTranslator, modifications: list,
                           nodes: list, datasets: dict,
                           missing_value_pruning_mode="nullify", opposing_value_pruning_mode=None,
                           opposing_value_minimum_amplitude=1., boundary_edge_minimum=6,
                           exact_boundary_outdegree=True, sparse=True, seed=None, verbose=True):
    """Evaluate the generated network modifications.

    :param graph: The network graph.
    :type graph: nx.DiGraph
    :param relation_translator: The relation translator.
    :type relation_translator: RelationTranslator
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
    """
    if verbose:
        logging.info("PREPROCESSING NETWORK")

    # Preprocess the datasets
    if datasets is not None:
        for dataset_id in datasets:
            datasets[dataset_id] = preprocessing.format_dataset(datasets[dataset_id])

    # Preprocess the graph
    preprocessing.infer_graph_attributes(graph, relation_translator, verbose=False)
    core_edge_count = sum(1 for src, trg in graph.edges if graph[src][trg]["type"] == "core")

    # Construct modified adjacency matrices
    adj_b, adj_c = matrices.generate_adjacencies(graph, sparse=sparse)
    adj_c_perms = [adj_c.copy() for _ in range(len(modifications))]
    rt = relation_translator if relation_translator is not None \
        else RelationTranslator()

    for modification, adj_c_perm in zip(modifications, adj_c_perms):
        edge_weights = []

        for src, trg, rel, _ in modification:
            src_idx = graph.nodes[src]["idx"]
            trg_idx = graph.nodes[trg]["idx"]

            weight = rt.translate(rel) if rel is not None else 0
            adj_c_perm[src_idx, trg_idx] = weight
            adj_c_perm[trg_idx, src_idx] = weight
            if weight > 0:
                edge_weights.append(weight)

        util.connect_adjacency_components(
            adj_c_perm, nodes, weights=edge_weights, seed=seed
        )

    # Compute NPAs for each dataset
    modifications = [(m, {}) for m in modifications]
    for dataset_id in datasets:
        dataset = datasets[dataset_id]
        if verbose:
            logging.info("COMPUTING NPA FOR DATASET '%s'" % dataset_id)

        # Prepare data
        lap_b, dataset = preprocessing.prune_network_dataset(
            graph, adj_b, dataset, dataset_id,
            missing_value_pruning_mode=missing_value_pruning_mode,
            opposing_value_pruning_mode=opposing_value_pruning_mode,
            opposing_value_minimum_amplitude=opposing_value_minimum_amplitude,
            boundary_edge_minimum=boundary_edge_minimum,
            verbose=verbose
        )

        for idx, adj_c_perm in enumerate(adj_c_perms):
            lap_c, lap_q = matrices.generate_core_laplacians(
                lap_b, adj_c_perm, exact_boundary_outdegree=exact_boundary_outdegree
            )

            # Compute NPA
            core_coefficients = core.coefficient_inference(lap_b, lap_c, dataset["logFC"].to_numpy())
            npa = core.perturbation_amplitude(lap_q, core_coefficients, core_edge_count)
            modifications[idx][1][dataset_id] = npa

    return modifications
