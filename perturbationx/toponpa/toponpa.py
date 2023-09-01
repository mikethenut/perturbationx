import logging

from perturbationx.io import RelationTranslator
from perturbationx.result import NPAResultBuilder
from . import preprocessing, matrices, permutation, core, statistics
import perturbationx.util as util

__all__ = ["toponpa", "evaluate_modifications"]


def toponpa(graph, relation_translator, datasets: dict, missing_value_pruning_mode="nullify",
            opposing_value_pruning_mode=None, opposing_value_minimum_amplitude=1.,
            boundary_edge_minimum=6, exact_boundary_outdegree=True, compute_statistics=True,
            alpha=0.95, permutations=('o', 'k2'), full_core_permutation=True, p_iters=500,
            p_rate=1., seed=None, verbose=True):
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
    adj_b, adj_c = matrices.generate_adjacencies(graph)
    if permutations is not None:
        adj_perms = permutation.permute_adjacency(
            adj_c, permutations=permutations, iterations=p_iters,
            permutation_rate=p_rate, seed=seed
        )
    else:
        adj_perms = {}

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
        lap_c, lap_q, lap_perms = matrices.generate_core_laplacians(
            lap_b, adj_c, adj_perms,
            exact_boundary_outdegree=exact_boundary_outdegree
        )

        # Compute NPA
        core_coefficients = core.value_inference(lap_b, lap_c, dataset["logFC"].to_numpy())
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
                lap_b, lap_c, lap_q, lap_perms, core_edge_count,
                dataset["logFC"].to_numpy(), permutations,
                iterations=p_iters, permutation_rate=p_rate, seed=seed
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


def evaluate_modifications(graph, relation_translator, modifications, nodes, datasets,
                           missing_value_pruning_mode="nullify", opposing_value_pruning_mode=None,
                           opposing_value_minimum_amplitude=1., boundary_edge_minimum=6,
                           exact_boundary_outdegree=True, seed=None, verbose=True):
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
    adj_b, adj_c = matrices.generate_adjacencies(graph)
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

        for idx, adj_c_perm in enumerate(adj_c_perms):
            # Prepare data
            lap_b, dataset = preprocessing.prune_network_dataset(
                graph, adj_b, dataset, dataset_id,
                missing_value_pruning_mode=missing_value_pruning_mode,
                opposing_value_pruning_mode=opposing_value_pruning_mode,
                opposing_value_minimum_amplitude=opposing_value_minimum_amplitude,
                boundary_edge_minimum=boundary_edge_minimum,
                verbose=verbose
            )
            lap_c, lap_q, _ = matrices.generate_core_laplacians(
                lap_b, adj_c, {},
                exact_boundary_outdegree=exact_boundary_outdegree
            )

            # Compute NPA
            core_coefficients = core.value_inference(lap_b, lap_c, dataset["logFC"].to_numpy())
            npa = core.perturbation_amplitude(lap_q, core_coefficients, core_edge_count)
            modifications[idx][1][dataset_id] = npa

    return modifications
