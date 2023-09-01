import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from perturbationx.io.network_io import read_dsv
from perturbationx.io.RelationTranslator import RelationTranslator
from perturbationx.toponpa.preprocessing import preprocess_dataset, preprocess_network, network_matrices, permute_network
from perturbationx.toponpa import core
from perturbationx.result.NPAResultBuilder import NPAResultBuilder


def build_graph(include_boundary=True):
    graph = nx.DiGraph()
    core_edges = read_dsv(
        "../../data/NPANetworks/Hs_CFA_Apoptosis_backbone.tsv", default_edge_type="core",
        delimiter="\t", header_cols=["subject", "object", "relation"]
    )
    graph.add_edges_from(core_edges)

    if include_boundary:
        boundary_edges = read_dsv(
            "../../data/NPANetworks/Hs_CFA_Apoptosis_downstream.tsv", default_edge_type="boundary",
            delimiter="\t", header_cols=["subject", "object", "relation"]
        )
        graph.add_edges_from(boundary_edges)

    return graph


def build_datasets(dataset_files):
    data = dict()
    for dataset_id in dataset_files:
        data[dataset_id] = pd.read_table("../../data/COPD1/" + dataset_id + ".tsv")
        data[dataset_id] = data[dataset_id].rename(columns={"nodeLabel": "nodeID", "foldChange": "logFC"})
        data[dataset_id] = preprocess_dataset.format_dataset(data[dataset_id])

    return data


def perturbations_k1_k2():
    adj_perms = permute_network.permute_adjacency(
        adj_c, permutations=["k1", "k2"], iterations=iterations, permutation_rate=1.
    )

    results = {
        "true": dict(),
        "k1_partial": dict(),
        "k1_full": dict(),
        "k2_partial": dict(),
        "k2_full": dict()
    }
    for dataset_id in data:
        dataset = data[dataset_id]

        lap_b, dataset = preprocess_dataset.prune_network_dataset(
            graph, adj_b, dataset, dataset_id,
            missing_value_pruning_mode="nullify",
            opposing_value_pruning_mode=None,
            boundary_edge_minimum=6,
            verbose=False
        )
        lap_c, lap_q, _ = network_matrices.generate_core_laplacians(
            lap_b, adj_c, {},
            boundary_outdegree_type="continuous"
        )

        core_coefficients = core.value_inference(lap_b, lap_c, dataset["logFC"].to_numpy())
        npa, node_contributions = core.perturbation_amplitude_contributions(
            lap_q, core_coefficients, core_edge_count
        )
        results["true"][dataset_id] = npa

        for perm in ["k1", "k2"]:
            results[perm + "_partial"][dataset_id] = []
            results[perm + "_full"][dataset_id] = []

            for idx, adj_c_perm in enumerate(adj_perms[perm]):
                lap_c_perm, lap_q_perm, _ = network_matrices.generate_core_laplacians(
                    lap_b, adj_c_perm, {},
                    boundary_outdegree_type="continuous"
                )
                core_coefficients = core.value_inference(lap_b, lap_c_perm, dataset["logFC"].to_numpy())
                npa_partial = core.perturbation_amplitude(lap_q, core_coefficients, core_edge_count)
                npa_full = core.perturbation_amplitude(lap_q_perm, core_coefficients, core_edge_count)

                results[perm + "_partial"][dataset_id].append(npa_partial)
                results[perm + "_full"][dataset_id].append(npa_full)

    return results


if __name__ == "__main__":
    iterations = 500
    dataset_files = ["CS (2m) + Sham (3m)", "CS (2m) + Sham (5m)", "CS (4m) + Sham (1m)",
                     "CS (4m) + Sham (3m)", "CS (5m)", "CS (7m)"]
    data = build_datasets(dataset_files)
    graph = build_graph()
    core_graph = build_graph(include_boundary=False).to_undirected()
    rt = RelationTranslator()
    for src, trg, ed in core_graph.edges.data():
        core_graph[src][trg]["weight"] = rt.translate(ed["relation"])

    preprocess_network.infer_graph_attributes(graph)
    core_edge_count = sum(1 for src, trg in graph.edges if graph[src][trg]["type"] == "core")
    adj_b, adj_c = network_matrices.generate_adjacency(graph)

    results = perturbations_k1_k2()

    result_builder = NPAResultBuilder(graph, dataset_files)
    for dataset_id in dataset_files:
        for perm in ["k1_partial", "k1_full", "k2_partial", "k2_full"]:
            distribution = results[perm][dataset_id]
            result_builder.set_distribution(dataset_id, perm, distribution, results["true"][dataset_id])
    results = result_builder.build()

    for perm in ["k1_partial", "k1_full", "k2_partial", "k2_full"]:
        results.plot_distribution(perm, show=False)
        plt.savefig("distribution_plots/" + perm + ".png")
        plt.clf()
