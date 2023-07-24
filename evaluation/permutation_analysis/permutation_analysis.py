from collections import Counter

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from bnpa.io.network_io import read_dsv
from bnpa.io.RelationTranslator import RelationTranslator
from bnpa.npa.preprocess import preprocess_dataset, preprocess_network, network_matrices, permute_network
from bnpa.npa import core, permutation_tests


def scatter_plot(x, y, title, output_file=None, show_plot=False):
    plt.clf()
    fig, ax = plt.subplots(nrows=len(y), figsize=(6, 3 * len(y)), constrained_layout=True)
    plt.suptitle(title)

    if len(y) == 1:
        ax = [ax]

    for idx, d in enumerate(y):
        ax[idx].set_ylabel(d)
        ax[idx].scatter(x[:-1], y[d][:-1], alpha=0.5, c='blue')
        ax[idx].scatter(x[-1], y[d][-1], alpha=0.5, c='red')

    if output_file is not None:
        plt.savefig(output_file)
    if show_plot:
        plt.show()
    plt.close()


def degree_plot(datasets, title, output_file=None, show_plot=False):
    plt.clf()
    fig, ax = plt.subplots(nrows=len(datasets), figsize=(6, 3 * len(datasets)), constrained_layout=True)
    plt.suptitle(title)

    if len(datasets) == 1:
        ax = [ax]

    for idx, d in enumerate(datasets):
        ax[idx].set_ylabel(d)
        ax[idx].set_xscale('log')
        ax[idx].set_yscale('log')
        for x, y in datasets[d]:
            ax[idx].plot(x, y, '-o', alpha=0.5, c='blue')

    if output_file is not None:
        plt.savefig(output_file)
    if show_plot:
        plt.show()
    plt.close()


def build_graph(include_boundary=True):
    graph = nx.DiGraph()
    core_edges = read_dsv(
        "../data/NPANetworks/Hs_CFA_Apoptosis_backbone.tsv", default_edge_type="core",
        delimiter="\t", header_cols=["subject", "object", "relation"]
    )
    graph.add_edges_from(core_edges)

    if include_boundary:
        boundary_edges = read_dsv(
            "../data/NPANetworks/Hs_CFA_Apoptosis_downstream.tsv", default_edge_type="boundary",
            delimiter="\t", header_cols=["subject", "object", "relation"]
        )
        graph.add_edges_from(boundary_edges)

    return graph


def build_datasets(dataset_files):
    data = dict()
    for dataset_id in dataset_files:
        data[dataset_id] = pd.read_table("../data/COPD1/" + dataset_id + ".tsv")
        data[dataset_id] = data[dataset_id].rename(columns={"nodeLabel": "nodeID", "foldChange": "logFC"})
        data[dataset_id] = preprocess_dataset.format_dataset(data[dataset_id])

    return data


def measure_stats(nx_graph):
    stats = dict()
    stats["core_degrees"] = [d for _, d in nx_graph.degree]
    core_weights = [e[2]["weight"] for e in nx_graph.get_edges.data()]
    stats["neg_core_weights"] = sum(1 for w in core_weights if w < 0) / len(core_weights)

    undir_graph = nx_graph.to_undirected()
    stats["radius"] = nx.radius(undir_graph)
    stats["diameter"] = nx.diameter(undir_graph)
    stats["transitivity"] = nx.transitivity(undir_graph)
    stats["average_clustering"] = nx.average_clustering(undir_graph)
    stats["average_shortest_path_length"] = nx.average_shortest_path_length(undir_graph)
    stats["degree_assortativity_coefficient"] = nx.degree_assortativity_coefficient(undir_graph)
    return stats


def get_formatted_stat(statistics, stat_name):
    stat_k1 = [net_stats[stat_name] for net_stats in statistics["k1"]]
    stat_k1.append(statistics["true"][stat_name])
    stat_k2 = [net_stats[stat_name] for net_stats in statistics["k2"]]
    stat_k2.append(statistics["true"][stat_name])
    return stat_k1, stat_k2


def sorted_count(data):
    counts = Counter(data)
    key_list = sorted([k for k in counts])
    return key_list, [counts[k] for k in key_list]


def generate_plots(statistics, datasets, output_dir):
    # Extract npa results
    npa_k1 = dict()
    npa_k2 = dict()
    for dataset_id in datasets:
        npa_k1[dataset_id] = [net_stats[dataset_id] for net_stats in statistics["k1"]]
        npa_k1[dataset_id].append(statistics["true"][dataset_id])

        npa_k2[dataset_id] = [net_stats[dataset_id] for net_stats in statistics["k2"]]
        npa_k2[dataset_id].append(statistics["true"][dataset_id])

    # NPA vs neg core edge proportion (scatter)
    x_k1, x_k2 = get_formatted_stat(statistics, "neg_core_weights")
    scatter_plot(x_k1, npa_k1, "NPA vs. neg. core weights (k1)",
                 output_file=output_dir + "/npa_neg_core_weights_k1.png")
    scatter_plot(x_k2, npa_k2, "NPA vs. neg. core weights (k2)",
                 output_file=output_dir + "/npa_neg_core_weights_k2.png")

    # NPA vs radius (scatter)
    x_k1, x_k2 = get_formatted_stat(statistics, "radius")
    scatter_plot(x_k1, npa_k1, "NPA vs. radius (k1)", output_file=output_dir + "/npa_radius_k1.png")
    scatter_plot(x_k2, npa_k2, "NPA vs. radius (k2)", output_file=output_dir + "/npa_radius_k2.png")

    # NPA vs diameter (scatter)
    x_k1, x_k2 = get_formatted_stat(statistics, "diameter")
    scatter_plot(x_k1, npa_k1, "NPA vs. diameter (k1)", output_file=output_dir + "/npa_diameter_k1.png")
    scatter_plot(x_k2, npa_k2, "NPA vs. diameter (k2)", output_file=output_dir + "/npa_diameter_k2.png")

    # NPA vs transitivity (scatter)
    x_k1, x_k2 = get_formatted_stat(statistics, "transitivity")
    scatter_plot(x_k1, npa_k1, "NPA vs. transitivity (k1)", output_file=output_dir + "/npa_transitivity_k1.png")
    scatter_plot(x_k2, npa_k2, "NPA vs. transitivity (k2)", output_file=output_dir + "/npa_transitivity_k2.png")

    # NPA vs average_clustering (scatter)
    x_k1, x_k2 = get_formatted_stat(statistics, "average_clustering")
    scatter_plot(x_k1, npa_k1, "NPA vs. average clustering (k1)",
                 output_file=output_dir + "/npa_average_clustering_k1.png")
    scatter_plot(x_k2, npa_k2, "NPA vs. average clustering (k2)",
                 output_file=output_dir + "/npa_average_clustering_k2.png")

    # NPA vs average_shortest_path_length (scatter)
    x_k1, x_k2 = get_formatted_stat(statistics, "average_shortest_path_length")
    scatter_plot(x_k1, npa_k1, "NPA vs. average shortest path length (k1)",
                 output_file=output_dir + "/npa_average_shortest_path_length_k1.png")
    scatter_plot(x_k2, npa_k2, "NPA vs. average shortest path length (k2)",
                 output_file=output_dir + "/npa_average_shortest_path_length_k2.png")

    # NPA vs degree_assortativity_coefficient (scatter)
    x_k1, x_k2 = get_formatted_stat(statistics, "degree_assortativity_coefficient")
    scatter_plot(x_k1, npa_k1, "NPA vs. degree assortativity coefficient (k1)",
                 output_file=output_dir + "/npa_degree_assortativity_coefficient_k1.png")
    scatter_plot(x_k2, npa_k2, "NPA vs. degree assortativity coefficient (k2)",
                 output_file=output_dir + "/npa_degree_assortativity_coefficient_k2.png")

    datasets = {"true": [], "k1": [], "k2": []}
    datasets["true"].append(sorted_count(statistics["true"]["core_degrees"]))
    for net_stats in statistics["k1"]:
        datasets["k1"].append(sorted_count(net_stats["core_degrees"]))
    for net_stats in statistics["k2"]:
        datasets["k2"].append(sorted_count(net_stats["core_degrees"]))
    degree_plot(datasets, "Degree distribution", output_file=output_dir + "/degree_distribution.png")


if __name__ == "__main__":
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
    adj_perms = permute_network.permute_adjacency(
        adj_c, permutations=('k1', 'k2'), iterations=25, permutation_rate=1.
    )

    results = dict()
    results["true"] = (core_graph, dict())
    for perm in ('k1', 'k2'):
        results[perm] = []
        for adj in adj_perms[perm]:
            perm_graph = nx.from_numpy_array(adj)
            results[perm].append((nx.from_numpy_array(adj), dict()))

    for dataset_id in data:
        dataset = data[dataset_id]

        lap_b, dataset = preprocess_dataset.prune_network_dataset(
            graph, adj_b, dataset, dataset_id, False
        )
        lap_b, lap_c, lap_q, lap_perms = network_matrices.generate_laplacians(lap_b, adj_c, adj_perms)

        core_coefficients = core.value_inference(lap_b, lap_c, dataset["logFC"].to_numpy())
        npa, node_contributions = core.perturbation_amplitude_contributions(
            lap_q, core_coefficients, core_edge_count
        )

        distributions = permutation_tests.permutation_tests(
            lap_b, lap_c, lap_q, lap_perms, core_edge_count, dataset["logFC"].to_numpy(),
            ('k1', 'k2'), iterations=25, permutation_rate=1.
        )

        results["true"][1][dataset_id] = npa

        for perm in ('k1', 'k2'):
            for idx, res in enumerate(distributions[perm]):
                results[perm][idx][1][dataset_id] = res

    statistics = dict()
    true_stats = measure_stats(results["true"][0])
    true_stats.update(results["true"][1])
    statistics["true"] = true_stats

    for perm in ('k1', 'k2'):
        statistics[perm] = []
        for perm_graph, perm_npa in results[perm]:
            perm_stats = measure_stats(perm_graph)
            perm_stats.update(perm_npa)
            statistics[perm].append(perm_stats)

    generate_plots(statistics, dataset_files, "../output/perm_stats")
