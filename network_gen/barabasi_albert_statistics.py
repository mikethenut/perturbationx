import os
import itertools
import json

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from bnpa.io.network_io import read_dsv
from bnpa.io.RelationTranslator import RelationTranslator


def histogram_plot(datasets, title, show_plot=True, output_file=None):
    plt.clf()
    fig, ax = plt.subplots(nrows=len(datasets), figsize=(6, 3 * len(datasets)), constrained_layout=True)
    plt.suptitle(title)

    if len(datasets) == 1:
        ax = [ax]

    for idx, d in enumerate(datasets):
        distr = datasets[d]
        ax[idx].set_ylabel(d)

        sns.histplot(distr, ax=ax[idx], color='lightblue', stat='density', bins=25)
        sns.kdeplot(distr, ax=ax[idx], color='navy')

    if output_file is not None:
        plt.savefig(output_file)
    if show_plot:
        plt.show()
    plt.close()


def scatter_plot(x_true, y_true, x_sample, y_sample, x_label, y_label, title, show_plot=True, output_file=None):
    # add some noise to the data to show overlapping points
    rng = np.random.default_rng()
    x_true_range = max(x_true) - min(x_true)
    y_true_range = max(y_true) - min(y_true)
    x_true = [d + rng.uniform(-x_true_range / 100., x_true_range / 100.) for d in x_true]
    y_true = [d + rng.uniform(-y_true_range / 100., y_true_range / 100.) for d in y_true]

    x_sample_range = max(x_sample) - min(x_sample)
    y_sample_range = max(y_sample) - min(y_sample)
    x_sample = [d + rng.uniform(-x_sample_range / 100., x_sample_range / 100.) for d in x_sample]
    y_sample = [d + rng.uniform(-y_sample_range / 100., y_sample_range / 100.) for d in y_sample]

    plt.clf()
    plt.figure(figsize=(6, 4))
    plt.scatter(x_sample, y_sample, alpha=0.5, c='green')
    plt.scatter(x_true[:-1], y_true[:-1], alpha=0.5, c='blue')
    plt.scatter(x_true[-1], y_true[-1], alpha=0.5, c='black')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if output_file is not None:
        plt.savefig(output_file)
    if show_plot:
        plt.show()
    plt.close()


def generate_plots(true_stats, sample_stats, output_dir):
    true_networks = [n for n in true_stats if n != "Dr_ORG_Heart"]
    true_networks.append("Dr_ORG_Heart")

    # CORE STATISTICS

    # Number of edges vs. nodes (scatter)
    x_true = [true_stats[n]["core_node_count"] for n in true_networks]
    y_true = [true_stats[n]["core_edge_count"] for n in true_networks]
    x_sample = [sample_stats[n]["core_node_count"] for n in sample_stats]
    y_sample = [sample_stats[n]["core_edge_count"] for n in sample_stats]
    scatter_plot(x_true, y_true, x_sample, y_sample, "Core node count", "Core edge count", "Core nodes vs. edges",
                 False, output_file=output_dir+"/core/core_node_edge_ba.png")

    # radius vs. number of nodes (scatter)
    y_true = [true_stats[n]["radius"] for n in true_networks]
    y_sample = [sample_stats[n]["radius"] for n in sample_stats]
    scatter_plot(x_true, y_true, x_sample, y_sample, "Core node count", "Core radius", "Core nodes vs. radius",
                 False, output_file=output_dir+"/core/core_node_radius_ba.png")

    # diameter vs. number of nodes (scatter)
    y_true = [true_stats[n]["diameter"] for n in true_networks]
    y_sample = [sample_stats[n]["diameter"] for n in sample_stats]
    scatter_plot(x_true, y_true, x_sample, y_sample, "Core node count", "Core diameter", "Core nodes vs. diameter",
                 False, output_file=output_dir+"/core/core_node_diameter_ba.png")

    # radius vs. number of edges (scatter)
    x_true = [true_stats[n]["core_edge_count"] for n in true_networks]
    y_true = [true_stats[n]["radius"] for n in true_networks]
    x_sample = [sample_stats[n]["core_edge_count"] for n in sample_stats]
    y_sample = [sample_stats[n]["radius"] for n in sample_stats]
    scatter_plot(x_true, y_true, x_sample, y_sample, "Core edge count", "Core radius", "Core edges vs. radius",
                 False, output_file=output_dir+"/core/core_edge_radius_ba.png")

    # diameter vs. number of edges (scatter)
    y_true = [true_stats[n]["diameter"] for n in true_networks]
    y_sample = [sample_stats[n]["diameter"] for n in sample_stats]
    scatter_plot(x_true, y_true, x_sample, y_sample, "Core edge count", "Core diameter", "Core edges vs. diameter",
                 False, output_file=output_dir+"/core/core_edge_diameter_ba.png")

    # core weight distribution (scatter)
    x_true = [sum(1 for _ in true_stats[n]["core_weights"]) for n in true_networks]
    y_true = [sum(1 for w in true_stats[n]["core_weights"] if w < 0.) for n in true_networks]
    x_sample = [sum(1 for _ in sample_stats[n]["core_weights"]) for n in sample_stats]
    y_sample = [sum(1 for w in sample_stats[n]["core_weights"] if w < 0.) for n in sample_stats]
    scatter_plot(x_true, y_true, x_sample, y_sample, "Core weights", "Negative core weights",
                 "Proportion of negative core weights", False, output_file=output_dir+"/core/core_weight_ba.png")

    # core degree distribution (histogram)
    datasets = dict()
    datasets["NPA-R"] = list(itertools.chain.from_iterable(
        [true_stats[n]["core_degrees"] for n in true_networks]
    ))
    datasets["Barabasi-Albert"] = list(itertools.chain.from_iterable(
        [sample_stats[n]["core_degrees"] for n in sample_stats]
    ))
    histogram_plot(datasets, "Core degree distribution", False,
                   output_file=output_dir+"/core/core_degree_distribution_ba.png")

    # core out-degree distribution (histogram)
    datasets.clear()
    datasets["NPA-R"] = list(itertools.chain.from_iterable(
        [true_stats[n]["core_out_degrees"] for n in true_networks]
    ))
    datasets["Barabasi-Albert"] = list(itertools.chain.from_iterable(
        [sample_stats[n]["core_out_degrees"] for n in sample_stats]
    ))
    histogram_plot(datasets, "Core out-degree distribution", False,
                   output_file=output_dir+"/core/core_out_degree_distribution_ba.png")

    # core in-degree distribution (histogram)
    datasets.clear()
    datasets["NPA-R"] = list(itertools.chain.from_iterable(
        [true_stats[n]["core_in_degrees"] for n in true_networks]
    ))
    datasets["Barabasi-Albert"] = list(itertools.chain.from_iterable(
        [sample_stats[n]["core_in_degrees"] for n in sample_stats]
    ))
    histogram_plot(datasets, "Core in-degree distribution", False,
                   output_file=output_dir+"/core/core_in_degree_distribution_ba.png")


def measure_network_stats():
    data_path = "../data/BAGen01/"
    networks = [filename[:-9] for filename in os.listdir(data_path) if filename.endswith("_core.tsv")]
    rt = RelationTranslator()

    all_stats = dict()
    for idx, network in enumerate(networks):
        print("Processing network " + str(idx + 1) + " of " + str(len(networks)) + ": " + network)
        n_stats = dict()
        all_stats[network] = n_stats

        core_edges = read_dsv(
            data_path + network + "_core.tsv", default_edge_type="core",
            delimiter="\t", header_cols=["subject", "object", "relation"]
        )
        core_graph = nx.DiGraph()
        core_graph.add_edges_from(core_edges)

        n_stats["core_node_count"] = core_graph.number_of_nodes()
        n_stats["core_edge_count"] = core_graph.number_of_edges()
        n_stats["core_degrees"] = [d for _, d in core_graph.degree]
        n_stats["core_in_degrees"] = [d for _, d in core_graph.in_degree]
        n_stats["core_out_degrees"] = [d for _, d in core_graph.out_degree]

        core_weights = [rt.translate(e[2]["relation"]) for e in core_edges]
        n_stats["core_weights"] = core_weights

        if not nx.is_weakly_connected(core_graph):
            print("Warning: Core graph %s is not weakly connected!" % network)
            wcc = sorted(nx.weakly_connected_components(core_graph), key=len, reverse=True)
            print("Number of weakly connected components: %d" % len(wcc))
            print("Size of largest weakly connected component: %d" % len(wcc[0]))
            print("Remaining nodes: %s" % str([n for n in core_graph.nodes if n not in wcc[0]]))
            continue

        core_graph = core_graph.to_undirected()
        n_stats["radius"] = nx.radius(core_graph)
        n_stats["diameter"] = nx.diameter(core_graph)

    return all_stats


if __name__ == "__main__":
    ba_sample_stats = measure_network_stats()
    with open("../output/network_stats/network_stats.json", "r") as in_file:
        npa_stats = json.load(in_file)

    generate_plots(npa_stats, ba_sample_stats, "../output/network_stats/")
