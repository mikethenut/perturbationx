import json
from collections import Counter

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from bnpa.io.network_io import read_dsv
from bnpa.io.RelationTranslator import RelationTranslator


def scatter_plot_stacked(datasets, title, show_plot=True, output_file=None):
    plt.clf()
    fig, ax = plt.subplots(nrows=len(datasets), figsize=(6, 3 * len(datasets)), constrained_layout=True)
    plt.suptitle(title)

    if len(datasets) == 1:
        ax = [ax]

    for idx, d in enumerate(datasets):
        ax[idx].set_xscale('log')
        ax[idx].set_yscale('log')
        distr = datasets[d]
        ax[idx].set_ylabel(d)
        ax[idx].scatter(distr[0], distr[1], alpha=0.5, c='blue')

    if output_file is not None:
        plt.savefig(output_file)
    if show_plot:
        plt.show()
    plt.close()


def scatter_plot(x, y, x_label, y_label, title, show_plot=True, output_file=None):
    # add some noise to the data to show overlapping points
    rng = np.random.default_rng()
    x_range = max(x) - min(x)
    y_range = max(y) - min(y)
    x = [d + rng.uniform(-x_range / 100., x_range / 100.) for d in x]
    y = [d + rng.uniform(-y_range / 100., y_range / 100.) for d in y]

    plt.clf()
    plt.figure(figsize=(6, 4))
    plt.scatter(x[:-1], y[:-1], alpha=0.5)
    plt.scatter(x[-1], y[-1], alpha=0.5, c='black')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if output_file is not None:
        plt.savefig(output_file)
    if show_plot:
        plt.show()
    plt.close()


def generate_plots(network_stats, output_dir):
    # CORE STATISTICS
    networks = [n for n in network_stats if n != "Dr_ORG_Heart"]
    networks.append("Dr_ORG_Heart")

    # Number of edges vs. nodes (scatter)
    x = [network_stats[n]["core_node_count"] for n in networks]
    y = [network_stats[n]["core_edge_count"] for n in networks]
    scatter_plot(x, y, "Core node count", "Core edge count", "Core nodes vs. edges",
                 False, output_file=output_dir+"/core/core_node_edge.png")

    # radius vs. number of nodes (scatter)
    y = [network_stats[n]["radius"] for n in networks]
    scatter_plot(x, y, "Core node count", "Core radius", "Core nodes vs. radius",
                 False, output_file=output_dir+"/core/core_node_radius.png")

    # diameter vs. number of nodes (scatter)
    y = [network_stats[n]["diameter"] for n in networks]
    scatter_plot(x, y, "Core node count", "Core diameter", "Core nodes vs. diameter",
                 False, output_file=output_dir+"/core/core_node_diameter.png")

    # radius vs. number of edges (scatter)
    x = [network_stats[n]["core_edge_count"] for n in networks]
    y = [network_stats[n]["radius"] for n in networks]
    scatter_plot(x, y, "Core edge count", "Core radius", "Core edges vs. radius",
                 False, output_file=output_dir+"/core/core_edge_radius.png")

    # diameter vs. number of edges (scatter)
    y = [network_stats[n]["diameter"] for n in networks]
    scatter_plot(x, y, "Core edge count", "Core diameter", "Core edges vs. diameter",
                 False, output_file=output_dir+"/core/core_edge_diameter.png")

    # core degree distribution (scatter)
    datasets = dict()
    for n in networks:
        degrees = Counter(network_stats[n]["core_degrees"])
        datasets[n] = ([d for d in degrees], [degrees[d] for d in degrees])
    scatter_plot_stacked(datasets, "Core degree distribution", False,
                         output_file=output_dir+"/core/core_degree_distribution.png")

    # core out-degree distribution (scatter)
    datasets.clear()
    for n in networks:
        degrees = Counter(network_stats[n]["core_out_degrees"])
        datasets[n] = ([d for d in degrees], [degrees[d] for d in degrees])
    scatter_plot_stacked(datasets, "Core out-degree distribution", False,
                         output_file=output_dir+"/core/core_out_degree_distribution.png")

    # core in-degree distribution (scatter)
    datasets.clear()
    for n in networks:
        degrees = Counter(network_stats[n]["core_in_degrees"])
        datasets[n] = ([d for d in degrees], [degrees[d] for d in degrees])
    scatter_plot_stacked(datasets, "Core in-degree distribution", False,
                         output_file=output_dir+"/core/core_in_degree_distribution.png")

    # core weight distribution (scatter)
    x = [sum(1 for _ in network_stats[n]["core_weights"]) for n in networks]
    y = [sum(1 for w in network_stats[n]["core_weights"] if w < 0.) for n in networks]
    scatter_plot(x, y, "Core weights", "Negative core weights",
                 "Proportion of negative core weights", False, output_file=output_dir+"/core/core_weight.png")
    # BOUNDARY STATISTICS

    # inner boundary nodes vs. core nodes (scatter)
    x = [network_stats[n]["core_node_count"] for n in networks]
    y = [network_stats[n]["inner_boundary_node_count"] for n in networks]
    scatter_plot(x, y, "Core node count", "Inner boundary node count", "Core nodes vs. inner boundary nodes",
                 False, output_file=output_dir+"/boundary/core_node_inner_boundary_node.png")

    # outer boundary nodes vs. core nodes (scatter)
    y = [network_stats[n]["boundary_node_count"] for n in networks]
    scatter_plot(x, y, "Core node count", "Boundary node count", "Core nodes vs. boundary nodes",
                 False, output_file=output_dir+"/boundary/core_node_boundary_node.png")

    # outer boundary nodes vs. inner boundary nodes (scatter)
    x = [network_stats[n]["inner_boundary_node_count"] for n in networks]
    scatter_plot(x, y, "Inner boundary node count", "Boundary node count", "Inner boundary nodes vs. boundary nodes",
                 False, output_file=output_dir+"/boundary/inner_boundary_node_boundary_node.png")

    # boundary edges vs. core edges (scatter)
    x = [network_stats[n]["core_edge_count"] for n in networks]
    y = [network_stats[n]["boundary_edge_count"] for n in networks]
    scatter_plot(x, y, "Core edge count", "Boundary edge count", "Core edges vs. boundary edges",
                 False, output_file=output_dir+"/boundary/core_edge_boundary_edge.png")

    # boundary edges vs. boundary nodes (scatter)
    x = [network_stats[n]["boundary_node_count"] for n in networks]
    scatter_plot(x, y, "Boundary node count", "Boundary edge count", "Boundary nodes vs. boundary edges",
                 False, output_file=output_dir+"/boundary/boundary_node_boundary_edge.png")

    # boundary edges vs. inner boundary nodes (scatter)
    x = [network_stats[n]["inner_boundary_node_count"] for n in networks]
    scatter_plot(x, y, "Inner boundary node count", "Boundary edge count", "Inner boundary nodes vs. boundary edges",
                 False, output_file=output_dir+"/boundary/inner_boundary_node_boundary_edge.png")

    # boundary in-degree distribution (scatter)
    datasets.clear()
    for n in networks:
        degrees = Counter(network_stats[n]["boundary_in_degrees"])
        datasets[n] = ([d for d in degrees], [degrees[d] for d in degrees])
    scatter_plot_stacked(datasets, "Boundary in-degree distribution", False,
                         output_file=output_dir+"/boundary/boundary_in_degree_distribution.png")

    # boundary out-degree distribution (scatter)
    datasets.clear()
    for n in networks:
        degrees = Counter(network_stats[n]["boundary_out_degrees"])
        datasets[n] = ([d for d in degrees], [degrees[d] for d in degrees])
    scatter_plot_stacked(datasets, "Boundary out-degree distribution", False,
                         output_file=output_dir+"/boundary/boundary_out_degree_distribution.png")

    # boundary weight distribution (scatter)
    x = [sum(1 for w in network_stats[n]["boundary_weights"] if w > 0.) for n in networks]
    y = [sum(1 for w in network_stats[n]["boundary_weights"] if w < 0.) for n in networks]
    scatter_plot(x, y, "Positive boundary weights", "Negative boundary weights", "Boundary weights",
                 False, output_file=output_dir+"/boundary/boundary_weight.png")


def measure_network_stats():
    data_path = "../data/NPANetworks/"
    networks = ["Dr_ORG_Heart", "Hs_CFA_Apoptosis", "Hs_CPR_Cell_Cycle", "Hs_CPR_Jak_Stat", "Hs_CST_Oxidative_Stress",
                "Hs_CST_Xenobiotic_Metabolism", "Hs_IPN_Epithelial_Innate_Immune_Activation",
                "Hs_IPN_Neutrophil_Signaling", "Hs_TRA_ECM_Degradation", "Mm_CFA_Apoptosis", "Mm_CPR_Cell_Cycle",
                "Mm_CPR_Jak_Stat", "Mm_CST_Oxidative_Stress", "Mm_CST_Xenobiotic_Metabolism",
                "Mm_IPN_Epithelial_Innate_Immune_Activation", "Mm_IPN_Neutrophil_Signaling", "Mm_TRA_ECM_Degradation"]
    rt = RelationTranslator()

    all_stats = dict()
    for idx, network in enumerate(networks):
        print("Processing network " + str(idx + 1) + " of " + str(len(networks)) + ": " + network)
        n_stats = dict()
        all_stats[network] = n_stats

        core_edges = read_dsv(
            data_path + network + "_backbone.tsv", default_edge_type="core",
            delimiter="\t", header_cols=["subject", "object", "relation"]
        )
        boundary_edges = read_dsv(
            data_path + network + "_downstream.tsv", default_edge_type="boundary",
            delimiter="\t", header_cols=["subject", "object", "relation"]
        )

        core_graph = nx.DiGraph()
        core_graph.add_edges_from(core_edges)
        boundary_graph = nx.DiGraph()
        boundary_graph.add_edges_from(boundary_edges)
        full_graph = core_graph.copy()
        full_graph.add_edges_from(boundary_edges)

        n_stats["core_node_count"] = core_graph.number_of_nodes()
        n_stats["core_edge_count"] = core_graph.number_of_edges()
        n_stats["boundary_node_count"] = full_graph.number_of_nodes() - core_graph.number_of_nodes()
        n_stats["boundary_edge_count"] = full_graph.number_of_edges() - core_graph.number_of_edges()
        n_stats["core_degrees"] = [d for _, d in core_graph.degree]
        n_stats["core_in_degrees"] = [d for _, d in core_graph.in_degree]
        n_stats["core_out_degrees"] = [d for _, d in core_graph.out_degree]

        boundary_out_degree = []
        boundary_in_degree = []
        for n in full_graph.nodes:
            if n not in core_graph.nodes:
                boundary_in_degree.append(boundary_graph.in_degree[n])
            elif n in boundary_graph.nodes:
                boundary_out_degree.append(boundary_graph.out_degree[n])
            else:
                boundary_out_degree.append(0)

        n_stats["inner_boundary_node_count"] = sum([1 for od in boundary_out_degree if od > 0])
        n_stats["boundary_in_degrees"] = boundary_in_degree
        n_stats["boundary_out_degrees"] = boundary_out_degree

        core_weights = [rt.translate(e[2]["relation"]) for e in core_edges]
        boundary_weights = [rt.translate(e[2]["relation"]) for e in boundary_edges]

        n_stats["core_weights"] = core_weights
        n_stats["boundary_weights"] = boundary_weights

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

    with open("../output/network_stats/network_stats.json", "w") as out_file:
        json.dump(all_stats, out_file, indent=4)
    return all_stats


if __name__ == "__main__":
    # net_stats = measure_network_stats()
    with open("../output/network_stats/network_stats.json", "r") as in_file:
        net_stats = json.load(in_file)
    generate_plots(net_stats, "../output/network_stats")
