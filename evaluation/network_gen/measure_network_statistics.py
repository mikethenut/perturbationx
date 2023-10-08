import os
import json

import networkx as nx

from perturbationx.io.network_io import read_dsv
from perturbationx.io.RelationTranslator import RelationTranslator


def measure_network_stats(network_folder, output_file):
    networks = []
    for filename in os.listdir(network_folder):
        if not filename.startswith("50_24_"):
            continue

        if filename.endswith("_core.tsv") or filename.endswith("_backbone.tsv"):
            networks.append(filename)
            
    rt = RelationTranslator()
    statistics = dict()
    
    for idx, core_network in enumerate(networks):
        if core_network.endswith("_core.tsv"):
            network_name = core_network[:-9]
            boundary_network = network_name + "_boundary.tsv"
        elif core_network.endswith("_backbone.tsv"):
            network_name = core_network[:-13]
            boundary_network = network_name + "_downstream.tsv"
        else:
            raise Exception("Unexpected network filename: " + core_network)
        
        print("Processing network " + str(idx + 1) + " of " + str(len(networks)) + ": " + network_name)
        network_stats = dict()
        statistics[network_name] = network_stats

        core_edges = read_dsv(
            network_folder + core_network, default_edge_type="core",
            delimiter="\t", header_cols=["subject", "object", "relation"]
        )
        boundary_edges = read_dsv(
            network_folder + boundary_network, default_edge_type="boundary",
            delimiter="\t", header_cols=["subject", "object", "relation"]
        )

        core_graph = nx.DiGraph()
        core_graph.add_edges_from(core_edges)
        boundary_graph = nx.DiGraph()
        boundary_graph.add_edges_from(boundary_edges)
        full_graph = core_graph.copy()
        full_graph.add_edges_from(boundary_edges)

        network_stats["core_node_count"] = core_graph.number_of_nodes()
        network_stats["core_edge_count"] = core_graph.number_of_edges()
        network_stats["boundary_node_count"] = full_graph.number_of_nodes() - core_graph.number_of_nodes()
        network_stats["boundary_edge_count"] = full_graph.number_of_edges() - core_graph.number_of_edges()
        network_stats["core_degrees"] = [d for _, d in core_graph.degree]
        network_stats["core_in_degrees"] = [d for _, d in core_graph.in_degree]
        network_stats["core_out_degrees"] = [d for _, d in core_graph.out_degree]

        boundary_out_degree = []
        boundary_in_degree = []
        for n in full_graph.nodes:
            if n not in core_graph.nodes:
                boundary_in_degree.append(boundary_graph.in_degree[n])
            elif n in boundary_graph.nodes:
                boundary_out_degree.append(boundary_graph.out_degree[n])
            else:
                boundary_out_degree.append(0)

        network_stats["inner_boundary_node_count"] = sum([1 for od in boundary_out_degree if od > 0])
        network_stats["boundary_in_degrees"] = boundary_in_degree
        network_stats["boundary_out_degrees"] = boundary_out_degree

        core_weights = [rt.translate(e[2]["relation"]) for e in core_edges]
        boundary_weights = [rt.translate(e[2]["relation"]) for e in boundary_edges]

        network_stats["core_negative_edge_count"] = sum(1 for w in core_weights if w < 0.)
        network_stats["boundary_negative_edge_count"] = sum(1 for w in boundary_weights if w < 0.)

        if not nx.is_weakly_connected(core_graph):
            print("Warning: Core graph %s is not weakly connected!" % network_stats)
            wcc = sorted(nx.weakly_connected_components(core_graph), key=len, reverse=True)
            print("Number of weakly connected components: %d" % len(wcc))
            print("Size of largest weakly connected component: %d" % len(wcc[0]))
            print("Remaining nodes: %s" % str([n for n in core_graph.nodes if n not in wcc[0]]))
            continue

        core_graph = core_graph.to_undirected()
        network_stats["radius"] = nx.radius(core_graph)
        network_stats["diameter"] = nx.diameter(core_graph)
        network_stats["transitivity"] = nx.transitivity(core_graph)
        network_stats["average_clustering"] = nx.average_clustering(core_graph)
        network_stats["average_shortest_path_length"] = nx.average_shortest_path_length(core_graph)
        network_stats["degree_assortativity_coefficient"] = nx.degree_assortativity_coefficient(core_graph)

        boundary_graph = boundary_graph.to_undirected()
        network_stats["boundary_transitivity"] = nx.transitivity(boundary_graph)
        network_stats["boundary_average_clustering"] = nx.average_clustering(boundary_graph)
        network_stats["boundary_degree_assortativity_coefficient"] = nx.degree_assortativity_coefficient(boundary_graph)

    with open(output_file, "w") as out_file:
        json.dump(statistics, out_file, indent=4)
    return statistics


if __name__ == "__main__":
    network_path = "../../data/BAGen03b/"
    json_file = "../../output/ba_stats_03/ba_stats_02.json"
    measure_network_stats(network_path, json_file)
