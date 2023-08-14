import json
import os
import sys
import logging

import pandas as pd
import networkx as nx

from bnpa.CausalNetwork import CausalNetwork
from bnpa.toponpa.preprocessing import preprocess_dataset, network_matrices, permute_network
from bnpa.toponpa import core


def create_networkx_from_adjacency(adjacency, nodes):
    pgraph = nx.from_numpy_array(adjacency)
    for idx, node in enumerate(nodes):
        pgraph.nodes[idx]["label"] = node
    return pgraph


def graph_edit_distance(
        g1, g2, edge_insertion_cost=1.,
        edge_deletion_cost=1., edge_substitution_cost=1.):
    ged = 0.
    for edge in g1.edges:
        if not g2.has_edge(*edge):
            ged += edge_deletion_cost
        elif (g1[edge[0]][edge[1]]["weight"]
              != g2[edge[0]][edge[1]]["weight"]):
            ged += edge_substitution_cost

    for edge in g2.edges:
        if not g1.has_edge(*edge):
            ged += edge_insertion_cost

    return ged


def test_copd1(permutations, permutation_rates, repetitions, out_file):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(levelname)s -- %(message)s")

    network_file = "../../data/NPANetworks/Mm_CFA_Apoptosis"
    core_suffix = "_backbone.tsv"
    boundary_suffix = "_downstream.tsv"

    core_file = network_file + core_suffix
    boundary_file = network_file + boundary_suffix

    causalbionet = CausalNetwork.from_tsv(core_file, edge_type="core")
    causalbionet.add_edges_from_tsv(boundary_file, edge_type="boundary")
    causalbionet.infer_graph_attributes(inplace=True)
    core_edge_count = causalbionet.number_of_edges(typ="core")

    graph = causalbionet.to_networkx()
    adj_b, adj_c = network_matrices.generate_adjacency(graph)
    core_nodes = sorted([node for node in graph.nodes
                         if graph.nodes[node]["type"] == "core"],
                        key=lambda x: graph.nodes[x]["idx"])
    base_graph = create_networkx_from_adjacency(adj_c, core_nodes)

    adj_perms = dict()
    for perm in permutations:
        adj_perms[perm] = dict()
        for perm_rate in permutation_rates:
            adj_perms[perm][perm_rate] = (
                permute_network.permute_adjacency(
                    adj_c, permutations=(perm,),
                    iterations=repetitions,
                    permutation_rate=perm_rate
                ))[perm]

    datasets_folder = "../../data/COPD1/"
    datasets = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".tsv"):
            dataset = pd.read_table(datasets_folder + file_name, sep="\t")
            dataset_name = file_name.split(".")[0]
            dataset.rename(columns={"nodeLabel": "nodeID", "foldChange": "logFC"}, inplace=True)

            dataset_lap_b, dataset = preprocess_dataset.prune_network_dataset(
                graph, adj_b, dataset, dataset_name,
                missing_value_pruning_mode="nullify",
                opposing_value_pruning_mode=None,
                boundary_edge_minimum=6,
                verbose=False
            )
            datasets[dataset_name] = (dataset, dataset_lap_b)

    true_results = dict()
    for data_id in datasets:
        dataset, dataset_lap_b = datasets[data_id]
        lap_c, lap_q, _ = network_matrices.generate_core_laplacians(
            dataset_lap_b, adj_c, {},
            boundary_outdegree_type="continuous"
        )

        core_coefficients = core.value_inference(dataset_lap_b, lap_c, dataset["logFC"].to_numpy())
        npa, node_contributions = core.perturbation_amplitude_contributions(
            lap_q, core_coefficients, core_edge_count
        )

        true_results[data_id] = npa

    results = [{
        "permutation": None,
        "permutation_rate": 0.,
        "ged": 0.,
        "npa": true_results
    }]
    for perm in permutations:
        for perm_rate in permutation_rates:
            logging.info("Evaluating permutation {}/{}".format(perm, perm_rate))
            adj_c_perms = adj_perms[perm][perm_rate]

            for adj_c_perm in adj_c_perms:
                perm_graph = create_networkx_from_adjacency(adj_c_perm, core_nodes)
                perm_ged = graph_edit_distance(base_graph, perm_graph)
                partial_perm_results = dict()
                full_perm_results = dict()

                for data_id in datasets:
                    dataset, dataset_lap_b = datasets[data_id]
                    lap_c, lap_q, _ = network_matrices.generate_core_laplacians(
                        dataset_lap_b, adj_c, {},
                        boundary_outdegree_type="continuous"
                    )

                    lap_c_perm, lap_q_perm, _ = network_matrices.generate_core_laplacians(
                        dataset_lap_b, adj_c_perm, {},
                        boundary_outdegree_type="continuous"
                    )
                    core_coefficients = core.value_inference(dataset_lap_b, lap_c_perm, dataset["logFC"].to_numpy())
                    npa_partial = core.perturbation_amplitude(lap_q, core_coefficients, core_edge_count)
                    npa_full = core.perturbation_amplitude(lap_q_perm, core_coefficients, core_edge_count)

                    partial_perm_results[data_id] = npa_partial
                    full_perm_results[data_id] = npa_full

                results.append({
                    "permutation": perm + "_partial",
                    "permutation_rate": perm_rate,
                    "ged": perm_ged,
                    "npa": partial_perm_results
                })
                results.append({
                    "permutation": perm + "_full",
                    "permutation_rate": perm_rate,
                    "ged": perm_ged,
                    "npa": full_perm_results
                })

    json.dump(results, open(out_file, "w"), indent=4)


def test_generated_data(network_folder, core_suffix, boundary_suffix,
                        permutations, permutation_rates, repetitions,
                        out_file, network_selection=None):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(levelname)s -- %(message)s")
    networks = dict()
    for file_name in os.listdir(network_folder):
        if file_name.endswith(core_suffix) and not file_name.startswith("Hs_CST_Xenobiotic"):
            if network_selection is not None and file_name[:-len(core_suffix)] not in network_selection:
                continue

            network_name = file_name[:-len(core_suffix)]
            core_file = network_folder + network_name + core_suffix
            boundary_file = network_folder + network_name + boundary_suffix

            causalbionet = CausalNetwork.from_tsv(core_file, edge_type="core")
            causalbionet.add_edges_from_tsv(boundary_file, edge_type="boundary")
            causalbionet.infer_graph_attributes(inplace=True, verbose=False)
            core_edge_count = causalbionet.number_of_edges(typ="core")

            graph = causalbionet.to_networkx()
            adj_b, adj_c = network_matrices.generate_adjacency(graph)
            core_nodes = sorted([node for node in graph.nodes
                                 if graph.nodes[node]["type"] == "core"],
                                key=lambda x: graph.nodes[x]["idx"])
            base_graph = create_networkx_from_adjacency(adj_c, core_nodes)

            networks[network_name] = (
                causalbionet, core_nodes, core_edge_count,
                graph, base_graph, adj_b, adj_c
            )

    datasets_folder = "../../data/ExpressionExamplesGen02/"
    datasets = {}
    dataset_types = {}
    for file_name in os.listdir(datasets_folder):
        dataset_network = None
        for network_name in networks:
            if file_name.startswith(network_name):
                dataset_network = network_name
                break

        dataset_type = None
        if "(1)" in file_name:
            dataset_type = 1
        elif "(0)" in file_name:
            dataset_type = 0
        elif "(-1)" in file_name:
            dataset_type = -1

        if dataset_network is None or dataset_type is None:
            continue

        dataset = pd.read_table(datasets_folder + file_name, sep=",")
        dataset_name = "[%d]_%s" % (dataset_type, file_name.split("_dataset_")[1][:8])
        if dataset_network not in datasets:
            datasets[dataset_network] = {}
        dataset_types[dataset_name] = dataset_type

        causalbionet, core_nodes, core_edge_count, graph, \
            base_graph, adj_b, adj_c = networks[dataset_network]

        dataset_lap_b, dataset = preprocess_dataset.prune_network_dataset(
            graph, adj_b, dataset, dataset_name,
            missing_value_pruning_mode="nullify",
            opposing_value_pruning_mode=None,
            boundary_edge_minimum=6,
            verbose=False
        )
        datasets[dataset_network][dataset_name] = (dataset, dataset_lap_b)

    results = []
    for network_name in networks:
        causalbionet, core_nodes, core_edge_count, graph, \
            base_graph, adj_b, adj_c = networks[network_name]

        true_results = dict()
        for data_id in datasets[network_name]:
            dataset, dataset_lap_b = datasets[network_name][data_id]
            lap_c, lap_q, _ = network_matrices.generate_core_laplacians(
                dataset_lap_b, adj_c, {},
                boundary_outdegree_type="continuous"
            )

            core_coefficients = core.value_inference(dataset_lap_b, lap_c, dataset["logFC"].to_numpy())
            npa, node_contributions = core.perturbation_amplitude_contributions(
                lap_q, core_coefficients, core_edge_count
            )

            true_results[data_id] = npa

        results.append({
            "network": network_name,
            "permutation": None,
            "permutation_rate": 0,
            "ged": 0,
            "npa": true_results
        })

    for network_name in networks:
        logging.info("Evaluating network {}".format(network_name))
        causalbionet, core_nodes, core_edge_count, graph, \
            base_graph, adj_b, adj_c = networks[network_name]

        for perm in permutations:
            for perm_rate in permutation_rates:
                logging.info("Evaluating permutation {}/{}".format(perm, perm_rate))
                adj_c_perms = (
                    permute_network.permute_adjacency(
                        adj_c, permutations=perm,
                        iterations=repetitions,
                        permutation_rate=perm_rate
                    ))[perm]

                for adj_c_perm in adj_c_perms:
                    perm_graph = create_networkx_from_adjacency(adj_c_perm, core_nodes)
                    perm_ged = graph_edit_distance(base_graph, perm_graph)
                    partial_perm_results = dict()
                    full_perm_results = dict()

                    for data_id in datasets[network_name]:
                        dataset, dataset_lap_b = datasets[network_name][data_id]
                        lap_c, lap_q, _ = network_matrices.generate_core_laplacians(
                            dataset_lap_b, adj_c, {},
                            boundary_outdegree_type="continuous"
                        )

                        lap_c_perm, lap_q_perm, _ = network_matrices.generate_core_laplacians(
                            dataset_lap_b, adj_c_perm, {},
                            boundary_outdegree_type="continuous"
                        )
                        core_coefficients = core.value_inference(dataset_lap_b, lap_c_perm, dataset["logFC"].to_numpy())
                        npa_partial = core.perturbation_amplitude(lap_q, core_coefficients, core_edge_count)
                        npa_full = core.perturbation_amplitude(lap_q_perm, core_coefficients, core_edge_count)

                        partial_perm_results[data_id] = npa_partial
                        full_perm_results[data_id] = npa_full

                    results.append({
                        "network": network_name,
                        "permutation": perm + "_partial",
                        "permutation_rate": perm_rate,
                        "ged": perm_ged,
                        "npa": partial_perm_results
                    })
                    results.append({
                        "network": network_name,
                        "permutation": perm + "_full",
                        "permutation_rate": perm_rate,
                        "ged": perm_ged,
                        "npa": full_perm_results
                    })

                adj_c_perms.clear()

        networks[network_name] = None
        datasets[network_name].clear()

    json.dump(results, open(out_file, "w"), indent=4)


if __name__ == "__main__":
    perms = ["k1", "k2"]
    perm_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    reps = 100

    # test_copd1(["k1", "k2"], perm_rates,
    #            reps, "k_perms_copd1.json")

    npa_networks = os.listdir("../../data/NPANetworks/")
    npa_networks = sorted([x[:-len("_backbone.tsv")] for x in npa_networks if x.endswith("_backbone.tsv")
                           and not x.startswith("Hs_CST_Xenobiotic")])
    ba_networks = os.listdir("../../data/BAGen03/")
    ba_networks = sorted([x[:-len("_core.tsv")] for x in ba_networks if x.endswith("_core.tsv")])

    """
    test_generated_data(
        "../../data/NPANetworks/",
        "_backbone.tsv",
        "_downstream.tsv",
        perms, perm_rates,
        reps, "k_perms_npa_1.json",
        network_selection=npa_networks[:8]
    )
    test_generated_data(
        "../../data/NPANetworks/",
        "_backbone.tsv",
        "_downstream.tsv",
        perms, perm_rates,
        reps, "k_perms_npa_2.json",
        network_selection=npa_networks[8:]
    )
    with open("k_perms_npa_1.json") as f1:
        results1 = json.load(f1)
    with open("k_perms_npa_2.json") as f2:
        results2 = json.load(f2)
    results = results1 + results2
    with open("k_perms_npa.json", "w") as f:
        json.dump(results, f, indent=4)
    """

    test_generated_data(
        "../../data/BAGen03/",
        "_core.tsv",
        "_boundary.tsv",
        perms, perm_rates,
        reps, "k_perms_ba_1.json",
        network_selection=ba_networks[:8]
    )
    """
    test_generated_data(
        "../../data/BAGen03/",
        "_core.tsv",
        "_boundary.tsv",
        perms, perm_rates,
        reps, "k_perms_ba_2.json",
        network_selection=ba_networks[8:]
    )
    """
    with open("k_perms_ba_1.json") as f1:
        results1 = json.load(f1)
    with open("k_perms_ba_2.json") as f2:
        results2 = json.load(f2)
    results = results1 + results2
    with open("k_perms_ba.json", "w") as f:
        json.dump(results, f, indent=4)

