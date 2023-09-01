import sys
import os
import json
import logging
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from perturbationx.CausalNetwork import CausalNetwork


def test_networks(network_folder, core_suffix, boundary_suffix, out_file):
    networks = {}
    for file_name in os.listdir(network_folder):
        if file_name.endswith(core_suffix) and not file_name.startswith("Hs_CST_Xenobiotic"):
            network_name = file_name[:-len(core_suffix)]
            core_file = network_folder + network_name + core_suffix
            boundary_file = network_folder + network_name + boundary_suffix

            causalbionet = CausalNetwork.from_tsv(core_file, edge_type="core")
            causalbionet.add_edges_from_tsv(boundary_file, edge_type="boundary")
            networks[network_name] = causalbionet

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
        datasets[dataset_network][dataset_name] = dataset
        dataset_types[dataset_name] = dataset_type


    opposing_pruning_modes = [None, "remove", "nullify"]
    boundary_edge_mins = [1, 6]
    boundary_outdegree_types = ["continuous", "binary"]

    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(levelname)s -- %(message)s")
    evaluations_per_network = len(opposing_pruning_modes) * len(boundary_edge_mins) * len(boundary_outdegree_types)
    results = []
    for network in networks:
        logging.info("Evaluating network %s" % network)
        for opposing_pruning_mode in opposing_pruning_modes:
            for boundary_edge_min in boundary_edge_mins:
                for boundary_outdegree_type in boundary_outdegree_types:
                    logging.info("Evaluation %d/%d" % (
                        len(results) % evaluations_per_network + 1, evaluations_per_network))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = networks[network].toponpa(
                            datasets[network],
                            opposing_value_pruning_mode=opposing_pruning_mode,
                            boundary_edge_minimum=boundary_edge_min,
                            exact_boundary_outdegree=boundary_outdegree_type,
                            permutations=None, verbose=False
                        )

                    matched_boundary_edges = {
                        dataset: result.metadata()["network_dataset_" + dataset]["matched_boundary_edges"]
                                 / result.metadata()["network_boundary_edges"]
                        for dataset in datasets[network]
                    }

                    results.append({
                        "network": network,
                        "opposing_pruning_mode": opposing_pruning_mode,
                        "boundary_edge_minimum": boundary_edge_min,
                        "boundary_outdegree_type": boundary_outdegree_type,
                        "npa": result.global_info()["NPA"].to_dict(),
                        "matched_boundary_edges": matched_boundary_edges
                    })

    json.dump(results, open(out_file, "w"), indent=4)


def plot_results(result_file):
    with open(result_file) as f:
        results = json.load(f)

    results_by_network = dict()
    for result in results:
        network = result["network"]
        if network not in results_by_network:
            results_by_network[network] = dict()

        opposing_pruning_mode = result["opposing_pruning_mode"]
        boundary_edge_minimum = result["boundary_edge_minimum"]
        boundary_outdegree_type = result["boundary_outdegree_type"]

        key = "n" if opposing_pruning_mode == "nullify" \
            else "r" if opposing_pruning_mode == "remove" \
            else "k"
        if boundary_edge_minimum > 1:
            key += "p"
        if boundary_outdegree_type == "binary":
            key += "l"

        for dataset in result["npa"]:
            npa = result["npa"][dataset]
            matched_boundary_edges = result["matched_boundary_edges"][dataset]

            if dataset not in results_by_network[network]:
                results_by_network[network][dataset] = dict()
            results_by_network[network][dataset][key] = (npa, matched_boundary_edges)

    color_legend_handles = [
        plt.scatter([], [], color="green", marker="o", s=20),
        plt.scatter([], [], color="red", marker="o", s=20),
        plt.scatter([], [], color="blue", marker="o", s=20)
    ]

    size_legend_handles = [
        plt.scatter([], [], color="black", marker="o", s=30),
        plt.scatter([], [], color="black", marker="o", s=5)
    ]

    x_labels = {
        "k": "No pruning",
        "r": "Remove",
        "rp": "Remove + Clip",
        "n": "Nullify",
        "nl": "Nullify + Legacy"
    }

    for network in results_by_network:
        fig, ax = plt.figure(figsize=(8, 4)), plt.gca()
        ax.set_title(network)

        for dataset in results_by_network[network]:
            x = [k for k in results_by_network[network][dataset]
                 if k not in ["kl", "kp", "kpl", "rl", "rpl", "np", "npl"]]
            y = [results_by_network[network][dataset][key][0] for key in x]
            size = [results_by_network[network][dataset][key][1] * 25 + 5 for key in x]

            x = [x_labels[key] for key in x]
            color = "green" if dataset.startswith("[1]") \
                else "red" if dataset.startswith("[-1]") \
                else "blue"
            ax.scatter(x, y, s=size, c=color)
            ax.plot(x, y, '-', color=color)

        color_legend = plt.legend(handles=color_legend_handles, labels=[
            "Maximum consistency", "Minimum consistency", "Random consistency"
        ], loc=(1.03, 0.65), title="Dataset type")
        ax.add_artist(color_legend)

        size_legend = plt.legend(handles=size_legend_handles, labels=["All", "None"],
                                 loc=(1.03, 0.05), title="Boundary edges kept")
        ax.add_artist(size_legend)

        plt.tight_layout(rect=[0, 0, 0.72, 1])
        fig.savefig("evaluation_modifications/%s.png" % network)
        plt.close(fig)


if __name__ == "__main__":
    test_networks(
        "../../data/NPANetworks/",
        "_backbone.tsv",
        "_downstream.tsv",
        "evaluation_results.json"
    )

    plot_results("evaluation_results.json")




