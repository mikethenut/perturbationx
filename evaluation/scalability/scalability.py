import os
import time
import json
from memory_profiler import memory_usage

import pandas as pd

from perturbationx import CausalNetwork


class FancyVariable:
    def __init__(self, value):
        self.value = value

    def set_value(self, value):
        self.value = value


def get_network_dataset_pairs():
    network_folder = "data/BAGen05/"
    core_suffix = "_core.tsv"
    dataset_folder = "data/ExpressionExamplesGen05/"
    network_data_pairs = dict()

    for file_name in os.listdir(network_folder):
        if file_name.endswith(core_suffix):
            network_file = file_name[:-len(core_suffix)]
            dataset_file = None

            for file_name_2 in os.listdir(dataset_folder):
                if file_name_2.startswith(network_file):
                    dataset_file = file_name_2
                    break

            if dataset_file is not None:
                network_data_pairs[network_file] = dataset_file

    return network_data_pairs


def npa_r_pipeline(sparse=False):
    print("Sparse: %r" % sparse)

    start = time.time_ns()
    dataset_files = ["CS (2m) + Sham (3m)", "CS (2m) + Sham (5m)", "CS (4m) + Sham (1m)",
                     "CS (4m) + Sham (3m)", "CS (5m)", "CS (7m)"]
    copd1_data = dict()
    for file in dataset_files:
        copd1_data[file] = pd.read_table("./data/COPD1/" + file + ".tsv")
        copd1_data[file] = copd1_data[file].rename(columns={"nodeLabel": "nodeID", "foldChange": "logFC"})

    mm_apoptosis = CausalNetwork.from_tsv("../data/NPANetworks/Mm_CFA_Apoptosis_backbone.tsv", edge_type="core")
    mm_apoptosis.add_edges_from_tsv("data/NPANetworks/Mm_CFA_Apoptosis_downstream.tsv", edge_type="boundary")

    mm_apoptosis.toponpa(copd1_data, boundary_edge_minimum=0, exact_boundary_outdegree=False,
                         permutations=["o", "k1"], full_core_permutation=False, sparse=sparse)
    end = time.time_ns()
    print("Time: ", (end - start) / 1e9)


def standard_pipeline(network_file, dataset_file, sparse=True, time_storage=None):
    start = time.time_ns()

    my_cbn = CausalNetwork.from_tsv(network_file + "_core.tsv", edge_type="core")
    my_cbn.add_edges_from_tsv(network_file + "_boundary.tsv", edge_type="boundary")

    dataset = {dataset_file.split("_dataset_")[-1][:8]: pd.read_table(dataset_file, sep=",")}
    my_cbn.toponpa(dataset, sparse=sparse)

    end = time.time_ns()
    print("Time: ", (end - start) / 1e9)

    if time_storage is not None:
        time_storage.set_value((end - start) / 1e9)


def permutation_pipeline(network_file, dataset_file):
    my_cbn = CausalNetwork.from_tsv(network_file + "_core.tsv", edge_type="core")
    my_cbn.add_edges_from_tsv(network_file + "_boundary.tsv", edge_type="boundary")

    dataset = {dataset_file.split("_dataset_")[-1][:8]: pd.read_table(dataset_file, sep=",")}
    my_cbn.toponpa(dataset, permutations=['o', 'k1', 'k2'], p_iters=5)


def scalability_testing():
    network_folder = "data/BAGen05/"
    dataset_folder = "data/ExpressionExamplesGen05/"
    network_data_pairs = get_network_dataset_pairs()

    scalability_results = "scalability_results.json"
    duration = FancyVariable(None)
    networks_all = sorted(list(network_data_pairs.keys()))
    network_core_sizes = sorted(list(set([int(x.split("_")[0]) for x in networks_all])), reverse=True)
    networks_sorted = []
    while len(networks_all) > 0:
        for core_size in network_core_sizes:
            for network_file in networks_all:
                if network_file.startswith(str(core_size) + "_"):
                    networks_sorted.append(network_file)
                    networks_all.remove(network_file)
                    break

    for network_file in networks_sorted:
        compute_sparse = True
        compute_dense = any([network_file.startswith(x) for x in ["500_", "1000_", "1500_"]])

        if os.path.exists(scalability_results):
            with open(scalability_results, "r") as f:
                results = json.load(f)
            if any([x["network"] == network_file and x["sparse"] == "True" for x in results]):
                compute_sparse = False
            if any([x["network"] == network_file and x["sparse"] == "False" for x in results]):
                compute_dense = False

        if not compute_sparse and not compute_dense:
            print("Skipping: ", network_file)
            continue

        dataset_file = network_data_pairs[network_file]
        results = []

        if compute_sparse:
            print("Processing: %s (sparse)" % network_file)
            mem = memory_usage((
                standard_pipeline,
                (network_folder + network_file,
                 dataset_folder + dataset_file,
                 True, duration)
            ))
            results.append({
                "network": network_file,
                "dataset": dataset_file,
                "sparse": "True",
                "max. memory": max(mem),
                "min. memory": min(mem),
                "avg. memory": sum(mem) / len(mem),
                "time": duration.value
            })

        if compute_dense:
            print("Processing: %s (dense)" % network_file)

            mem = memory_usage((
                standard_pipeline,
                (network_folder + network_file,
                 dataset_folder + dataset_file,
                 False, duration)
            ))
            results.append({
                "network": network_file,
                "dataset": dataset_file,
                "sparse": "False",
                "max. memory": max(mem),
                "min. memory": min(mem),
                "avg. memory": sum(mem) / len(mem),
                "time": duration.value
            })

        if os.path.exists(scalability_results):
            with open(scalability_results, "r") as f:
                results = json.load(f) + results

        with open(scalability_results, "w") as f:
            json.dump(results, f, indent=4)


def permutation_optimization():
    network_folder = "data/BAGen05/"
    dataset_folder = "data/ExpressionExamplesGen05/"
    network_data_pairs = get_network_dataset_pairs()

    network_selection = [n for n in network_data_pairs.keys() if n.startswith("3000_")]

    for network_file in network_selection:
        print("Processing: ", network_file)
        dataset_file = network_data_pairs[network_file]

        try:
            mem = memory_usage((
                permutation_pipeline,
                (network_folder + network_file,
                 dataset_folder + dataset_file)
            ))
            print({
                "network": network_file,
                "dataset": dataset_file,
                "sparse": "True",
                "max. memory": max(mem),
                "min. memory": min(mem),
                "avg. memory": sum(mem) / len(mem)
            })

        except RecursionError as e:
            print("Recursion error")
            print(e)
            continue


if __name__ == "__main__":
    mem = memory_usage((npa_r_pipeline, [True]))
    print(max(mem), min(mem), max(mem) - min(mem), sum(mem) / len(mem))

    mem = memory_usage((npa_r_pipeline, [False]))
    print(max(mem), min(mem), max(mem) - min(mem), sum(mem) / len(mem))

    # scalability_testing()
