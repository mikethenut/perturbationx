import os
import sys
import logging
import itertools
import json
import warnings
import uuid

import numpy as np
import pandas as pd

from perturbationx.CausalNetwork import CausalNetwork


def test_copd1(argument_product, signal_to_noise_ratio, missing_value_ratio, shuffled_value_ratio,
               repetitions, out_file):
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
    boundary_node_count = causalbionet.number_of_nodes(typ="boundary")

    datasets_folder = "../../data/COPD1/"
    datasets = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".tsv"):
            dataset = pd.read_table(datasets_folder + file_name, sep="\t")
            dataset_name = file_name.split(".")[0]
            dataset.rename(columns={"nodeLabel": "nodeID", "foldChange": "logFC"}, inplace=True)
            datasets[dataset_name] = dataset

    # Generate datasets
    dataset_source = dict()
    dataset_snr = dict()
    dataset_mvr = dict()
    dataset_svr = dict()
    perturbed_datasets = dict()
    for data_name in datasets:

        data_id = str(uuid.uuid4())[:8]
        base_dataset = datasets[data_name].copy()
        base_dataset = base_dataset[~base_dataset["logFC"].isna()]
        base_dataset = base_dataset[base_dataset["nodeID"].isin(causalbionet.nodes(data=False))]
        dataset_base_mvr = 1 - len(base_dataset) / boundary_node_count
        base_dataset.reset_index(inplace=True, drop=True)

        dataset_source[data_id] = data_name
        dataset_snr[data_id] = None
        dataset_mvr[data_id] = dataset_base_mvr
        dataset_svr[data_id] = 0.
        perturbed_datasets[data_id] = base_dataset.copy()
        dataset_avg_db = 10 * np.log10(np.mean(np.square(base_dataset["logFC"])))

        for snr in signal_to_noise_ratio:
            noise_avg_db = dataset_avg_db - snr
            noise_avg_power = 10 ** (noise_avg_db / 10)

            for _ in range(repetitions):
                data_id = str(uuid.uuid4())[:8]
                dataset = base_dataset.copy()
                noise = np.random.normal(0, np.sqrt(noise_avg_power), len(dataset))
                dataset["logFC"] = dataset["logFC"] + noise

                dataset_source[data_id] = data_name
                dataset_snr[data_id] = snr
                dataset_mvr[data_id] = dataset_base_mvr
                dataset_svr[data_id] = 0.
                perturbed_datasets[data_id] = dataset

        for mvr in missing_value_ratio:
            adjusted_mvr = mvr - dataset_base_mvr
            if adjusted_mvr <= 0:
                print("Cannot remove %.2f%% of values from dataset with %.2f%% missing values." %
                      (mvr * 100, dataset_base_mvr * 100))
                continue
            missing_value_count = int(len(base_dataset) * adjusted_mvr)

            for _ in range(repetitions):
                data_id = str(uuid.uuid4())[:8]
                dataset = base_dataset.copy()
                missing_idx = np.random.choice(len(dataset), missing_value_count, replace=False)
                dataset.loc[missing_idx, "logFC"] = np.nan

                dataset_source[data_id] = data_name
                dataset_snr[data_id] = None
                dataset_mvr[data_id] = mvr
                dataset_svr[data_id] = 0.
                perturbed_datasets[data_id] = dataset

        for svr in shuffled_value_ratio:
            shuffled_value_count = int(len(base_dataset) * svr)

            for _ in range(repetitions):
                data_id = str(uuid.uuid4())[:8]
                dataset = base_dataset.copy()
                shuffled_idx = np.random.choice(len(dataset), shuffled_value_count, replace=False)
                dataset.loc[shuffled_idx, "logFC"] = np.random.choice(
                    dataset["logFC"], shuffled_value_count, replace=False
                )

                dataset_source[data_id] = data_name
                dataset_snr[data_id] = None
                dataset_mvr[data_id] = dataset_base_mvr
                dataset_svr[data_id] = svr
                perturbed_datasets[data_id] = dataset

    results = []
    for mvm, opm, bem, bot in argument_product:
        logging.info("Evaluating parameter set {}/{}".format(len(results) + 1, len(argument_product)))

        with (warnings.catch_warnings()):
            warnings.simplefilter("ignore")
            result = causalbionet.toponpa(
                perturbed_datasets,
                missing_value_pruning_mode=mvm,
                opposing_value_pruning_mode=opm,
                boundary_edge_minimum=bem,
                exact_boundary_outdegree=bot,
                permutations=None, verbose=False
            )

            matched_boundary_edges = {
                dataset: result.metadata()["network_dataset_" + str(dataset)]["matched_boundary_edges"]
                         / result.metadata()["network_boundary_edges"]
                for dataset in perturbed_datasets
            }

            leading_nodes = {
                dataset: list(result.get_leading_nodes(dataset))
                for dataset in perturbed_datasets
            }

            # Add gene orderings to results
            results.append({
                "missing_value_mode": mvm,
                "opposing_pruning_mode": opm,
                "boundary_edge_minimum": bem,
                "boundary_outdegree_type": bot,
                "dataset_source": dataset_source,
                "dataset_snr": dataset_snr,
                "dataset_mvr": dataset_mvr,
                "dataset_svr": dataset_svr,
                "npa": result.global_info()["NPA"].to_dict(),
                "leading_nodes": leading_nodes,
                "matched_boundary_edges": matched_boundary_edges
            })

    json.dump(results, open(out_file, "w"), indent=4)


def test_generated_data(network_folder, core_suffix, boundary_suffix, argument_product,
                        signal_to_noise_ratio, missing_value_ratio, shuffled_value_ratio,
                        repetitions, out_file):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(levelname)s -- %(message)s")
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

    results = []
    for network in networks:
        logging.info("Evaluating network %s" % network)

        # Generate datasets
        dataset_source = dict()
        dataset_snr = dict()
        dataset_mvr = dict()
        dataset_svr = dict()
        perturbed_datasets = dict()
        for data_name in datasets[network]:
            data_id = str(uuid.uuid4())[:8]
            original_dataset = datasets[network][data_name]
            dataset_source[data_id] = data_name
            dataset_snr[data_id] = None
            dataset_mvr[data_id] = 0.
            dataset_svr[data_id] = 0.
            perturbed_datasets[data_id] = original_dataset
            dataset_avg_db = 10 * np.log10(np.mean(np.square(original_dataset["logFC"])))

            for snr in signal_to_noise_ratio:
                noise_avg_db = dataset_avg_db - snr
                noise_avg_power = 10 ** (noise_avg_db / 10)
                for _ in range(repetitions):
                    data_id = str(uuid.uuid4())[:8]
                    dataset = original_dataset.copy()
                    noise = np.random.normal(0, np.sqrt(noise_avg_power), len(dataset))
                    dataset["logFC"] = dataset["logFC"] + noise
                    dataset_source[data_id] = data_name
                    dataset_snr[data_id] = snr
                    dataset_mvr[data_id] = 0.
                    dataset_svr[data_id] = 0.
                    perturbed_datasets[data_id] = dataset

            for mvr in missing_value_ratio:
                missing_value_count = int(len(original_dataset) * mvr)
                for _ in range(repetitions):
                    data_id = str(uuid.uuid4())[:8]
                    dataset = original_dataset.copy()
                    missing_idx = np.random.choice(len(dataset), missing_value_count, replace=False)
                    dataset.loc[missing_idx, "logFC"] = np.nan
                    dataset_source[data_id] = data_name
                    dataset_snr[data_id] = None
                    dataset_mvr[data_id] = mvr
                    dataset_svr[data_id] = 0.
                    perturbed_datasets[data_id] = dataset

            for svr in shuffled_value_ratio:
                shuffled_value_count = int(len(original_dataset) * svr)
                for _ in range(repetitions):
                    data_id = str(uuid.uuid4())[:8]
                    dataset = original_dataset.copy()
                    shuffled_idx = np.random.choice(len(dataset), shuffled_value_count, replace=False)
                    dataset.loc[shuffled_idx, "logFC"] = np.random.choice(
                        dataset["logFC"], shuffled_value_count, replace=False
                    )

                    dataset_source[data_id] = data_name
                    dataset_snr[data_id] = None
                    dataset_mvr[data_id] = 0.
                    dataset_svr[data_id] = svr
                    perturbed_datasets[data_id] = dataset

        for mvm, opm, bem, bot in argument_product:
            logging.info("Evaluating parameter set {}/{}".format(
                len(results) % len(argument_product) + 1, len(argument_product))
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = networks[network].toponpa(
                    perturbed_datasets,
                    missing_value_pruning_mode=mvm,
                    opposing_value_pruning_mode=opm,
                    boundary_edge_minimum=bem,
                    exact_boundary_outdegree=bot,
                    permutations=None, verbose=False
                )

                matched_boundary_edges = {
                    dataset: result.metadata()["network_dataset_" + str(dataset)]["matched_boundary_edges"]
                             / result.metadata()["network_boundary_edges"]
                    for dataset in perturbed_datasets
                }

                leading_nodes = {
                    dataset: list(result.get_leading_nodes(dataset))
                    for dataset in perturbed_datasets
                }

                # Add gene orderings to results
                results.append({
                    "network": network,
                    "missing_value_mode": mvm,
                    "opposing_pruning_mode": opm,
                    "boundary_edge_minimum": bem,
                    "boundary_outdegree_type": bot,
                    "dataset_source": dataset_source,
                    "dataset_type": dataset_types,
                    "dataset_snr": dataset_snr,
                    "dataset_mvr": dataset_mvr,
                    "dataset_svr": dataset_svr,
                    "npa": result.global_info()["NPA"].to_dict(),
                    "leading_nodes": leading_nodes,
                    "matched_boundary_edges": matched_boundary_edges
                })

    json.dump(results, open(out_file, "w"), indent=4)


def evaluate_all():
    mvpms = ["remove", "nullify"]
    ovpms = [None, "remove", "nullify"]
    bems = [1, 6]
    bots = ["continuous", "binary"]
    argument_product = itertools.product(mvpms, ovpms, bems, bots)

    snrs = [25, 22.5, 20, 17.5, 15, 12.5, 10, 7.5, 5, 2.5, 0]
    mvrs = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    svrs = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    reps = 5
    test_copd1(
        argument_product,
        snrs, mvrs, svrs, reps,
        "copd1_evaluation.json"
    )

    snrs = [25, 20, 15, 10, 5, 0]
    mvrs = [0.05, 0.1, 0.15, 0.2, 0.25]
    svrs = [0.05, 0.1, 0.15, 0.2, 0.25]
    reps = 3

    test_generated_data(
        "../../data/NPANetworks/",
        "_backbone.tsv",
        "_downstream.tsv",
        argument_product,
        snrs, mvrs, svrs, reps,
        "npa_evaluation.json"
    )
    test_generated_data(
        "../../data/BAGen03/",
        "_core.tsv",
        "_boundary.tsv",
        argument_product,
        snrs, mvrs, svrs, reps,
        "ba_evaluation.json"
    )


def evaluate_missing_value_pruning():
    argument_product = [
        ("remove", None, 6, "binary"),
        ("nullify", None, 6, "binary"),
        ("nullify", None, 6, "continuous")
    ]
    snrs = []
    mvrs = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.30]
    svrs = []
    reps = 10

    test_copd1(
        argument_product,
        snrs, mvrs, svrs, reps,
        "mvr_evaluation_copd1.json"
    )

    mvrs = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    test_generated_data(
        "../../data/NPANetworks/",
        "_backbone.tsv",
        "_downstream.tsv",
        argument_product,
        snrs, mvrs, svrs, reps,
        "mvr_evaluation_npa.json"
    )
    test_generated_data(
        "../../data/BAGen03/",
        "_core.tsv",
        "_boundary.tsv",
        argument_product,
        snrs, mvrs, svrs, reps,
        "mvr_evaluation_ba.json"
    )


def evaluate_opposing_value_pruning():
    argument_product = [
        ("nullify", None, 6, "binary"),
        ("remove", "remove", 6, "binary"),
        ("nullify", "nullify", 6, "binary"),
        ("nullify", "nullify", 6, "continuous")
    ]
    snrs = []
    mvrs = []
    svrs = []
    reps = 1

    test_generated_data(
        "../../data/NPANetworks/",
        "_backbone.tsv",
        "_downstream.tsv",
        argument_product,
        snrs, mvrs, svrs, reps,
        "ovr_evaluation_npa.json"
    )
    test_generated_data(
        "../../data/BAGen03/",
        "_core.tsv",
        "_boundary.tsv",
        argument_product,
        snrs, mvrs, svrs, reps,
        "ovr_evaluation_ba.json"
    )


def evaluate_signal_to_noise_robustness():
    argument_product = [
        ("nullify", None, 6, "continuous")
    ]
    snrs = [25, 22.5, 20, 17.5, 15, 12.5, 10, 7.5, 5, 2.5, 0]
    mvrs = []
    svrs = []
    reps = 10

    test_copd1(
        argument_product,
        snrs, mvrs, svrs, reps,
        "snr_evaluation/snr_evaluation_copd1.json"
    )
    test_generated_data(
        "../../data/NPANetworks/",
        "_backbone.tsv",
        "_downstream.tsv",
        argument_product,
        snrs, mvrs, svrs, reps,
        "snr_evaluation/snr_evaluation_npa.json"
    )
    test_generated_data(
        "../../data/BAGen03/",
        "_core.tsv",
        "_boundary.tsv",
        argument_product,
        snrs, mvrs, svrs, reps,
        "snr_evaluation/snr_evaluation_ba.json"
    )


def evaluate_shuffled_value_robustness():
    argument_product = [
        ("nullify", None, 6, "continuous")
    ]
    snrs = []
    mvrs = []
    svrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    reps = 10

    test_copd1(
        argument_product,
        snrs, mvrs, svrs, reps,
        "svr_evaluation/svr_evaluation_copd1.json"
    )
    test_generated_data(
        "../../data/NPANetworks/",
        "_backbone.tsv",
        "_downstream.tsv",
        argument_product,
        snrs, mvrs, svrs, reps,
        "svr_evaluation/svr_evaluation_npa.json"
    )
    test_generated_data(
        "../../data/BAGen03/",
        "_core.tsv",
        "_boundary.tsv",
        argument_product,
        snrs, mvrs, svrs, reps,
        "svr_evaluation/svr_evaluation_ba.json"
    )


if __name__ == "__main__":
    evaluate_signal_to_noise_robustness()
