import json

import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_key(result):
    key = ""
    if result["missing_value_mode"] == "remove":
        key += "r"
    elif result["missing_value_mode"] == "nullify":
        key += "n"
    if result["opposing_pruning_mode"] == "remove":
        key += "r"
    elif result["opposing_pruning_mode"] == "nullify":
        key += "n"
    if result["boundary_edge_minimum"] > 1:
        key += "c"
    if result["boundary_outdegree_type"] == "binary":
        key += "l"

    return key


def key_to_label(key):
    label = ""
    if key.startswith("r"):
        label += "Remove missing"
    elif key.startswith("n"):
        label += "Nullify missing"
    else:
        return "Invalid label"
    key = key[1:]

    if key.startswith("r"):
        label += "\nRemove opposing"
        key = key[1:]
    elif key.startswith("n"):
        label += "\nNullify opposing"
        key = key[1:]
    else:
        label += "\nKeep opposing"

    if key.startswith("c"):
        label += "\nClip"

    if not key.endswith("l"):
        label += "\nExact outdegree"

    return label


def plot_copd1_robustness(filename):
    with open(filename) as f:
        results = json.load(f)

    snr_results_by_params = dict()
    mvr_results_by_params = dict()
    for idx, result in enumerate(results):
        key = get_key(result)

        if key.startswith("nr"):
            continue

        if 'p' in key and 'r' not in key:
            continue

        if 'l' in key and 'n' not in key:
            continue

        snr_results_by_params[key] = {}
        mvr_results_by_params[key] = {}
        for dataset in result["dataset_source"]:
            dataset_source = result["dataset_source"][dataset]
            dataset_snr = result["dataset_csr"][dataset]
            dataset_mvr = result["dataset_mvr"][dataset]

            if dataset_snr is None:
                if dataset_source not in mvr_results_by_params[key]:
                    mvr_results_by_params[key][dataset_source] = dict()
                if dataset_mvr not in mvr_results_by_params[key][dataset_source]:
                    mvr_results_by_params[key][dataset_source][dataset_mvr] = {
                        "NPA": [], "Node ranking": [], "Matched boundary edges": []
                    }

                mvr_results_by_params[key][dataset_source][dataset_mvr]["NPA"].append(
                    result["npa"][dataset]
                )
                mvr_results_by_params[key][dataset_source][dataset_mvr]["Node ranking"].append(
                    result["node_rankings"][dataset]
                )
                mvr_results_by_params[key][dataset_source][dataset_mvr]["Matched boundary edges"].append(
                    result["matched_boundary_edges"][dataset]
                )

            if dataset_mvr == 0.:
                if dataset_source not in snr_results_by_params[key]:
                    snr_results_by_params[key][dataset_source] = dict()
                if dataset_snr not in snr_results_by_params[key][dataset_source]:
                    snr_results_by_params[key][dataset_source][dataset_snr] = {
                        "NPA": [], "Node ranking": [], "Matched boundary edges": []
                    }

                snr_results_by_params[key][dataset_source][dataset_snr]["NPA"].append(
                    result["npa"][dataset]
                )
                snr_results_by_params[key][dataset_source][dataset_snr]["Node ranking"].append(
                    result["node_rankings"][dataset]
                )

                snr_results_by_params[key][dataset_source][dataset_snr]["Matched boundary edges"].append(
                    result["matched_boundary_edges"][dataset]
                )

    columns = 3
    rows = len(snr_results_by_params)

    # Plot SNR results
    fig, ax = plt.subplots(rows, columns, figsize=(8*columns, 4*rows))
    for idx, key in enumerate(snr_results_by_params):
        row = idx

        x = []
        npa_mrae = []
        top_10_recall = []
        spearman_rho = []

        for dataset_source in snr_results_by_params[key]:
            ref_npa = snr_results_by_params[key][dataset_source][None]["NPA"][0]
            if ref_npa == 0.:
                print("Skipping", key, dataset_source)
                continue

            ref_node_ranking = snr_results_by_params[key][dataset_source][None]["Node ranking"][0]
            top_10_size = np.ceil(len(ref_node_ranking) / 10.).astype(int)

            for dataset_snr in snr_results_by_params[key][dataset_source]:
                if dataset_snr is None:
                    continue

                x.append(dataset_snr)

                count = 0
                npa_rae = 0
                for npa in snr_results_by_params[key][dataset_source][dataset_snr]["NPA"]:
                    count += 1
                    npa_rae += np.abs(npa - ref_npa) / ref_npa
                npa_mrae.append(npa_rae / count)

                count = 0
                recall_10 = 0
                spearman = 0
                for node_ranking in snr_results_by_params[key][dataset_source][dataset_snr]["Node ranking"]:
                    count += 1
                    recall_10 += (len(set(node_ranking[:top_10_size])
                                   & set(ref_node_ranking[:top_10_size]))
                               / top_10_size)
                    spearman += spearmanr(node_ranking, ref_node_ranking)[0]
                top_10_recall.append(recall_10 / count)
                spearman_rho.append(spearman / count)

        sns.boxplot(x=x, y=npa_mrae, ax=ax[row, 0],  width=0.2)
        sns.boxplot(x=x, y=spearman_rho, ax=ax[row, 1],  width=0.2)
        sns.boxplot(x=x, y=top_10_recall, ax=ax[row, 2],  width=0.2)

        for col in range(columns):
            ax[row, col].set_xlabel("Signal-to-Noise Ratio")
            ax[row, col].invert_xaxis()

        ax[row, 0].set_ylim([0, 1])
        ax[row, 1].set_ylim([-0.2, 1])
        ax[row, 2].set_ylim([0.5, 1])

        ax[row, 0].annotate(key_to_label(key), xy=(0, 0.5), xytext=(-ax[row, 0].yaxis.labelpad, 0),
                            xycoords=ax[row, 0].yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center')
        ax[row, 0].set_ylabel("NPA Relative Absolute Error")
        ax[row, 1].set_ylabel("Spearman Correlation Coefficient")
        ax[row, 2].set_ylabel("Top 10% Recall")

    plt.tight_layout()
    plt.savefig("copd1_snr_results.png")
    plt.clf()

    # Plot MVR results
    fig, ax = plt.subplots(rows, columns, figsize=(8 * columns, 4 * rows))
    for idx, key in enumerate(mvr_results_by_params):
        row = idx

        x = []
        npa_mrae = []
        spearman_rho = []
        top_10_recall = []

        for dataset_source in mvr_results_by_params[key]:
            ref_npa = mvr_results_by_params[key][dataset_source][0]["NPA"][0]
            if ref_npa == 0.:
                print("Skipping", key, dataset_source)
                continue

            ref_node_ranking = mvr_results_by_params[key][dataset_source][0]["Node ranking"][0]
            top_10_size = np.ceil(len(ref_node_ranking) / 10.).astype(int)

            for dataset_mvr in mvr_results_by_params[key][dataset_source]:
                if dataset_mvr == 0:
                    continue

                x.append(dataset_mvr)

                count = 0
                npa_rae = 0
                for npa in mvr_results_by_params[key][dataset_source][dataset_mvr]["NPA"]:
                    count += 1
                    npa_rae += np.abs(npa - ref_npa) / ref_npa
                npa_mrae.append(npa_rae / count)

                count = 0
                recall_10 = 0
                spearman = 0
                for node_ranking in mvr_results_by_params[key][dataset_source][dataset_mvr]["Node ranking"]:
                    count += 1
                    recall_10 += (len(set(node_ranking[:top_10_size])
                                   & set(ref_node_ranking[:top_10_size]))
                               / top_10_size)
                    spearman += spearmanr(node_ranking, ref_node_ranking)[0]
                top_10_recall.append(recall_10 / count)
                spearman_rho.append(spearman / count)

        sns.boxplot(x=x, y=npa_mrae, ax=ax[row, 0],  width=0.2)
        sns.boxplot(x=x, y=spearman_rho, ax=ax[row, 1],  width=0.2)
        sns.boxplot(x=x, y=top_10_recall, ax=ax[row, 2],  width=0.2)

        for col in range(columns):
            ax[row, col].set_xlabel("Missing Value Ratio")

        ax[row, 0].set_ylim([0, 1.2])
        ax[row, 1].set_ylim([-0.2, 0.5])
        ax[row, 2].set_ylim([0.2, 1])

        ax[row, 0].annotate(key_to_label(key), xy=(0, 0.5), xytext=(-ax[row, 0].yaxis.labelpad, 0),
                            xycoords=ax[row, 0].yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center')
        ax[row, 0].set_ylabel("NPA Relative Absolute Error")
        ax[row, 1].set_ylabel("Spearman Correlation Coefficient")
        ax[row, 2].set_ylabel("Top 10% Recall")

    plt.tight_layout()
    plt.savefig("copd1_mvr_results.png")
    plt.clf()


def plot_copd1_mvr_robustness(in_file):
    with open(in_file) as f:
        results = json.load(f)

    results_by_params = dict()
    for idx, result in enumerate(results):
        key = get_key(result)
        results_by_params[key] = {}

        for dataset in result["dataset_source"]:
            dataset_source = result["dataset_source"][dataset]
            dataset_mvr = result["dataset_mvr"][dataset]

            if dataset_source not in results_by_params[key]:
                results_by_params[key][dataset_source] = dict()
            if dataset_mvr not in results_by_params[key][dataset_source]:
                results_by_params[key][dataset_source][dataset_mvr] = {
                    "NPA": [], "Leading nodes": [], "Matched boundary edges": []
                }

            results_by_params[key][dataset_source][dataset_mvr]["NPA"].append(
                result["npa"][dataset]
            )
            results_by_params[key][dataset_source][dataset_mvr]["Leading nodes"].append(
                result["leading_nodes"][dataset]
            )
            results_by_params[key][dataset_source][dataset_mvr]["Matched boundary edges"].append(
                result["matched_boundary_edges"][dataset]
            )

    # Plot MVR results
    keys = ["ncl", "rcl", "nc"]
    colors = {
        "ncl": "tab:green",
        "nc": "tab:blue",
        "rcl": "tab:orange"
    }
    columns = 3
    fig, ax = plt.subplots(1, columns, figsize=(4 * columns, 5))
    for idx, key in enumerate(keys):
        x = []
        npa_mrae = []
        ln_precision = []
        ln_recall = []

        for dataset_source in results_by_params[key]:
            smallest_mvr = min(results_by_params[key][dataset_source].keys())

            x.append(smallest_mvr)
            npa_mrae.append(0.)
            ln_precision.append(1.)
            ln_recall.append(1.)

            ref_npa = results_by_params[key][dataset_source][smallest_mvr]["NPA"][0]
            if ref_npa == 0.:
                print("Skipping", key, dataset_source)
                continue
            ref_leading_nodes = results_by_params[key][dataset_source][smallest_mvr]["Leading nodes"][0]

            for dataset_mvr in results_by_params[key][dataset_source]:
                if dataset_mvr == smallest_mvr:
                    continue

                x.append(dataset_mvr)

                count = 0
                npa_rae = 0
                for npa in results_by_params[key][dataset_source][dataset_mvr]["NPA"]:
                    count += 1
                    npa_rae += np.abs(npa - ref_npa) / ref_npa
                npa_mrae.append(npa_rae / count)

                count = 0
                recall = 0
                precision = 0
                for leading_nodes in results_by_params[key][dataset_source][dataset_mvr]["Leading nodes"]:
                    count += 1
                    recall += len(set(leading_nodes) & set(ref_leading_nodes)) / len(ref_leading_nodes)
                    precision += len(set(leading_nodes) & set(ref_leading_nodes)) / len(leading_nodes)
                ln_recall.append(recall / count)
                ln_precision.append(precision / count)

        label = key_to_label(key)
        label = label.replace("\nKeep opposing\nClip", "")
        color = colors[key]

        sns.lineplot(x=x, y=npa_mrae, ax=ax[0], color=color, label=label)
        sns.lineplot(x=x, y=ln_recall, ax=ax[1], color=color)
        sns.lineplot(x=x, y=ln_precision, ax=ax[2], color=color)

    for col in range(columns):
        ax[col].set_xlabel("Missing Value Ratio")

    handles, labels = ax[0].get_legend_handles_labels()
    order = [0, 2, 1]
    ax[0].legend([handles[idx] for idx in order],
                 [labels[idx] for idx in order],
                 loc="upper left")

    ax[0].set_ylabel("NPA Relative Absolute Error")
    ax[0].set_ylim([-0.05, 1.05])
    ax[1].set_ylabel("Leading Node Recall")
    ax[1].set_ylim([0.15, 1.05])
    ax[2].set_ylabel("Leading Node Precision")
    ax[2].set_ylim([0.15, 1.05])

    plt.suptitle("Mm Apoptosis")
    plt.tight_layout()
    out_file = in_file.replace(".json", ".png")
    plt.savefig(out_file, dpi=300)
    plt.clf()


def plot_gen_mvr_robustness(in_file, title):
    with open(in_file) as f:
        results = json.load(f)

    results_by_network = dict()
    for idx, result in enumerate(results):
        key = get_key(result)
        network = result["network"]

        if network not in results_by_network:
            results_by_network[network] = dict()
        results_by_network[network][key] = dict()

        for dataset in result["dataset_source"]:
            dataset_source = result["dataset_source"][dataset]
            dataset_mvr = result["dataset_mvr"][dataset]

            if dataset_source not in results_by_network[network][key]:
                results_by_network[network][key][dataset_source] = dict()
            if dataset_mvr not in results_by_network[network][key][dataset_source]:
                results_by_network[network][key][dataset_source][dataset_mvr] = {
                    "NPA": [], "Leading nodes": [], "Matched boundary edges": []
                }

            results_by_network[network][key][dataset_source][dataset_mvr]["NPA"].append(
                result["npa"][dataset]
            )
            results_by_network[network][key][dataset_source][dataset_mvr]["Leading nodes"].append(
                result["leading_nodes"][dataset]
            )
            results_by_network[network][key][dataset_source][dataset_mvr]["Matched boundary edges"].append(
                result["matched_boundary_edges"][dataset]
            )

    # Plot MVR results
    keys = ["ncl", "rcl", "nc"]
    columns = 3
    fig, ax = plt.subplots(1, columns, figsize=(4 * columns, 5))
    lineplot_data = {key: ([], [], [], []) for key in keys}

    for network in results_by_network:
        for idx, key in enumerate(keys):
            x, npa_mrae, ln_precision, ln_recall = lineplot_data[key]

            for dataset_source in results_by_network[network][key]:
                smallest_mvr = min(results_by_network[network][key][dataset_source].keys())

                x.append(smallest_mvr)
                npa_mrae.append(0.)
                ln_precision.append(1.)
                ln_recall.append(1.)

                ref_npa = results_by_network[network][key][dataset_source][smallest_mvr]["NPA"][0]
                if ref_npa == 0.:
                    print("Skipping", key, dataset_source)
                    continue
                ref_leading_nodes = results_by_network[network][key][dataset_source][smallest_mvr]["Leading nodes"][0]

                for dataset_mvr in results_by_network[network][key][dataset_source]:
                    if dataset_mvr == smallest_mvr:
                        continue

                    x.append(dataset_mvr)

                    count = 0
                    npa_rae = 0
                    for npa in results_by_network[network][key][dataset_source][dataset_mvr]["NPA"]:
                        count += 1
                        npa_rae += np.abs(npa - ref_npa) / ref_npa
                    npa_mrae.append(npa_rae / count)

                    count = 0
                    recall = 0
                    precision = 0
                    for leading_nodes in results_by_network[network][key][dataset_source][dataset_mvr]["Leading nodes"]:
                        count += 1
                        recall += len(set(leading_nodes) & set(ref_leading_nodes)) / len(ref_leading_nodes)
                        precision += len(set(leading_nodes) & set(ref_leading_nodes)) / len(leading_nodes)
                    ln_recall.append(recall / count)
                    ln_precision.append(precision / count)

    colors = {
        "ncl": "tab:green",
        "nc": "tab:blue",
        "rcl": "tab:orange"
    }
    for key in keys:
        x, npa_mrae, ln_precision, ln_recall = lineplot_data[key]

        label = key_to_label(key)
        label = label.replace("\nKeep opposing\nClip", "")
        color = colors[key]

        sns.lineplot(x=x, y=npa_mrae, ax=ax[0], color=color, label=label)
        sns.lineplot(x=x, y=ln_recall, ax=ax[1], color=color)
        sns.lineplot(x=x, y=ln_precision, ax=ax[2], color=color)

    handles, labels = ax[0].get_legend_handles_labels()
    order = [0, 2, 1]
    ax[0].legend([handles[idx] for idx in order],
                 [labels[idx] for idx in order],
                 loc="upper left")

    for col in range(columns):
        ax[col].set_xlabel("Missing Value Ratio")

    ax[0].set_ylabel("NPA Relative Absolute Error")
    ax[0].set_ylim([-0.05, 0.55])
    ax[1].set_ylabel("Leading Node Recall")
    ax[1].set_ylim([0.73, 1.02])
    ax[2].set_ylabel("Leading Node Precision")
    ax[2].set_ylim([0.73, 1.02])

    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    out_file = in_file.replace(".json", ".png")
    plt.savefig(out_file, dpi=300)
    plt.clf()


def plot_gen_opposite_pruning(in_file, title):
    with open(in_file) as f:
        results = json.load(f)

    results_by_network = dict()
    dataset_types = results[0]["dataset_type"]
    for idx, result in enumerate(results):
        key = get_key(result)
        network = result["network"]

        if network not in results_by_network:
            results_by_network[network] = dict()
        results_by_network[network][key] = dict()

        for dataset in result["dataset_source"]:
            dataset_source = result["dataset_source"][dataset]
            results_by_network[network][key][dataset_source] = dict()

            results_by_network[network][key][dataset_source]["NPA"] = \
                result["npa"][dataset]
            results_by_network[network][key][dataset_source]["Leading nodes"] = \
                result["leading_nodes"][dataset]
            results_by_network[network][key][dataset_source]["Matched boundary edges"] = \
                result["matched_boundary_edges"][dataset]

    # Plot OVR results
    rows = 3
    keys = ["ncl", "nncl", "nnc", "rrcl"]
    ref_key = "ncl"
    fig, ax = plt.subplots(rows, 1, figsize=(6, 5 * rows))
    lineplot_data = {
        -1: ([], [], [], []),
        0: ([], [], [], []),
        1: ([], [], [], [])
    }

    for network in results_by_network:
        ref_npa = np.mean([results_by_network[network][ref_key][dataset]["NPA"]
                           for dataset in results_by_network[network][ref_key]])
        ref_lns = {dataset: results_by_network[network][ref_key][dataset]["Leading nodes"]
                   for dataset in results_by_network[network][ref_key]}

        for idx, key in enumerate(keys):
            for dataset_source in results_by_network[network][key]:
                dataset_type = dataset_types[dataset_source]
                x, relative_npa, ln_precision, ln_recall = lineplot_data[dataset_type]

                label = key_to_label(key)
                label = (label.replace("\nClip", "")
                         .replace("Nullify missing\n", "")
                         .replace("Remove missing\n", "")
                         .replace(" opposing", ""))

                x.append(label)
                relative_npa.append(
                    results_by_network[network][key][dataset_source]["NPA"] / ref_npa
                )

                lns = results_by_network[network][key][dataset_source]["Leading nodes"]
                ln_precision.append(
                    len(set(lns) & set(ref_lns[dataset_source])) / len(lns)
                )
                ln_recall.append(
                    len(set(lns) & set(ref_lns[dataset_source])) / len(ref_lns[dataset_source])
                )

    dataset_labels = {
        -1: "Inconsistent datasets",
        0: "Random datasets",
        1: "Consistent datasets"
    }
    palette = sns.color_palette("Set2")
    palette = [palette[2], palette[1], palette[0]]
    plot_dataframe = pd.DataFrame()
    for dataset_type, color in zip([0, -1, 1], palette):
        x, relative_npa, ln_precision, ln_recall = lineplot_data[dataset_type]
        dataset_dataframe = pd.DataFrame({
            "Opposing edge pruning method": x,
            "Relative NPA": relative_npa,
            "Leading Node Recall": ln_recall,
            "Leading Node Precision": ln_precision
        })
        dataset_dataframe["Dataset type"] = dataset_labels[dataset_type]
        plot_dataframe = pd.concat([plot_dataframe, dataset_dataframe])

    sns.boxplot(x="Opposing edge pruning method",
                y="Relative NPA", hue="Dataset type",
                data=plot_dataframe, ax=ax[0], palette=palette)
    plot_dataframe = plot_dataframe[plot_dataframe["Opposing edge pruning method"] != "Keep"]
    sns.boxplot(x="Opposing edge pruning method",
                y="Leading Node Recall", hue="Dataset type",
                data=plot_dataframe, ax=ax[1], palette=palette)
    sns.boxplot(x="Opposing edge pruning method",
                y="Leading Node Precision", hue="Dataset type",
                data=plot_dataframe, ax=ax[2], palette=palette)

    ax[0].set_xlabel(None)
    ax[0].set_ylim(-0.15, 2.65)
    ax[1].set_xlabel(None)
    ax[1].get_legend().remove()
    ax[1].set_ylim(-0.05, 1.05)
    ax[2].get_legend().remove()
    ax[2].set_ylim(-0.05, 1.05)

    handles, labels = ax[0].get_legend_handles_labels()
    order = [2, 0, 1]
    ax[0].legend([handles[idx] for idx in order],
                 [labels[idx] for idx in order],
                 loc="upper left")

    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    out_file = in_file.replace(".json", ".png")
    plt.savefig(out_file, dpi=300)
    plt.clf()


def plot_copd1_snr_robustness(in_file):
    with open(in_file) as f:
        results = json.load(f)

    results_by_dataset = dict()
    for idx, result in enumerate(results):
        key = get_key(result)
        if key != "nc":
            print("Skipping %s" % key)
            continue

        for dataset in result["dataset_source"]:
            dataset_source = result["dataset_source"][dataset]
            dataset_snr = result["dataset_snr"][dataset]

            if dataset_source not in results_by_dataset:
                results_by_dataset[dataset_source] = dict()
            if dataset_snr not in results_by_dataset[dataset_source]:
                results_by_dataset[dataset_source][dataset_snr] = {
                    "NPA": [], "Leading nodes": [], "Matched boundary edges": []
                }

            results_by_dataset[dataset_source][dataset_snr]["NPA"].append(
                result["npa"][dataset]
            )
            results_by_dataset[dataset_source][dataset_snr]["Leading nodes"].append(
                result["leading_nodes"][dataset]
            )
            results_by_dataset[dataset_source][dataset_snr]["Matched boundary edges"].append(
                result["matched_boundary_edges"][dataset]
            )

    # Plot SNR results
    columns = 3
    fig, ax = plt.subplots(1, columns, figsize=(4 * columns, 6))
    x = []
    npa_mrae = []
    ln_precision = []
    ln_recall = []

    for dataset_source in results_by_dataset:
        base_results = results_by_dataset[dataset_source][None]

        ref_npa = base_results["NPA"][0]
        if ref_npa == 0.:
            print("Skipping", dataset_source)
            continue
        ref_leading_nodes = base_results["Leading nodes"][0]

        for dataset_snr in results_by_dataset[dataset_source]:
            if dataset_snr is None:
                continue

            x.append(dataset_snr)

            count = 0
            npa_rae = 0
            for npa in results_by_dataset[dataset_source][dataset_snr]["NPA"]:
                count += 1
                npa_rae += np.abs(npa - ref_npa) / ref_npa
            npa_mrae.append(npa_rae / count)

            count = 0
            recall = 0
            precision = 0
            for leading_nodes in results_by_dataset[dataset_source][dataset_snr]["Leading nodes"]:
                count += 1
                recall += len(set(leading_nodes) & set(ref_leading_nodes)) / len(ref_leading_nodes)
                precision += len(set(leading_nodes) & set(ref_leading_nodes)) / len(leading_nodes)
            ln_recall.append(recall / count)
            ln_precision.append(precision / count)

    sns.lineplot(x=x, y=npa_mrae, ax=ax[0], color="tab:green")
    sns.lineplot(x=x, y=ln_recall, ax=ax[1], color="tab:green")
    sns.lineplot(x=x, y=ln_precision, ax=ax[2], color="tab:green")

    for col in range(columns):
        ax[col].set_xlabel("Signal to Noise Ratio")
        ax[col].invert_xaxis()

    ax[0].set_ylabel("NPA Relative Absolute Error")
    ax[0].set_ylim([0, 0.3])
    ax[1].set_ylabel("Leading Node Recall")
    ax[1].set_ylim([0.75, 1])
    ax[2].set_ylabel("Leading Node Precision")
    ax[2].set_ylim([0.75, 1])

    plt.tight_layout()
    out_file = in_file.replace(".json", ".png")
    plt.savefig(out_file, dpi=300)
    plt.clf()


def plot_gen_snr_robustness(in_file):
    with open(in_file) as f:
        results = json.load(f)

    results_by_network = dict()
    for idx, result in enumerate(results):
        key = get_key(result)
        if key != "nc":
            print("Skipping %s" % key)
            continue

        network = result["network"]
        if network not in results_by_network:
            results_by_network[network] = dict()

        for dataset in result["dataset_source"]:
            dataset_source = result["dataset_source"][dataset]
            dataset_snr = result["dataset_snr"][dataset]

            if dataset_source not in results_by_network[network]:
                results_by_network[network][dataset_source] = dict()
            if dataset_snr not in results_by_network[network][dataset_source]:
                results_by_network[network][dataset_source][dataset_snr] = {
                    "NPA": [], "Leading nodes": [], "Matched boundary edges": []
                }

            results_by_network[network][dataset_source][dataset_snr]["NPA"].append(
                result["npa"][dataset]
            )
            results_by_network[network][dataset_source][dataset_snr]["Leading nodes"].append(
                result["leading_nodes"][dataset]
            )
            results_by_network[network][dataset_source][dataset_snr]["Matched boundary edges"].append(
                result["matched_boundary_edges"][dataset]
            )

    # Plot SNR results
    columns = 3
    fig, ax = plt.subplots(1, columns, figsize=(4 * columns, 6))
    x, npa_mrae, ln_precision, ln_recall = [], [], [], []

    for network in results_by_network:

        for dataset_source in results_by_network[network]:
            base_results = results_by_network[network][dataset_source][None]

            ref_npa = base_results["NPA"][0]
            if ref_npa == 0.:
                print("Skipping", dataset_source)
                continue
            ref_leading_nodes = base_results["Leading nodes"][0]

            for dataset_snr in results_by_network[network][dataset_source]:
                if dataset_snr is None:
                    continue

                x.append(dataset_snr)

                count = 0
                npa_rae = 0
                for npa in results_by_network[network][dataset_source][dataset_snr]["NPA"]:
                    count += 1
                    npa_rae += np.abs(npa - ref_npa) / ref_npa
                npa_mrae.append(npa_rae / count)

                count = 0
                recall = 0
                precision = 0
                for leading_nodes in results_by_network[network][dataset_source][dataset_snr]["Leading nodes"]:
                    count += 1
                    recall += len(set(leading_nodes) & set(ref_leading_nodes)) / len(ref_leading_nodes)
                    precision += len(set(leading_nodes) & set(ref_leading_nodes)) / len(leading_nodes)
                ln_recall.append(recall / count)
                ln_precision.append(precision / count)

    sns.lineplot(x=x, y=npa_mrae, ax=ax[0], color="tab:green")
    sns.lineplot(x=x, y=ln_recall, ax=ax[1], color="tab:green")
    sns.lineplot(x=x, y=ln_precision, ax=ax[2], color="tab:green")

    for col in range(columns):
        ax[col].set_xlabel("Signal to Noise Ratio")
        ax[col].invert_xaxis()

    ax[0].set_ylabel("NPA Relative Absolute Error")
    ax[0].set_ylim([0, 0.06])
    ax[1].set_ylabel("Leading Node Recall")
    ax[1].set_ylim([0.92, 1])
    ax[2].set_ylabel("Leading Node Precision")
    ax[2].set_ylim([0.92, 1])

    plt.tight_layout()
    out_file = in_file.replace(".json", ".png")
    plt.savefig(out_file, dpi=300)
    plt.clf()


def plot_snr_combined(in_files):
    results_by_file = dict()
    for in_file in in_files:
        with open(in_file) as f:
            results = json.load(f)

        if in_files[in_file] == "COPD1":
            results_by_dataset = dict()
            results_by_file[in_file] = results_by_dataset
            for idx, result in enumerate(results):
                key = get_key(result)
                if key != "nc":
                    print("Skipping %s" % key)
                    continue

                for dataset in result["dataset_source"]:
                    dataset_source = result["dataset_source"][dataset]
                    dataset_snr = result["dataset_snr"][dataset]

                    if dataset_source not in results_by_dataset:
                        results_by_dataset[dataset_source] = dict()
                    if dataset_snr not in results_by_dataset[dataset_source]:
                        results_by_dataset[dataset_source][dataset_snr] = {
                            "NPA": [], "Leading nodes": [], "Matched boundary edges": []
                        }

                    results_by_dataset[dataset_source][dataset_snr]["NPA"].append(
                        result["npa"][dataset]
                    )
                    results_by_dataset[dataset_source][dataset_snr]["Leading nodes"].append(
                        result["leading_nodes"][dataset]
                    )
                    results_by_dataset[dataset_source][dataset_snr]["Matched boundary edges"].append(
                        result["matched_boundary_edges"][dataset]
                    )

        else:
            results_by_network = dict()
            results_by_file[in_file] = results_by_network
            for idx, result in enumerate(results):
                key = get_key(result)
                if key != "nc":
                    print("Skipping %s" % key)
                    continue

                network = result["network"]
                if network not in results_by_network:
                    results_by_network[network] = dict()

                for dataset in result["dataset_source"]:
                    dataset_source = result["dataset_source"][dataset]
                    dataset_snr = result["dataset_snr"][dataset]

                    if dataset_source not in results_by_network[network]:
                        results_by_network[network][dataset_source] = dict()
                    if dataset_snr not in results_by_network[network][dataset_source]:
                        results_by_network[network][dataset_source][dataset_snr] = {
                            "NPA": [], "Leading nodes": [], "Matched boundary edges": []
                        }

                    results_by_network[network][dataset_source][dataset_snr]["NPA"].append(
                        result["npa"][dataset]
                    )
                    results_by_network[network][dataset_source][dataset_snr]["Leading nodes"].append(
                        result["leading_nodes"][dataset]
                    )
                    results_by_network[network][dataset_source][dataset_snr]["Matched boundary edges"].append(
                        result["matched_boundary_edges"][dataset]
                    )

    # Plot SNR results
    columns = 3
    fig, ax = plt.subplots(1, columns, figsize=(4 * columns, 5))
    colors = ["tab:olive", "tab:cyan", "tab:purple"]
    c_idx = 0

    for in_file in results_by_file:
        x, npa_mrae, ln_precision, ln_recall = [], [], [], []

        if in_files[in_file] == "COPD1":
            results_by_dataset = results_by_file[in_file]
            for dataset_source in results_by_dataset:
                base_results = results_by_dataset[dataset_source][None]

                ref_npa = base_results["NPA"][0]
                if ref_npa == 0.:
                    print("Skipping", dataset_source)
                    continue
                ref_leading_nodes = base_results["Leading nodes"][0]

                for dataset_snr in results_by_dataset[dataset_source]:
                    if dataset_snr is None:
                        continue

                    x.append(dataset_snr)

                    count = 0
                    npa_rae = 0
                    for npa in results_by_dataset[dataset_source][dataset_snr]["NPA"]:
                        count += 1
                        npa_rae += np.abs(npa - ref_npa) / ref_npa
                    npa_mrae.append(npa_rae / count)

                    count = 0
                    recall = 0
                    precision = 0
                    for leading_nodes in results_by_dataset[dataset_source][dataset_snr]["Leading nodes"]:
                        count += 1
                        recall += len(set(leading_nodes) & set(ref_leading_nodes)) / len(ref_leading_nodes)
                        precision += len(set(leading_nodes) & set(ref_leading_nodes)) / len(leading_nodes)
                    ln_recall.append(recall / count)
                    ln_precision.append(precision / count)

        else:
            results_by_network = results_by_file[in_file]
            for network in results_by_network:

                for dataset_source in results_by_network[network]:
                    base_results = results_by_network[network][dataset_source][None]

                    ref_npa = base_results["NPA"][0]
                    if ref_npa == 0.:
                        print("Skipping", dataset_source)
                        continue
                    ref_leading_nodes = base_results["Leading nodes"][0]

                    for dataset_snr in results_by_network[network][dataset_source]:
                        if dataset_snr is None:
                            continue

                        x.append(dataset_snr)

                        count = 0
                        npa_rae = 0
                        for npa in results_by_network[network][dataset_source][dataset_snr]["NPA"]:
                            count += 1
                            npa_rae += np.abs(npa - ref_npa) / ref_npa
                        npa_mrae.append(npa_rae / count)

                        count = 0
                        recall = 0
                        precision = 0
                        for leading_nodes in results_by_network[network][dataset_source][dataset_snr]["Leading nodes"]:
                            count += 1
                            recall += len(set(leading_nodes) & set(ref_leading_nodes)) / len(ref_leading_nodes)
                            precision += len(set(leading_nodes) & set(ref_leading_nodes)) / len(leading_nodes)
                        ln_recall.append(recall / count)
                        ln_precision.append(precision / count)

        label = in_files[in_file]
        if label == "COPD1":
            label = "Mm Apoptosis"
        sns.lineplot(x=x, y=npa_mrae, ax=ax[0], label=label, color=colors[c_idx])
        sns.lineplot(x=x, y=ln_recall, ax=ax[1], color=colors[c_idx])
        sns.lineplot(x=x, y=ln_precision, ax=ax[2], color=colors[c_idx])
        c_idx += 1 if c_idx < len(colors) - 1 else 0

    for col in range(columns):
        ax[col].set_xlabel("Signal to Noise Ratio")
        ax[col].invert_xaxis()

    ax[0].set_ylabel("NPA Relative Absolute Error")
    ax[0].set_ylim([-0.02, 0.32])
    ax[0].legend(loc="upper left")
    ax[1].set_ylabel("Leading Node Recall")
    ax[1].set_ylim([0.73, 1.02])
    ax[2].set_ylabel("Leading Node Precision")
    ax[2].set_ylim([0.73, 1.02])

    plt.tight_layout()
    out_file = "snr_evaluation/snr_evaluation_combined.png"
    plt.savefig(out_file, dpi=300)
    plt.clf()


def plot_copd1_svr_robustness(in_file):
    with open(in_file) as f:
        results = json.load(f)

    results_by_dataset = dict()
    for idx, result in enumerate(results):
        key = get_key(result)
        if key != "nc":
            print("Skipping %s" % key)
            continue

        for dataset in result["dataset_source"]:
            dataset_source = result["dataset_source"][dataset]
            dataset_svr = result["dataset_svr"][dataset]

            if dataset_source not in results_by_dataset:
                results_by_dataset[dataset_source] = dict()
            if dataset_svr not in results_by_dataset[dataset_source]:
                results_by_dataset[dataset_source][dataset_svr] = {
                    "NPA": [], "Leading nodes": [], "Matched boundary edges": []
                }

            results_by_dataset[dataset_source][dataset_svr]["NPA"].append(
                result["npa"][dataset]
            )
            results_by_dataset[dataset_source][dataset_svr]["Leading nodes"].append(
                result["leading_nodes"][dataset]
            )
            results_by_dataset[dataset_source][dataset_svr]["Matched boundary edges"].append(
                result["matched_boundary_edges"][dataset]
            )

    # Plot SVR results
    columns = 3
    fig, ax = plt.subplots(1, columns, figsize=(4 * columns, 6))
    x = []
    relative_npa = []
    ln_precision = []
    ln_recall = []

    for dataset_source in results_by_dataset:
        smallest_svr = min(results_by_dataset[dataset_source].keys())
        base_results = results_by_dataset[dataset_source][smallest_svr]

        ref_npa = base_results["NPA"][0]
        if ref_npa == 0.:
            print("Skipping", dataset_source)
            continue
        ref_leading_nodes = base_results["Leading nodes"][0]

        x.append(smallest_svr)
        relative_npa.append(1.)
        ln_precision.append(1.)
        ln_recall.append(1.)

        for dataset_svr in results_by_dataset[dataset_source]:
            if dataset_svr == smallest_svr:
                continue

            x.append(dataset_svr)

            count = 0
            rel_npa = 0
            for npa in results_by_dataset[dataset_source][dataset_svr]["NPA"]:
                count += 1
                rel_npa += npa / ref_npa
            relative_npa.append(rel_npa / count)

            count = 0
            recall = 0
            precision = 0
            for leading_nodes in results_by_dataset[dataset_source][dataset_svr]["Leading nodes"]:
                count += 1
                recall += len(set(leading_nodes) & set(ref_leading_nodes)) / len(ref_leading_nodes)
                precision += len(set(leading_nodes) & set(ref_leading_nodes)) / len(leading_nodes)
            ln_recall.append(recall / count)
            ln_precision.append(precision / count)

    sns.lineplot(x=x, y=relative_npa, ax=ax[0], color="tab:green")
    sns.lineplot(x=x, y=ln_recall, ax=ax[1], color="tab:green")
    sns.lineplot(x=x, y=ln_precision, ax=ax[2], color="tab:green")

    for col in range(columns):
        ax[col].set_xlabel("Shuffled Value Ratio")

    ax[0].set_ylabel("Relative NPA")
    ax[0].set_ylim([-0.05, 1.05])
    ax[1].set_ylabel("Leading Node Recall")
    ax[1].set_ylim([0.2, 1.05])
    ax[2].set_ylabel("Leading Node Precision")
    ax[2].set_ylim([0.2, 1.05])

    plt.tight_layout()
    out_file = in_file.replace(".json", ".png")
    plt.savefig(out_file, dpi=300)
    plt.clf()


def plot_gen_svr_robustness(in_file):
    with open(in_file) as f:
        results = json.load(f)

    results_by_network = dict()
    for idx, result in enumerate(results):
        key = get_key(result)
        if key != "nc":
            print("Skipping %s" % key)
            continue

        network = result["network"]
        if network not in results_by_network:
            results_by_network[network] = dict()

        for dataset in result["dataset_source"]:
            dataset_source = result["dataset_source"][dataset]
            dataset_svr = result["dataset_svr"][dataset]

            if dataset_source not in results_by_network[network]:
                results_by_network[network][dataset_source] = dict()
            if dataset_svr not in results_by_network[network][dataset_source]:
                results_by_network[network][dataset_source][dataset_svr] = {
                    "NPA": [], "Leading nodes": [], "Matched boundary edges": []
                }

            results_by_network[network][dataset_source][dataset_svr]["NPA"].append(
                result["npa"][dataset]
            )
            results_by_network[network][dataset_source][dataset_svr]["Leading nodes"].append(
                result["leading_nodes"][dataset]
            )
            results_by_network[network][dataset_source][dataset_svr]["Matched boundary edges"].append(
                result["matched_boundary_edges"][dataset]
            )

    # Plot SVR results
    columns = 3
    fig, ax = plt.subplots(1, columns, figsize=(4 * columns, 6))
    x, relative_npa, ln_precision, ln_recall = [], [], [], []

    for network in results_by_network:

        for dataset_source in results_by_network[network]:
            smallest_svr = min(results_by_network[network][dataset_source].keys())
            base_results = results_by_network[network][dataset_source][smallest_svr]

            x.append(smallest_svr)
            relative_npa.append(1.)
            ln_precision.append(1.)
            ln_recall.append(1.)

            ref_npa = base_results["NPA"][0]
            if ref_npa == 0.:
                print("Skipping", dataset_source)
                continue
            ref_leading_nodes = base_results["Leading nodes"][0]

            for dataset_svr in results_by_network[network][dataset_source]:
                if dataset_svr == smallest_svr:
                    continue

                x.append(dataset_svr)

                count = 0
                rel_npa = 0
                for npa in results_by_network[network][dataset_source][dataset_svr]["NPA"]:
                    count += 1
                    rel_npa += npa / ref_npa
                relative_npa.append(rel_npa / count)

                count = 0
                recall = 0
                precision = 0
                for leading_nodes in results_by_network[network][dataset_source][dataset_svr]["Leading nodes"]:
                    count += 1
                    recall += len(set(leading_nodes) & set(ref_leading_nodes)) / len(ref_leading_nodes)
                    precision += len(set(leading_nodes) & set(ref_leading_nodes)) / len(leading_nodes)
                ln_recall.append(recall / count)
                ln_precision.append(precision / count)

    sns.lineplot(x=x, y=relative_npa, ax=ax[0], color="tab:green")
    sns.lineplot(x=x, y=ln_recall, ax=ax[1], color="tab:green")
    sns.lineplot(x=x, y=ln_precision, ax=ax[2], color="tab:green")

    for col in range(columns):
        ax[col].set_xlabel("Shuffled Value Ratio")

    ax[0].set_ylabel("Relative NPA")
    ax[0].set_ylim([-0.05, 1.05])
    ax[1].set_ylabel("Leading Node Recall")
    ax[1].set_ylim([0.2, 1.05])
    ax[2].set_ylabel("Leading Node Precision")
    ax[2].set_ylim([0.2, 1.05])

    plt.tight_layout()
    out_file = in_file.replace(".json", ".png")
    plt.savefig(out_file, dpi=300)
    plt.clf()


def plot_svr_combined(in_files):
    fig, axs = plt.subplots(1, len(in_files), figsize=(4 * len(in_files), 5))
    if len(in_files) == 1:
        axs = [axs]

    for file_idx, in_file in enumerate(in_files):
        with open(in_file) as f:
            results = json.load(f)

        # Plot SVR results
        ax = axs[file_idx]
        x = []
        relative_npa = []

        if in_files[in_file] == "COPD1":
            results_by_dataset = dict()
            for idx, result in enumerate(results):
                key = get_key(result)
                if key != "nc":
                    print("Skipping %s" % key)
                    continue

                for dataset in result["dataset_source"]:
                    dataset_source = result["dataset_source"][dataset]
                    dataset_svr = result["dataset_svr"][dataset]

                    if dataset_source not in results_by_dataset:
                        results_by_dataset[dataset_source] = dict()
                    if dataset_svr not in results_by_dataset[dataset_source]:
                        results_by_dataset[dataset_source][dataset_svr] = {
                            "NPA": [], "Leading nodes": [], "Matched boundary edges": []
                        }

                    results_by_dataset[dataset_source][dataset_svr]["NPA"].append(
                        result["npa"][dataset]
                    )
                    results_by_dataset[dataset_source][dataset_svr]["Leading nodes"].append(
                        result["leading_nodes"][dataset]
                    )
                    results_by_dataset[dataset_source][dataset_svr]["Matched boundary edges"].append(
                        result["matched_boundary_edges"][dataset]
                    )



            for dataset_source in results_by_dataset:
                smallest_svr = min(results_by_dataset[dataset_source].keys())
                base_results = results_by_dataset[dataset_source][smallest_svr]

                ref_npa = base_results["NPA"][0]
                if ref_npa == 0.:
                    print("Skipping", dataset_source)
                    continue

                for dataset_svr in results_by_dataset[dataset_source]:
                    if dataset_svr == smallest_svr:
                        continue

                    x.append(dataset_svr)

                    count = 0
                    rel_npa = 0
                    for npa in results_by_dataset[dataset_source][dataset_svr]["NPA"]:
                        count += 1
                        rel_npa += npa / ref_npa
                    relative_npa.append(rel_npa / count)

                    print(dataset_svr, dataset_source, rel_npa / count)

        else:
            results_by_network = dict()
            for idx, result in enumerate(results):
                key = get_key(result)
                if key != "nc":
                    print("Skipping %s" % key)
                    continue

                network = result["network"]
                if network not in results_by_network:
                    results_by_network[network] = dict()

                for dataset in result["dataset_source"]:
                    dataset_source = result["dataset_source"][dataset]
                    dataset_svr = result["dataset_svr"][dataset]

                    if dataset_source not in results_by_network[network]:
                        results_by_network[network][dataset_source] = dict()
                    if dataset_svr not in results_by_network[network][dataset_source]:
                        results_by_network[network][dataset_source][dataset_svr] = {
                            "NPA": [], "Leading nodes": [], "Matched boundary edges": []
                        }

                    results_by_network[network][dataset_source][dataset_svr]["NPA"].append(
                        result["npa"][dataset]
                    )
                    results_by_network[network][dataset_source][dataset_svr]["Leading nodes"].append(
                        result["leading_nodes"][dataset]
                    )
                    results_by_network[network][dataset_source][dataset_svr]["Matched boundary edges"].append(
                        result["matched_boundary_edges"][dataset]
                    )

            for network in results_by_network:
                for dataset_source in results_by_network[network]:
                    smallest_svr = min(results_by_network[network][dataset_source].keys())
                    base_results = results_by_network[network][dataset_source][smallest_svr]

                    ref_npa = base_results["NPA"][0]
                    if ref_npa == 0.:
                        print("Skipping", dataset_source)
                        continue

                    for dataset_svr in results_by_network[network][dataset_source]:
                        if dataset_svr == smallest_svr:
                            continue

                        x.append(dataset_svr)

                        count = 0
                        rel_npa = 0
                        for npa in results_by_network[network][dataset_source][dataset_svr]["NPA"]:
                            count += 1
                            rel_npa += npa / ref_npa
                        relative_npa.append(rel_npa / count)

        sns.boxplot(x=x, y=relative_npa, ax=ax, color=sns.color_palette()[1])

        ax.set_xlabel("Permutation Rate")
        ax.set_ylabel("Relative NPA")
        ax.set_ylim([-0.05, 1.05])

        title = in_files[in_file]
        if title == "COPD1":
            title = "Mm Apoptosis"
        ax.set_title(title)

    plt.suptitle("Boundary node permutation \"O\"")
    plt.tight_layout()
    out_file = "svr_evaluation/svr_evaluation_combined.png"
    plt.savefig(out_file, dpi=300)
    plt.clf()


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 12})

    # plot_gen_opposite_pruning("ovr_evaluation/ovr_evaluation_npa.json", "NPA-R")
    # plot_gen_opposite_pruning("ovr_evaluation/ovr_evaluation_ba.json", "Barabási–Albert")

    # plot_copd1_mvr_robustness("mvr_evaluation/mvr_evaluation_copd1.json")
    # plot_gen_mvr_robustness("mvr_evaluation/mvr_evaluation_npa.json", "NPA-R")
    # plot_gen_mvr_robustness("mvr_evaluation/mvr_evaluation_ba.json", "Barabási–Albert")

    # plot_copd1_snr_robustness("snr_evaluation/snr_evaluation_copd1.json")
    # plot_gen_snr_robustness("snr_evaluation/snr_evaluation_npa.json")
    # plot_gen_snr_robustness("snr_evaluation/snr_evaluation_ba.json")

    # plot_snr_combined({
    #    "snr_evaluation/snr_evaluation_copd1.json": "COPD1",
    #    "snr_evaluation/snr_evaluation_npa.json": "NPA-R",
    #    "snr_evaluation/snr_evaluation_ba.json": "Barabási–Albert"
    # })

    # plot_copd1_svr_robustness("svr_evaluation/svr_evaluation_copd1.json")
    # plot_gen_svr_robustness("svr_evaluation/svr_evaluation_npa.json")
    # plot_gen_svr_robustness("svr_evaluation/svr_evaluation_ba.json")

    plot_svr_combined({
        "svr_evaluation/svr_evaluation_copd1.json": "COPD1",
        "svr_evaluation/svr_evaluation_npa.json": "NPA-R",
        "svr_evaluation/svr_evaluation_ba.json": "Barabási–Albert"
    })
