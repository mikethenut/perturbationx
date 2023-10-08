import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_perm_sensitivity(in_files):
    results_by_network = dict()
    for file_idx, in_file in enumerate(in_files):
        with open(in_file) as f:
            results = json.load(f)

        if in_files[in_file] == "COPD1":
            for result in results:
                p_rate = result["permutation_rate"]
                ged = result["ged"]
                core_edges = result["core edges"]

                for dataset in result["npa"]:
                    score = result["npa"][dataset]
                    leading_nodes = result["leading nodes"][dataset]

                    if in_file not in results_by_network:
                        results_by_network[in_file] = dict()
                    if dataset not in results_by_network[in_file]:
                        results_by_network[in_file][dataset] = dict()
                    if p_rate not in results_by_network[in_file][dataset]:
                        results_by_network[in_file][dataset][p_rate] = []

                    results_by_network[in_file][dataset][p_rate].append((core_edges, ged, score, leading_nodes))

        else:
            for result in results:
                network = result["network"]
                p_rate = result["permutation_rate"]
                ged = result["ged"]
                core_edges = result["core edges"]

                for dataset in result["npa"]:
                    score = result["npa"][dataset]
                    leading_nodes = result["leading nodes"][dataset]

                    if in_file not in results_by_network:
                        results_by_network[in_file] = dict()
                    if (network, dataset) not in results_by_network[in_file]:
                        results_by_network[in_file][(network, dataset)] = dict()
                    if p_rate not in results_by_network[in_file][(network, dataset)]:
                        results_by_network[in_file][(network, dataset)][p_rate] = []

                    results_by_network[in_file][(network, dataset)][p_rate].append(
                        (core_edges, ged, score, leading_nodes))

    columns = len(results_by_network)
    fig, axs = plt.subplots(1, columns, figsize=(4 * columns, 5))
    if columns == 1:
        axs = [axs]

    colors = ["tab:olive", "tab:cyan", "tab:purple"]
    c_idx = 0

    for file_idx, in_file in enumerate(results_by_network):
        results_by_dataset = results_by_network[in_file]

        x, npa_mrae, ln_precision, ln_recall = [], [], [], []

        if in_files[in_file] == "COPD1":
            for dataset in results_by_dataset:
                base_results = results_by_dataset[dataset][0.0]

                ref_npa = base_results[0][2]
                ref_leading_nodes = base_results[0][3]
                base_ged = base_results[0][1] / base_results[0][0]

                if base_ged != 0:
                    print("Base GED is not zero for %s." % dataset)
                if ref_npa == 0:
                    print("Skipping %s." % dataset)
                    continue

                x.append(base_ged)
                npa_mrae.append(0)
                ln_recall.append(1)
                ln_precision.append(1)

                for p_rate in results_by_dataset[dataset]:
                    if p_rate == 0.0:
                        continue

                    count = 0
                    avg_ged = 0
                    npa_rae = 0
                    recall = 0
                    precision = 0

                    for core_edges, ged, npa, leading_nodes in results_by_dataset[dataset][p_rate]:
                        count += 1
                        avg_ged += ged / core_edges
                        npa_rae += np.abs(npa - ref_npa) / ref_npa
                        shared_leading_nodes = len(set(leading_nodes) & set(ref_leading_nodes))
                        recall += shared_leading_nodes / len(ref_leading_nodes)
                        precision += shared_leading_nodes / len(leading_nodes)

                        #x.append(ged / core_edges)
                        #npa_mrae.append(np.abs(npa - ref_npa) / ref_npa)
                        #ln_recall.append(shared_leading_nodes / len(ref_leading_nodes))
                        #ln_precision.append(shared_leading_nodes / len(leading_nodes))

                    x.append(p_rate)
                    npa_mrae.append(npa_rae / count)
                    ln_recall.append(recall / count)
                    ln_precision.append(precision / count)

        else:
            for network, dataset in results_by_dataset:
                base_results = results_by_dataset[(network, dataset)][0.0]

                ref_npa = base_results[0][2]
                ref_leading_nodes = base_results[0][3]
                base_ged = base_results[0][1] / base_results[0][0]

                if base_ged != 0:
                    print("Base GED is not zero for %s." % dataset)
                if ref_npa == 0:
                    print("Skipping %s." % dataset)
                    continue

                x.append(base_ged)
                npa_mrae.append(0)
                ln_recall.append(1)
                ln_precision.append(1)

                for p_rate in results_by_dataset[(network, dataset)]:
                    if p_rate == 0.0:
                        continue

                    count = 0
                    avg_ged = 0
                    npa_rae = 0
                    recall = 0
                    precision = 0

                    for core_edges, ged, npa, leading_nodes in results_by_dataset[(network, dataset)][p_rate]:
                        count += 1
                        avg_ged += ged / core_edges
                        npa_rae += np.abs(npa - ref_npa) / ref_npa
                        shared_leading_nodes = len(set(leading_nodes) & set(ref_leading_nodes))
                        recall += shared_leading_nodes / len(ref_leading_nodes)
                        precision += shared_leading_nodes / len(leading_nodes)

                        #x.append(ged / core_edges)
                        #npa_mrae.append(np.abs(npa - ref_npa) / ref_npa)
                        #ln_recall.append(shared_leading_nodes / len(ref_leading_nodes))
                        #ln_precision.append(shared_leading_nodes / len(leading_nodes))

                    x.append(p_rate)
                    npa_mrae.append(npa_rae / count)
                    ln_recall.append(recall / count)
                    ln_precision.append(precision / count)

        label = in_files[in_file]
        if label == "COPD1":
            label = "Mm Apoptosis"

        # sort and round
        # x, npa_mrae, ln_recall, ln_precision = zip(*sorted(zip(x, npa_mrae, ln_recall, ln_precision)))
        # x = np.round(x, 2)

        sns.lineplot(x=x, y=npa_mrae, ax=axs[0], label=label, color=colors[c_idx])
        sns.lineplot(x=x, y=ln_recall, ax=axs[1], color=colors[c_idx])
        sns.lineplot(x=x, y=ln_precision, ax=axs[2], color=colors[c_idx])
        c_idx += 1 if c_idx < len(colors) - 1 else 0

    for col in range(columns):
        axs[col].set_xlabel("Permutation Rate")

    axs[0].set_ylabel("NPA Relative Absolute Error")
    axs[0].set_ylim([-0.02, 0.33])
    axs[0].legend(loc="upper left")
    axs[1].set_ylabel("Leading Node Recall")
    axs[1].set_ylim([0.76, 1.02])
    axs[2].set_ylabel("Leading Node Precision")
    axs[2].set_ylim([0.76, 1.02])

    plt.tight_layout()
    out_file = "graph_edit_distance/sensitivity_combined.pdf"
    plt.savefig(out_file, dpi=300)
    plt.clf()


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 12})
    plot_perm_sensitivity(
        {
            "sensitivity_copd1.json": "COPD1",
            "sensitivity_npa.json": "NPA-R",
            "sensitivity_ba.json": "Barabási–Albert"
        }
    )
