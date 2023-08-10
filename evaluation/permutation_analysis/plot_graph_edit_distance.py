import json

import matplotlib.pyplot as plt
import seaborn as sns


def plot_copd1_k_perms(in_file):
    with open(in_file) as f:
        results = json.load(f)

    results_by_permutation = dict()
    for result in results:
        permutation = result["permutation"]
        p_rate = result["permutation_rate"]
        ged = result["ged"]

        for dataset in result["npa"]:
            score = result["npa"][dataset]

            if permutation not in results_by_permutation:
                results_by_permutation[permutation] = dict()
            if dataset not in results_by_permutation[permutation]:
                results_by_permutation[permutation][dataset] = dict()
            if p_rate not in results_by_permutation[permutation][dataset]:
                results_by_permutation[permutation][dataset][p_rate] = []

            results_by_permutation[permutation][dataset][p_rate].append((ged, score))

    base_results = results_by_permutation[None]
    for permutation in results_by_permutation:
        if permutation is None:
            continue

        # Plot results
        columns = 3
        fig, ax = plt.subplots(1, columns, figsize=(4 * columns, 6))
        permutation_rates = []
        graph_edit_distances = []
        relative_npa = []

        for dataset in results_by_permutation[permutation]:
            base_ged, ref_npa = base_results[dataset][0][0]
            permutation_rates.append(0.)
            graph_edit_distances.append(base_ged)
            relative_npa.append(1.)

            if base_ged != 0:
                print("Base GED is not zero for %s." % dataset)
            if ref_npa == 0:
                print("Skipping %s." % dataset)
                continue

            for p_rate in results_by_permutation[permutation][dataset]:
                count = 0
                rel_npa = 0
                avg_ged = 0
                for ged, npa in results_by_permutation[permutation][dataset][p_rate]:
                    count += 1
                    rel_npa += npa / ref_npa
                    avg_ged += ged

                permutation_rates.append(p_rate)
                graph_edit_distances.append(avg_ged / count)
                relative_npa.append(rel_npa / count)


        sns.lineplot(x=permutation_rates, y=relative_npa, ax=ax[0], color="tab:green")
        sns.lineplot(x=permutation_rates, y=graph_edit_distances, ax=ax[1], color="tab:green")
        sns.lineplot(x=graph_edit_distances, y=relative_npa, ax=ax[2], color="tab:green")

        ax[0].set_xlabel("Permutation Rate")
        ax[0].set_ylabel("Relative NPA")
        ax[1].set_xlabel("Permutation Rate")
        ax[1].set_ylabel("Graph Edit Distance")
        ax[2].set_xlabel("Graph Edit Distance")
        ax[2].set_ylabel("Relative NPA")

        ax[0].set_ylim([0.25, 1.05])
        ax[1].set_ylim([-20, 520])
        ax[2].set_ylim([0.25, 1.05])

        plt.tight_layout()
        out_file = in_file.replace(".json", "_%s.png" % permutation)
        plt.savefig(out_file, dpi=300)
        plt.clf()


def plot_gen_k_perms(in_file, stat_file):
    with open(stat_file) as f:
        stats = json.load(f)

    with open(in_file) as f:
        results = json.load(f)

    results_by_permutation = dict()
    for result in results:
        network = result["network"]
        permutation = result["permutation"]
        p_rate = result["permutation_rate"]
        ged = result["ged"]
        core_edge_count = stats[network]["core_edge_count"]

        for dataset in result["npa"]:
            score = result["npa"][dataset]

            if permutation not in results_by_permutation:
                results_by_permutation[permutation] = dict()
            if (network, dataset) not in results_by_permutation[permutation]:
                results_by_permutation[permutation][(network, dataset)] = dict()
            if p_rate not in results_by_permutation[permutation][(network, dataset)]:
                results_by_permutation[permutation][(network, dataset)][p_rate] = []

            results_by_permutation[permutation][(network, dataset)][p_rate].append((core_edge_count, ged, score))

    base_results = results_by_permutation[None]
    for permutation in results_by_permutation:
        if permutation is None:
            continue

        # Plot results
        columns = 2
        fig, ax = plt.subplots(1, columns, figsize=(6 * columns, 6))
        permutation_rates = []
        relative_graph_edit_distances = []
        relative_npa = []

        for network, dataset in results_by_permutation[permutation]:
            edge_count, base_ged, ref_npa = base_results[(network, dataset)][0][0]

            if base_ged != 0:
                print("Base GED is not zero for %s." % dataset)
            if ref_npa == 0:
                print("Skipping %s." % dataset)
                continue

            for p_rate in results_by_permutation[permutation][(network, dataset)]:
                count = 0
                rel_npa = 0
                avg_rel_ged = 0
                for edge_count, ged, npa in results_by_permutation[permutation][(network, dataset)][p_rate]:
                    count += 1
                    rel_npa += npa / ref_npa
                    avg_rel_ged += ged / edge_count

                permutation_rates.append(p_rate)
                relative_graph_edit_distances.append(avg_rel_ged / count)
                relative_npa.append(rel_npa / count)

        sns.violinplot(x=permutation_rates, y=relative_npa, ax=ax[0], color="tab:green",
                       inner="stick")
        sns.lineplot(x=permutation_rates, y=relative_graph_edit_distances, ax=ax[1], color="tab:green")

        ax[0].set_xlabel("Permutation Rate")
        ax[0].set_ylabel("Relative NPA")
        ax[1].set_xlabel("Permutation Rate")
        ax[1].set_ylabel("Relative Graph Edit Distance")

        ax[0].set_ylim([-0.05, 1.05])
        ax[1].set_ylim([-0.1, 2.1])

        plt.tight_layout()
        out_file = in_file.replace(".json", "_%s.png" % permutation)
        plt.savefig(out_file, dpi=300)
        plt.clf()


if __name__ == "__main__":
    # plot_copd1_k_perms("graph_edit_distance/k_perms_copd1.json")
    # plot_gen_k_perms("graph_edit_distance/k_perms_npa.json",
    #                "../../output/network_stats/network_stats.json")
    plot_gen_k_perms("graph_edit_distance/k_perms_ba.json",
                     "../../output/ba_stats_03/ba_stats.json")
