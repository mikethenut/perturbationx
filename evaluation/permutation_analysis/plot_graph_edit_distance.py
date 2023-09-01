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

        sns.violinplot(x=permutation_rates, y=relative_npa, ax=ax[0],
                       color="tab:green", inner="stick")
        sns.violinplot(x=permutation_rates, y=graph_edit_distances, ax=ax[1],
                       color="tab:green", inner="stick")
        sns.violinplot(x=graph_edit_distances, y=relative_npa, ax=ax[2],
                       color="tab:green", inner="stick")

        ax[0].set_xlabel("Permutation Rate")
        ax[0].set_ylabel("Relative NPA")
        ax[1].set_xlabel("Permutation Rate")
        ax[1].set_ylabel("Graph Edit Distance")
        ax[2].set_xlabel("Graph Edit Distance")
        ax[2].set_ylabel("Relative NPA")

        ax[0].set_ylim([-0.05, 1.05])
        ax[1].set_ylim([-20, 520])
        ax[2].set_ylim([-0.1, 1.05])

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

        sns.violinplot(x=permutation_rates, y=relative_npa, ax=ax[0],
                       color="tab:green", inner="stick")
        sns.violinplot(x=permutation_rates, y=relative_graph_edit_distances, ax=ax[1],
                       color="tab:green", inner="stick")

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


def plot_all_k_perms(in_files, stat_files):
    results_by_permutation = dict()
    for file_idx, in_file in enumerate(in_files):
        with open(in_file) as f:
            results = json.load(f)

        if in_files[in_file] == "COPD1":
            for result in results:
                permutation = result["permutation"]
                p_rate = result["permutation_rate"]
                ged = result["ged"]

                for dataset in result["npa"]:
                    score = result["npa"][dataset]

                    if permutation not in results_by_permutation:
                        results_by_permutation[permutation] = dict()
                    if in_file not in results_by_permutation[permutation]:
                        results_by_permutation[permutation][in_file] = dict()
                    if dataset not in results_by_permutation[permutation][in_file]:
                        results_by_permutation[permutation][in_file][dataset] = dict()
                    if p_rate not in results_by_permutation[permutation][in_file][dataset]:
                        results_by_permutation[permutation][in_file][dataset][p_rate] = []

                    results_by_permutation[permutation][in_file][dataset][p_rate].append((ged, score))

        else:
            stat_file = stat_files[in_file]
            with open(stat_file) as f:
                stats = json.load(f)

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
                    if in_file not in results_by_permutation[permutation]:
                        results_by_permutation[permutation][in_file] = dict()
                    if (network, dataset) not in results_by_permutation[permutation][in_file]:
                        results_by_permutation[permutation][in_file][(network, dataset)] = dict()
                    if p_rate not in results_by_permutation[permutation][in_file][(network, dataset)]:
                        results_by_permutation[permutation][in_file][(network, dataset)][p_rate] = []

                    results_by_permutation[permutation][in_file][(network, dataset)][p_rate].append(
                        (core_edge_count, ged, score))

    for permutation in results_by_permutation:
        if permutation is None:
            continue

        permutation_name = permutation.split("_")

        if permutation_name[1] == "full":
            color = sns.color_palette()[0]
        else:
            color = sns.color_palette()[2]

        # Plot results
        columns = len(results_by_permutation[permutation])
        fig, axs = plt.subplots(1, columns, figsize=(4 * columns, 5))
        if columns == 1:
            axs = [axs]

        for file_idx, in_file in enumerate(results_by_permutation[permutation]):
            results_by_file = results_by_permutation[permutation][in_file]
            base_results = results_by_permutation[None][in_file]
            ax = axs[file_idx]

            permutation_rates = []
            relative_npa = []

            if in_files[in_file] == "COPD1":
                for dataset in results_by_file:
                    base_ged, ref_npa = base_results[dataset][0][0]

                    if base_ged != 0:
                        print("Base GED is not zero for %s." % dataset)
                    if ref_npa == 0:
                        print("Skipping %s." % dataset)
                        continue

                    for p_rate in results_by_file[dataset]:
                        count = 0
                        rel_npa = 0
                        for ged, npa in results_by_file[dataset][p_rate]:
                            count += 1
                            rel_npa += npa / ref_npa

                        permutation_rates.append(p_rate)
                        relative_npa.append(rel_npa / count)

            else:
                for network, dataset in results_by_file:
                    edge_count, base_ged, ref_npa = base_results[(network, dataset)][0][0]

                    if base_ged != 0:
                        print("Base GED is not zero for %s." % dataset)
                    if ref_npa == 0:
                        print("Skipping %s." % dataset)
                        continue

                    for p_rate in results_by_file[(network, dataset)]:
                        count = 0
                        rel_npa = 0
                        for edge_count, ged, npa in results_by_file[(network, dataset)][p_rate]:
                            count += 1
                            rel_npa += npa / ref_npa

                        permutation_rates.append(p_rate)
                        relative_npa.append(rel_npa / count)

            sns.boxplot(x=permutation_rates, y=relative_npa, ax=ax, color=color)

            if file_idx == 0:
                ax.set_ylabel(permutation_name[1].capitalize() + " permutation\nRelative NPA")
            ax.set_xlabel("Permutation Rate")
            ax.set_ylim([-0.05, 1.75])

            title = in_files[in_file]
            if title == "COPD1":
                title = "Mm Apoptosis"
            ax.set_title(title)

        plt.suptitle("Core edge permutation \"%s\"" % permutation_name[0].upper())
        plt.tight_layout()
        out_file = "graph_edit_distance/k_perms_combined_%s_%s.png" % (permutation_name[0], permutation_name[1])
        plt.savefig(out_file, dpi=300)
        plt.clf()


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 12})
    # plot_copd1_k_perms("graph_edit_distance/k_perms_copd1.json")
    # plot_gen_k_perms("graph_edit_distance/k_perms_npa.json",
    #                 "../../output/network_stats/network_stats.json")
    # plot_gen_k_perms("graph_edit_distance/k_perms_ba.json",
    #                "../../output/ba_stats_03/ba_stats.json")
    plot_all_k_perms(
        {
            "graph_edit_distance/k_perms_copd1.json": "COPD1",
         "graph_edit_distance/k_perms_npa.json": "NPA-R",
         "graph_edit_distance/k_perms_ba.json": "Barabási–Albert"
         },
        {
            "graph_edit_distance/k_perms_npa.json" : "../../output/network_stats/network_stats.json",
            "graph_edit_distance/k_perms_ba.json" : "../../output/ba_stats_03/ba_stats.json"
        }
    )
