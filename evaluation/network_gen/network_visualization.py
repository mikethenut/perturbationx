import json
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.manifold import MDS


CORE_FEATURES = [
    "core_node_count",
    "core_edge_count",
    "core_negative_edge_count"
]

BOUNDARY_FEATURES = [
    "inner_boundary_node_count",
    "boundary_node_count",
    "boundary_edge_count",
    "boundary_negative_edge_count"
]

DISTANCE_FEATURES = [
    "radius",
    "diameter",
    "average_shortest_path_length"
]

CLUSTERING_FEATURES = [
    "transitivity",
    "average_clustering",
    "degree_assortativity_coefficient",
    "boundary_degree_assortativity_coefficient"
]

DISTRIBUTION_FEATURES = [
    "core_degrees",
    "core_out_degrees",
    "core_in_degrees",
    "boundary_in_degrees",
    "boundary_out_degrees"
]


def paired_scatterplot(statistics_df, hue_column, filename=None):
    features = {}
    for column in statistics_df.columns:
        if column != hue_column:
            if column[0] not in features:
                features[column[0]] = [column[1]]
            else:
                features[column[0]].append(column[1])


    statistics_df_flat = statistics_df
    statistics_df_flat.columns = statistics_df.columns.droplevel("feature_type")
    hue_column_flat = hue_column.droplevel("feature_type")[0]
    statistics_df_npa = statistics_df_flat[statistics_df_flat[hue_column_flat] != "Barabási–Albert"]

    feature_types = list(features.keys())
    for i in range(len(feature_types)):
        feature_type_x = feature_types[i]
        type_x_features = features[feature_type_x]

        for j in range(i+1):
            feature_type_y = feature_types[j]
            rows = len(features[feature_type_y])
            cols = len(type_x_features)

            if feature_type_y == feature_type_x:
                splt = sns.PairGrid(statistics_df_flat, vars=type_x_features,
                                    hue=hue_column_flat, palette="tab10", corner=True)
                plot_name = feature_type_x
            else:
                type_y_features = features[feature_type_y]
                splt = sns.PairGrid(statistics_df_flat, x_vars=type_x_features, y_vars=type_y_features,
                                    hue=hue_column_flat, palette="tab10")
                plot_name = feature_type_y + "_" + feature_type_x

            splt.map_diag(sns.histplot, multiple="stack", bins=10)
            splt.map_offdiag(sns.scatterplot, alpha=0.75)
            if feature_type_x in ["core", "boundary"] and feature_type_y in ["core", "boundary"]:
                for idy, axs in enumerate(splt.axes):
                    for idx, ax in enumerate(axs):
                        if feature_type_y != feature_type_x:
                            sns.regplot(data=statistics_df_npa, x=type_x_features[idx], y=type_y_features[idy],
                                        scatter=False, ax=ax, line_kws={"alpha": 0.5}, color="black", truncate=False)
                            res = linregress(x=statistics_df_npa[type_x_features[idx]],
                                             y=statistics_df_npa[type_y_features[idy]])
                            ax.annotate(f'$r^2 = {res.rvalue ** 2:.2f}$\ny = ${res.slope:.2f}x{res.intercept:+.2f}$',
                                        xy=(.05, .95), xycoords=ax.transAxes, fontsize=8,
                                        color='black', ha='left', va='top')
                        elif idx < idy:
                            sns.regplot(data=statistics_df_npa, x=type_x_features[idx], y=type_x_features[idy],
                                        scatter=False, ax=ax, line_kws={"alpha": 0.5}, color="black", truncate=False)
                            res = linregress(x=statistics_df_npa[type_x_features[idx]],
                                             y=statistics_df_npa[type_x_features[idy]])
                            ax.annotate(f'$r^2 = {res.rvalue ** 2:.2f}$\ny = ${res.slope:.2f}x{res.intercept:+.2f}$',
                                        xy=(.05, .95), xycoords=ax.transAxes, fontsize=8,
                                        color='black', ha='left', va='top')

            splt.add_legend(loc="upper center")
            fig = plt.gcf()
            if feature_type_y != feature_type_x:
                fig.set_size_inches(2.5 * cols, 2.5 * rows + 1.5)
                y_ratio = 1. - 1.5 / (2.5 * rows + 1.5)
                plt.tight_layout(rect=[0, 0, 1, y_ratio])
            else:
                fig.set_size_inches(2.5 * cols, 2.5 * rows)
                plt.tight_layout()
            if filename is not None:
                plt.savefig(filename + "_" + plot_name + "_pairplot.pdf")
            # plt.show()
            plt.close()


def mds_plot(statistics_df, hue_column, filename=None):
    mds_data = statistics_df.drop(hue_column, axis=1)
    mds_data.columns = mds_data.columns.droplevel("feature_type")
    scaled_data = MinMaxScaler().fit_transform(mds_data)

    # regress out core node count
    node_count_idx = mds_data.columns.get_loc("core node count")
    scaled_data_x = scaled_data[:, node_count_idx].reshape(-1, 1)
    scaled_data_y = np.delete(scaled_data, node_count_idx, axis=1)
    regression = LinearRegression()
    regression.fit(scaled_data_x, scaled_data_y)
    residual_data = scaled_data_y - regression.predict(scaled_data_x)

    mds = MDS(n_components=2, random_state=0, normalized_stress='auto')
    mds_transformed = mds.fit_transform(residual_data)

    hue_column_flat = hue_column.droplevel("feature_type")[0]
    mds_df = pd.DataFrame(mds_transformed, index=mds_data.index, columns=["x", "y"])
    mds_df[hue_column_flat] = statistics_df[hue_column]

    plt.figure(figsize=(6, 6))
    splt = sns.scatterplot(data=mds_df, x="x", y="y", hue=hue_column_flat, alpha=0.75)
    sns.move_legend(splt, "upper left")
    plt.xlabel("MDS 1")
    plt.ylabel("MDS 2")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename + "_mds.pdf")
    plt.show()
    plt.close()


def degree_plot(distributions, filename=None):
    for feature in distributions:
        group_count = len(distributions[feature])
        fig, ax = plt.subplots(nrows=group_count, figsize=(6, 3 * group_count), constrained_layout=True, sharex=True)
        suptitle = feature[0].upper() + feature[1:].replace("_", " ")
        if suptitle == "Core degrees":
            suptitle = "Core total degrees"
        plt.suptitle(suptitle)

        if group_count == 1:
            ax = [ax]

        cmap = plt.get_cmap("tab10")
        for idx, group in enumerate(distributions[feature]):
            ax[idx].set_title(group)
            ax[idx].set_xscale("symlog")
            ax[idx].set_yscale("symlog")

            for n in distributions[feature][group]:
                x, y = distributions[feature][group][n]
                ax[idx].plot(x, y, '-o', alpha=0.5, c=cmap(idx))

            if filename is not None:
                plt.savefig(filename + "_" + feature + "_distribution.png", dpi=300)

        # plt.show()
        plt.close()


def transform_labels(labels):
    tr_labs = []
    for label in labels:
        if label == "Dr_ORG_Heart":
            tr_labs.append("Dr ORG Cardiotoxicity")
        elif label == "degree_assortativity_coefficient":
            tr_labs.append("core degree assortativity")
        elif label == "boundary_degree_assortativity_coefficient":
            tr_labs.append("boundary degree assortativity")
        else:
            tr_labs.append(label.replace("_", " "))
    return np.array(tr_labs)


def format_samples(network_statistics, network_names=None):
    if network_names is None:
        network_names = list(network_statistics.keys())

    core_df = pd.DataFrame(np.array(
        [[network_statistics[n][f] for n in network_names] for f in CORE_FEATURES]
    )).set_index(pd.MultiIndex.from_product([["core"], transform_labels(CORE_FEATURES)],
                                            names=["feature_type", "feature_name"]))
    boundary_df = pd.DataFrame(np.array(
        [[network_statistics[n][f] for n in network_names] for f in BOUNDARY_FEATURES]
    )).set_index(pd.MultiIndex.from_product([["boundary"], transform_labels(BOUNDARY_FEATURES)],
                                            names=["feature_type", "feature_name"]))
    distance_df = pd.DataFrame(np.array(
        [[network_statistics[n][f] for n in network_names] for f in DISTANCE_FEATURES]
    )).set_index(pd.MultiIndex.from_product([["distance"], transform_labels(DISTANCE_FEATURES)],
                                            names=["feature_type", "feature_name"]))
    clustering_df = pd.DataFrame(np.array(
        [[network_statistics[n][f] for n in network_names] for f in CLUSTERING_FEATURES]
    )).set_index(pd.MultiIndex.from_product([["clustering"], transform_labels(CLUSTERING_FEATURES)],
                                            names=["feature_type", "feature_name"]))

    combined_df = pd.concat([core_df, boundary_df, distance_df, clustering_df]).T
    combined_df.rename(index={i: network_names[i] for i in range(len(network_names))}, inplace=True)
    return combined_df


def get_distributions(network_statistics, network_names=None):
    if network_names is None:
        network_names = list(network_statistics.keys())

    distributions = {}
    for f in DISTRIBUTION_FEATURES:
        distributions[f] = {}
        for n in network_names:
            feature_distribution = Counter(network_statistics[n][f])
            sorted_fd_keys = np.array(sorted([k for k in feature_distribution.keys() if float(k) > 0]))
            sorted_fd_values = np.array([feature_distribution[k] for k in sorted_fd_keys])
            distributions[f][n] = (sorted_fd_keys, sorted_fd_values)
    return distributions


if __name__ == "__main__":
    network_type_header = pd.MultiIndex.from_product([["metadata"], ["Network group"]],
                                                     names=["feature_type", "feature_name"])

    group_a = ["Hs_CST_Oxidative_Stress", "Mm_CST_Oxidative_Stress", "Hs_CFA_Apoptosis", "Mm_CFA_Apoptosis",
               "Hs_IPN_Epithelial_Innate_Immune_Activation", "Mm_IPN_Epithelial_Innate_Immune_Activation",
               "Hs_IPN_Neutrophil_Signaling", "Mm_IPN_Neutrophil_Signaling"]
    group_b = ["Hs_CPR_Cell_Cycle", "Mm_CPR_Cell_Cycle", "Hs_TRA_ECM_Degradation", "Mm_TRA_ECM_Degradation"]
    group_c = ["Hs_CST_Xenobiotic_Metabolism", "Mm_CST_Xenobiotic_Metabolism", "Hs_CPR_Jak_Stat", "Mm_CPR_Jak_Stat"]
    group_d = ["Dr_ORG_Heart"]

    distribution_features = {f: {} for f in DISTRIBUTION_FEATURES}
    with open("../../output/network_stats/network_stats.json", "r") as in_file:
        npa_stats = json.load(in_file)
        npa_samples = None

        for i, group_networks in enumerate([group_a, group_b, group_c, group_d]):
            group_samples = format_samples(npa_stats, group_networks)
            group_samples[network_type_header] = "NPA group " + chr(ord('A') + i)

            if npa_samples is None:
                npa_samples = group_samples
            else:
                npa_samples = pd.concat([npa_samples, group_samples])

            group_distributions = get_distributions(npa_stats, group_networks)
            for f in group_distributions:
                distribution_features[f]["NPA group " + chr(ord('A') + i)] = group_distributions[f]

    with open("../../output/ba_stats_03/ba_stats.json", "r") as in_file:
        ba_stats = json.load(in_file)
        ba_samples = format_samples(ba_stats)
        ba_samples[network_type_header] = "Barabási–Albert"

        ba_distributions = get_distributions(ba_stats)
        for f in ba_distributions:
            full_ba_distribution = ba_distributions[f]
            partial_ba_distribution = {
                n: full_ba_distribution[n] for idx, n in enumerate(full_ba_distribution) if idx % 10 == 5
            }
            distribution_features[f]["Barabási–Albert"] = partial_ba_distribution

    all_stats = pd.concat([npa_samples, ba_samples])
    # paired_scatterplot(all_stats, network_type_header, "network_gen_plots/network_attributes")
    mds_plot(all_stats, network_type_header, "network_gen_plots/network_distance")
    # degree_plot(distribution_features, "network_gen_plots/degree_distributions")


