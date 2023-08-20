import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from scipy.cluster.hierarchy import linkage, dendrogram


def hierarchical_clustering(data, labels, method='average', color_threshold=None, filename=None):
    clusters = linkage(data, method)
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    dn = dendrogram(clusters, labels=labels, orientation="left", color_threshold=color_threshold)
    fig.tight_layout()
    if filename is not None:
        plt.savefig("clustering_plots/" + filename)
    plt.close()


def barplot(data, filename=None):
    fig = plt.figure(figsize=(6, 4))
    fig.patch.set_facecolor('white')
    plt.bar(range(1, len(data)+1), data, align='center')
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    fig.tight_layout()
    if filename is not None:
        plt.savefig("clustering_plots/" + filename)
    plt.close()


def plot_correlation(data, features, filename=None):
    data_df = pd.DataFrame(data, columns=features)
    corr_df = data_df.corr()
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(corr_df, annot=True, cmap="vlag", vmin=-1, center=0, vmax=1)
    plt.tight_layout()
    if filename is not None:
        plt.savefig("clustering_plots/" + filename)
    plt.close()


def plot_pca(data, filename=None):
    pca = PCA()
    pca.fit(data)
    if filename is not None:
        print("Dataset " + filename + ":")
        print("\tExplained variance ratio: " + str(pca.explained_variance_ratio_))
        print("\tCumulative explained variance ratio: " + str(np.cumsum(pca.explained_variance_ratio_)))
        print("\tSingular values: " + str(pca.singular_values_))
    barplot(pca.explained_variance_ratio_, filename)


def get_feature_cluster_names(agglomeration, feature_names):
    cluster_names = []
    for i in range(agglomeration.n_clusters):
        cluster_features = [feature_names[j] for j in range(len(feature_names))
                            if agglomeration.labels_[j] == i]
        cluster_names.append(", ".join(cluster_features))
    return cluster_names


if __name__ == "__main__":
    with open("../../output/network_stats/network_stats.json", "r") as in_file:
        net_stats = json.load(in_file)

    data = []
    labels = []
    feature_ordering = ["core_node_count", "core_edge_count", "boundary_node_count", "boundary_edge_count",
                        "inner_boundary_node_count", "radius", "diameter", "transitivity", "average_clustering",
                        "average_shortest_path_length", "degree_assortativity_coefficient",
                        "boundary_degree_assortativity_coefficient",
                        "core_negative_edge_ratio", "boundary_negative_edge_ratio"]
    for n in net_stats:
        data_point = []
        data.append(data_point)
        labels.append(n)

        for f in feature_ordering:
            if f == "core_negative_edge_ratio":
                data_point.append(net_stats[n]["core_negative_edge_count"] / net_stats[n]["core_edge_count"])
            elif f == "boundary_negative_edge_ratio":
                data_point.append(net_stats[n]["boundary_negative_edge_count"] / net_stats[n]["boundary_edge_count"])
            else:
                data_point.append(net_stats[n][f])


    data = np.array(data)
    plot_correlation(data, feature_ordering, "correlation_heatmap.png")
    scaled_data = MinMaxScaler().fit_transform(data)
    standardized_data = StandardScaler().fit_transform(data)

    plot_pca(scaled_data, "variance_barplot_scaled.png")
    hierarchical_clustering(
        scaled_data.T, feature_ordering, method="ward",
        color_threshold=1.5, filename="feature_clustering_scaled.png"
    )
    plot_pca(standardized_data, "variance_barplot_standardized.png")
    hierarchical_clustering(
        standardized_data.T, feature_ordering, method="ward",
        color_threshold=4.5, filename="feature_clustering_standardized.png"
    )

    n_clusters = 6
    agglomeration = FeatureAgglomeration(n_clusters=n_clusters)
    scaled_agglomerated_data = agglomeration.fit_transform(scaled_data)
    scaled_agglomerated_features = get_feature_cluster_names(agglomeration, feature_ordering)
    standardized_agglomerated_data = agglomeration.fit_transform(standardized_data)
    standardized_agglomerated_features = get_feature_cluster_names(agglomeration, feature_ordering)

    plot_correlation(scaled_agglomerated_data, scaled_agglomerated_features,
                     "correlation_heatmap_scaled_agglomerated.png")
    plot_pca(scaled_agglomerated_data, "variance_barplot_scaled_agglomerated.png")
    hierarchical_clustering(
        scaled_agglomerated_data.T, scaled_agglomerated_features, method="ward",
        color_threshold=1.5, filename="feature_clustering_scaled_agglomerated.png"
    )
    plot_correlation(standardized_agglomerated_data, standardized_agglomerated_features,
                     "correlation_heatmap_standardized_agglomerated.png")
    plot_pca(standardized_agglomerated_data, "variance_barplot_standardized_agglomerated.png")
    hierarchical_clustering(
        standardized_agglomerated_data.T, standardized_agglomerated_features, method="ward",
        color_threshold=4.5, filename="feature_clustering_standardized_agglomerated.png"
    )

    datasets = {"scaled": scaled_data, "standardized": standardized_data,
                "scaled_agglomerated": scaled_agglomerated_data,
                "standardized_agglomerated": standardized_agglomerated_data}
    thresholds = {
        "scaled": 1.5, "standardized": 5, "scaled_agglomerated": 0.9, "standardized_agglomerated": 3
    }

    pca = PCA(n_components=5)
    for dataset_name in datasets:
        hierarchical_clustering(datasets[dataset_name], labels, color_threshold=thresholds[dataset_name],
                                filename="hierarchical_clustering_" + dataset_name + ".png")
        pca_data = pca.fit_transform(datasets[dataset_name])
        hierarchical_clustering(pca_data, labels,
                                filename="hierarchical_clustering_" + dataset_name + "_pca.png")
