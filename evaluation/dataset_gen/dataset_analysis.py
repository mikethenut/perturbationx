import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_gene_distribution(
        axes, dataframe,
        header_labels=('logFC', 't', 'adj.P.Val', 'Gene.symbol'),
        dataset_name=None):

    if dataset_name is not None:
        axes.set_title(dataset_name)
    axes.set_xlabel(header_labels[0])

    # If columns contain adj.P.Val, generate volcano plot
    if header_labels[2] is not None and header_labels[2] in dataframe.columns:
        axes.scatter(x=dataframe[header_labels[0]], y=dataframe[header_labels[2]]
                     .apply(lambda x: -np.log10(x)), s=1, alpha=0.75)
        axes.set_ylabel("-log10(" + header_labels[2] + ")")

    elif header_labels[1] is not None and header_labels[1] in dataframe.columns:
        # else plot logFC vs t
        axes.scatter(x=dataframe[header_labels[0]], y=dataframe[header_labels[1]],
                     s=1, alpha=0.75)
        axes.set_ylabel(header_labels[1])

    else:  # plot histogram
        axes.hist(dataframe[header_labels[0]], bins=100)
        axes.set_ylabel("Frequency")


def plot_datasets(file_paths, header_labels=('logFC', 't', 'adj.P.Val', 'Gene.symbol'),
                  pairwise=True, separator='\t', filename=None):
    dataframes = {}
    for file in file_paths:
        data_df = pd.read_csv(file_paths[file], sep=separator)
        if len(file) > 22:
            file = '...' + file[-22:]
        dataframes[file] = data_df
        if pairwise:
            data_df.set_index(header_labels[3], inplace=True)

    if pairwise:
        fig, axs = plt.subplots(
            len(file_paths), len(file_paths),
            figsize=(3 * len(file_paths), 3 * len(file_paths)))

        for idx, file_x in enumerate(dataframes):
            dataframe_x = dataframes[file_x]

            for idy, file_y in enumerate(dataframes):
                ax = axs[idx, idy]

                if idx == idy:
                    plot_gene_distribution(ax, dataframe_x, header_labels, file_x)

                else:  # Plot logFC vs logFC
                    dataframe_y = dataframes[file_y]

                    gene_set_x = set(dataframe_x.index)
                    gene_set_y = set(dataframe_y.index)
                    common_genes = list(gene_set_x.intersection(gene_set_y))

                    # Reverse axes for clarity
                    logfc_y = dataframe_x.loc[common_genes, header_labels[0]]
                    logfc_x = dataframe_y.loc[common_genes, header_labels[0]]
                    ax.set_xlabel(file_y + " " + header_labels[0])
                    ax.set_ylabel(file_x + " " + header_labels[0])
                    ax.scatter(x=logfc_x, y=logfc_y, s=1, alpha=0.75)

    else:
        fig, axs = plt.subplots(
            len(file_paths), 1,
            figsize=(6, 3 * len(file_paths))
        )

        for idx, file in enumerate(dataframes):
            ax = axs[idx]
            dataframe = dataframes[file]
            plot_gene_distribution(ax, dataframe, header_labels, file)

    plt.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(filename + "_distribution.png")
    plt.close()


if __name__ == "__main__":
    """"
    example_files = {}
    example_folder = "../../data/ExpressionExamples"
    for file_name in os.listdir(example_folder):
        if file_name.endswith(".tsv"):
            example_files[file_name[:-4]] = os.path.join(example_folder, file_name)
    plot_datasets(example_files, pairwise=False, filename="dataset_analysis_plots/examples")

    copd_files = {}
    copd_folder = "../../data/COPD1"
    for file_name in os.listdir(copd_folder):
        if file_name.endswith(".tsv"):
            copd_files[file_name[:-4]] = os.path.join(copd_folder, file_name)
    plot_datasets(copd_files, header_labels=('foldChange', 't', None, 'nodeLabel'),
                  filename="dataset_analysis_plots/copd1")
                  """

    generated_files = {}
    generated_folder = "../../data/COPD1Gen01"
    for file_name in os.listdir(generated_folder):
        if file_name.endswith(".csv"):
            generated_files[file_name[:-4]] = os.path.join(generated_folder, file_name)
    plot_datasets(generated_files, header_labels=('logFC', None, None, 'nodeID'),
                  filename="dataset_analysis_plots/copd1_gen01", separator=',')
