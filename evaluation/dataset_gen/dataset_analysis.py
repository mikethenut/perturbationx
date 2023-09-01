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


def plot_correlations(file_paths, header_labels=('logFC', 't', 'adj.P.Val', 'Gene.symbol'),
                      separator='\t', filename=None):
    dataframes = {}
    for file in file_paths:
        data_df = pd.read_csv(file_paths[file], sep=separator)
        if "dataset" in file:
            file = file[-12:]
        dataframes[file] = data_df
        data_df.set_index(header_labels[3], inplace=True)

    total_plots = len(file_paths) * (len(file_paths) - 1) // 2
    fig, axs = plt.subplots(1, total_plots, figsize=(4 * total_plots, 4))

    files = {idx: file for idx, file in enumerate(dataframes)}
    idx, idy = 0, 1
    for idz in range(total_plots):
        dataframe_x = dataframes[files[idx]]
        dataframe_y = dataframes[files[idy]]
        ax = axs[idz]

        gene_set_x = set(dataframe_x.index)
        gene_set_y = set(dataframe_y.index)
        common_genes = list(gene_set_x.intersection(gene_set_y))

        # Reverse axes for clarity
        logfc_x = dataframe_x.loc[common_genes, header_labels[0]]
        logfc_y = dataframe_y.loc[common_genes, header_labels[0]]
        if header_labels[3] == 'nodeID':
            ax.set_xlabel(files[idx] + " coefficients")
            ax.set_ylabel(files[idy] + " coefficients")
        else:
            ax.set_xlabel(files[idx] + " logFC")
            ax.set_ylabel(files[idy] + " logFC")

        ax.scatter(x=logfc_x, y=logfc_y, s=1, alpha=0.75)

        idy += 1
        if idy == len(file_paths):
            idx += 1
            idy = idx + 1

    plt.tight_layout()
    if filename is not None:
        fig.savefig(filename + ".png")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    copd_files = {}
    copd_folder = "../../data/COPD1"
    for file_name in os.listdir(copd_folder):
        if file_name.endswith(".tsv"):
            copd_files[file_name[:-4]] = os.path.join(copd_folder, file_name)

    generated_files = {}
    generated_folder = "../../data/ExpressionExamplesGen02"
    for file_name in os.listdir(generated_folder):
        if file_name.endswith(".csv"):
            file_name_start = file_name[:10]
            if file_name_start not in generated_files:
                generated_files[file_name_start] = {}
            generated_files[file_name_start][file_name[:-4]] = os.path.join(generated_folder, file_name)

    """"
    example_files = {}
    example_folder = "../../data/ExpressionExamples"
    for file_name in os.listdir(example_folder):
        if file_name.endswith(".tsv"):
            example_files[file_name[:-4]] = os.path.join(example_folder, file_name)
    plot_datasets(example_files, pairwise=False, filename="dataset_analysis_plots/examples")

    plot_datasets(copd_files, header_labels=('foldChange', 't', None, 'nodeLabel'),
                  filename="dataset_analysis_plots/copd1")

    for file_name_start in generated_files:
        plot_datasets(generated_files[file_name_start], header_labels=('logFC', None, None, 'nodeID'),
                      filename="dataset_analysis_plots/" + file_name_start + "_gen02", separator=',')
    """

    copd_selection = ["CS (5m)", "CS (7m)", "CS (4m) + Sham (1m)"]
    copd_selection = {file: copd_files[file] for file in copd_selection}

    plot_correlations(copd_selection, header_labels=('foldChange', None, None, 'nodeLabel'),
                      filename="dataset_analysis_plots/copd1_correlations")

    generated_selection = generated_files["Mm_CFA_Apo"]
    generated_selection = {file: generated_selection[file] for file in generated_selection
                           if "(0)" in file}
    generated_selection = {file: generated_selection[file] for file in list(generated_selection)[:3]}
    plot_correlations(generated_selection, header_labels=('logFC', None, None, 'nodeID'),
                      separator=",", filename="dataset_analysis_plots/gen02_correlations")

