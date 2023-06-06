import itertools
from collections import Counter

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from bnpa.npa.preprocess.permute_adjacency import permute_adjacency_k1, permute_adjacency_k2


def scatter_plot_stacked(datasets, title, show_plot=True, output_file=None):
    plt.clf()
    fig, ax = plt.subplots(nrows=len(datasets), figsize=(6, 3 * len(datasets)),
                           constrained_layout=True, sharex=True, sharey=True)
    plt.suptitle(title)

    if len(datasets) == 1:
        ax = [ax]

    for idx, d in enumerate(datasets):
        ax[idx].set_xscale('log')
        ax[idx].set_yscale('log')
        distr = datasets[d]
        ax[idx].set_ylabel(d)
        ax[idx].scatter(distr[0], distr[1], alpha=0.5, c='blue')

    if output_file is not None:
        plt.savefig(output_file)
    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    no_of_tests = 3
    no_of_iterations = 500
    datasets = dict()
    rng = np.random.default_rng()

    for sample_idx in range(no_of_tests):
        datasets.clear()
        sample = nx.barabasi_albert_graph(100, 2)
        sample_adj = nx.to_numpy_array(sample)
        for x in range(10):
            for y in range(x):
                if sample_adj[x, y] > 0:
                    weight = -1 if rng.uniform() < 0.5 else 1
                    sample_adj[x, y] = weight
                    sample_adj[y, x] = weight

        degree_counter = Counter(np.abs(sample_adj).sum(axis=1))
        datasets["original"] = [degree_counter.keys(), degree_counter.values()]

        perm1 = permute_adjacency_k1(sample_adj, iterations=no_of_iterations)
        perm2 = permute_adjacency_k2(sample_adj, iterations=no_of_iterations)

        degree_counter = Counter(itertools.chain.from_iterable(np.abs(p).sum(axis=1) for p in perm1))
        datasets["k1"] = [degree_counter.keys(), degree_counter.values()]

        degree_counter = Counter(itertools.chain.from_iterable(np.abs(p).sum(axis=1) for p in perm2))
        datasets["k2"] = [degree_counter.keys(), degree_counter.values()]

        scatter_plot_stacked(
            datasets, f"Test {sample_idx + 1}", show_plot=True,
            output_file=f"permutation_test_{sample_idx + 1}.png"
        )
