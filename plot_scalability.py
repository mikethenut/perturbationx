
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_boundary_vs_core_nodes():
    x = np.arange(1000, 30001, 100)

    x_ticks = []
    for memory_size in [8, 16, 32, 64, 128]:
        max_array_size = memory_size / 8
        max_array_size *= 1024 ** 3
        y = [(max_array_size / core_nodes) / core_nodes for core_nodes in x]

        x_clipped = [cn for idx, cn in enumerate(x)
                     if y[idx] <= 150]
        y_clipped = [yn for yn in y if yn <= 150]
        sns.lineplot(x=x_clipped, y=y_clipped, label=f"{memory_size} GB")

        max_core_nodes = np.sqrt(max_array_size / 30)
        x_ticks.append(max_core_nodes)

    ax = plt.gca()
    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    ymax = (30 - ax.get_ylim()[0]) / yrange

    plt.axhline(y=30, color="tab:gray")
    for max_core_nodes in x_ticks:
        plt.axvline(x=max_core_nodes, color="tab:gray", ymax=ymax)

    plt.xticks(x_ticks)
    plt.xlabel("Number of core nodes")
    plt.ylabel("Maximum number of boundary nodes per core node")
    plt.show()


if __name__ == "__main__":
    boundary_nodes_per_core_node = [10, 20, 30, 40, 50]

    x = np.arange(8, 129, 1)
    max_memory = [x / 8 * 1024 ** 3 for x in x]

    for bn_per_cn in boundary_nodes_per_core_node:
        y = [np.sqrt(max_mem / bn_per_cn) for max_mem in max_memory]
        plt.plot(x, y, label="BNR=" + str(bn_per_cn))

    plt.xticks([8, 16, 32, 64, 128])
    for mem in [8, 16, 32, 64, 128]:
        plt.axvline(x=mem, color="tab:gray", alpha=0.3)

    plt.legend(loc="upper left")
    plt.xlabel("System memory (GB)")
    plt.ylabel("Maximum number of core nodes")
    plt.show()
