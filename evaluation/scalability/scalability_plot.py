import json

import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    with open("scalability_results.json", "r") as f:
        results = json.load(f)

    net_size_sparse = []
    mem_sparse = []
    time_sparse = []

    net_size_dense = []
    mem_dense = []
    time_dense = []

    for result in results:
        net_size = int(result["network"].split("_")[2]) - int(result["network"].split("_")[0])
        mem_gb = float(result["max. memory"]) / 1024
        time_min = float(result["time"]) / 60

        if result["sparse"] == "True":
            net_size_sparse.append(net_size)
            mem_sparse.append(mem_gb)
            time_sparse.append(time_min)
        else:
            net_size_dense.append(net_size)
            mem_dense.append(mem_gb)
            time_dense.append(time_min)

    fig, ax = plt.subplots(figsize=(7, 6), nrows=2, ncols=1, sharex=True)
    sns.lineplot(x=net_size_sparse, y=mem_sparse, color="tab:blue", ax=ax[1])
    sns.lineplot(x=net_size_dense, y=mem_dense, color="tab:orange", ax=ax[1])
    ax[1].set_xlabel("Number of core and boundary nodes")
    ax[1].set_ylabel("Memory usage (GB)")

    sns.lineplot(x=net_size_sparse, y=time_sparse, color="tab:blue", label="sparse", ax=ax[0])
    sns.lineplot(x=net_size_dense, y=time_dense, color="tab:orange", label="non-sparse", ax=ax[0])
    ax[0].set_xlabel("Number of boundary nodes")
    ax[0].set_ylabel("Runtime (min)")
    ax[0].set_yscale("log")
    fig.tight_layout()
    plt.savefig("scalability.pdf")
    plt.show()
    plt.clf()
