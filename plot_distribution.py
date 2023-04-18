import csv

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":

    perturbations = {
        "CS (2m) + Sham (3m)":  0.007157,
        "CS (2m) + Sham (5m)":  0.001191,
        "CS (4m) + Sham (1m)":  0.076700,
        "CS (4m) + Sham (3m)":  0.036512,
        "CS (5m)": 0.102559,
        "CS (7m)": 0.085898
    }

    for perm_path in ["perm_o.tsv", "perm_k.tsv"]:
        with open("output/permutations/" + perm_path) as perm_file:
            dsv_file = csv.reader(perm_file, delimiter="\t")
            next(dsv_file)

            datasets = []
            distribution = dict()

            for line in dsv_file:
                dataset_id = line[0]
                values = [float(v) for v in line[1:] if v != "NA"]
                datasets.append(dataset_id)
                distribution[dataset_id] = values

            plt.clf()
            fig, ax = plt.subplots(nrows=len(datasets), figsize=(6, 3 * len(datasets)))
            if len(datasets) == 1:
                ax = [ax]
            plt.suptitle(perm_path)

            for idx, d in enumerate(datasets):
                distr = distribution[d]
                ax[idx].set_ylabel(d)
                sns.histplot(distr, ax=ax[idx], color='lightblue', stat='density', bins=25)
                sns.kdeplot(distr, ax=ax[idx], color='navy')

                if perturbations[d] is not None:
                    ax[idx].axvline(x=perturbations[d], color='red')

            plt.show()
