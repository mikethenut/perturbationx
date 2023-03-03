import csv
import warnings

import numpy as np


def import_dataset_npa(input_path: str):
    node_count = 0
    node_idx = dict()

    node_name = dict()
    fold_change = []
    t_statistic = []

    with open(input_path) as input_file:
        tsv_file = csv.reader(input_file, delimiter="\t")
        header = next(tsv_file)
        label_idx, t_idx, fold_idx = header.index('nodeLabel'), header.index('t'), header.index('foldChange')

        for line in tsv_file:
            node_label, node_t, node_fold = line[label_idx].upper(), float(line[t_idx]), float(line[fold_idx])

            if node_label in node_idx:
                warnings.warn("Warning: multiple data points for %s. "
                              "Only the first instance will be kept." % node_label)
                continue

            node_name[node_count] = node_label
            fold_change.append(node_fold)
            t_statistic.append(node_t)

            node_idx[node_label] = node_count
            node_count += 1

    return node_count, node_name, np.array(t_statistic), np.array(fold_change)
