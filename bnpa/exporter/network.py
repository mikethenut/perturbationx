import sys
import csv

import numpy as np


def write_edge_list(edge_list: list, node_names=None, output_file="./output/network_edges.tsv", mode='w'):
    with open(output_file, mode, newline='') as file:
        writer = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
        for edge in edge_list:
            if node_names is None:
                writer.writerow(list(edge))
            else:
                writer.writerow(list(edge))


def write_matrix(matrix, output_file="./output/matrix.txt", fmt='array2string'):
    print_options = np.get_printoptions()
    threshold, line_width = print_options['threshold'], print_options['linewidth']
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    # TODO: add options

    match fmt:
        case 'array2string':
            with open(output_file, "w") as f:
                f.write(np.array2string(matrix))
        case '_':
            raise ValueError("Unknown format %s, writing aborted." % fmt)

    np.set_printoptions(threshold=threshold, linewidth=line_width)


def write_matrix_alt(matrix, output_file="./output/matrix.txt", digits=5):
    with open(output_file, "w") as f:
        f.write("[")
        for row in matrix:
            f.write("[")
            for el in row:
                f.write(str(el)[:digits])
                f.write(", ")
            f.write("],\n ")
        f.write("]")