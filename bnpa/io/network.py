import sys
import csv

import numpy as np


def import_network_npa(backbone_network: str, downstream_network: str):
    backbone_node_count = 0
    backbone_node_idx = dict()
    downstream_node_count = 0
    downstream_node_idx = dict()

    node_name = dict()
    backbone_edges = []
    downstream_edges = []

    with open(backbone_network) as input_file:
        tsv_file = csv.reader(input_file, delimiter="\t")
        header = next(tsv_file)
        src_idx, trg_idx, rel_idx = header.index('subject'), header.index('object'), header.index('relation')

        for line in tsv_file:
            src, trg, rel = line[src_idx].upper(), line[trg_idx].upper(), int(line[rel_idx])

            if src not in backbone_node_idx:
                node_name[backbone_node_count] = src
                backbone_node_idx[src] = backbone_node_count
                backbone_node_count += 1

            if trg not in backbone_node_idx:
                node_name[backbone_node_count] = trg
                backbone_node_idx[trg] = backbone_node_count
                backbone_node_count += 1

            backbone_edges.append((backbone_node_idx[src], backbone_node_idx[trg], rel))

    with open(downstream_network) as input_file:
        tsv_file = csv.reader(input_file, delimiter="\t")
        header = next(tsv_file)
        src_idx, trg_idx, rel_idx = header.index('subject'), header.index('object'), header.index('relation')

        for line in tsv_file:
            src = line[src_idx].upper()
            if src not in backbone_node_idx:
                node_name[backbone_node_count] = src
                backbone_node_idx[src] = backbone_node_count
                backbone_node_count += 1

        input_file.seek(0)
        next(tsv_file)

        for line in tsv_file:
            src, trg, rel = line[src_idx].upper(), line[trg_idx].upper(), int(line[rel_idx])

            if trg not in downstream_node_idx:
                node_name[backbone_node_count + downstream_node_count] = trg
                downstream_node_idx[trg] = backbone_node_count + downstream_node_count
                downstream_node_count += 1

            downstream_edges.append((backbone_node_idx[src], downstream_node_idx[trg], rel))

    return backbone_node_count, downstream_node_count, node_name, backbone_edges, downstream_edges


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
