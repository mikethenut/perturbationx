import sys
import csv
import warnings

import numpy as np


def _import_from_npa(backbone_network: str, downstream_network: str):
    backbone_node_count = 0
    backbone_node_idx = dict()
    backbone_edges = []

    downstream_node_count = 0
    downstream_node_idx = dict()
    downstream_edges = []

    node_name = dict()
    node_type = dict()  # 'gene', 'UBE' or 'dUBE' (direct upstream biological entity, i.e. has downstream links)
    edge_type = dict()  # 1 (increasing) or -1 (decreasing)

    with open(backbone_network) as input_file:
        tsv_file = csv.reader(input_file, delimiter="\t")
        header = next(tsv_file)
        src_idx, trg_idx, rel_idx = header.index('subject'), header.index('object'), header.index('relation')

        for line in tsv_file:
            src, trg, rel = line[src_idx], line[trg_idx], int(line[rel_idx])

            if src not in backbone_node_idx:
                node_name[backbone_node_count] = src
                backbone_node_idx[src] = backbone_node_count
                backbone_node_count += 1

            if trg not in backbone_node_idx:
                node_name[backbone_node_count] = trg
                backbone_node_idx[trg] = backbone_node_count
                backbone_node_count += 1

            node_type[backbone_node_idx[src]] = 'UBE'
            node_type[backbone_node_idx[trg]] = 'UBE'

            edge = (backbone_node_idx[src], backbone_node_idx[trg])
            backbone_edges.append(edge)
            edge_type[edge] = rel

    with open(downstream_network) as input_file:
        tsv_file = csv.reader(input_file, delimiter="\t")
        header = next(tsv_file)
        src_idx, trg_idx, rel_idx = header.index('subject'), header.index('object'), header.index('relation')

        for line in tsv_file:
            src = line[src_idx]
            if src not in backbone_node_idx:
                node_name[backbone_node_count] = src
                backbone_node_idx[src] = backbone_node_count
                backbone_node_count += 1
            node_type[backbone_node_idx[src]] = 'dUBE'

        input_file.seek(0)
        next(tsv_file)

        for line in tsv_file:
            src, trg, rel = line[src_idx], line[trg_idx], int(line[rel_idx])

            if trg not in downstream_node_idx:
                node_name[backbone_node_count + downstream_node_count] = trg
                downstream_node_idx[trg] = backbone_node_count + downstream_node_count
                downstream_node_count += 1
            node_type[downstream_node_idx[trg]] = 'gene'

            edge = (backbone_node_idx[src], downstream_node_idx[trg])
            downstream_edges.append(edge)
            edge_type[edge] = rel

    return backbone_node_count, downstream_node_count, backbone_edges, downstream_edges, \
           node_name, node_type, edge_type


def _write_edge_list(edge_list: list, edge_type: dict, node_names=None,
                     output_file="./output/network_edges.tsv", mode='w'):
    with open(output_file, mode, newline='') as file:
        writer = csv.writer(file, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
        for edge in edge_list:
            if node_names is None:
                writer.writerow([edge[0], edge[1], edge_type[edge]])
            else:
                writer.writerow([node_names[edge[0]], node_names[edge[1]], edge_type[edge]])


def _write_matrix(matrix, output_file="./output/matrix.txt", fmt='array2string'):
    print_options = np.get_printoptions()
    threshold, line_width = print_options['threshold'], print_options['linewidth']
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    # TODO: add options

    match fmt:
        case 'array2string':
            with open(output_file, "w") as f:
                f.write(np.array2string(matrix))
        case '_':
            warnings.warn("Warning: unknown format %s in _write_matrix, writing aborted." % fmt)

    np.set_printoptions(threshold=threshold, linewidth=line_width)
