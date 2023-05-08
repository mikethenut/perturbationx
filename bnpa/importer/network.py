import csv
import warnings

import networkx as nx


def parse_dsv(input_path, delimiter='\t', default_edge_type='infer'):
    edge_list = []

    with open(input_path) as input_file:
        dsv_file = csv.reader(input_file, delimiter=delimiter)

        header = [h.lower() for h in next(dsv_file)]
        if {'subject', 'relation', 'object'}.issubset(header):
            src_idx, rel_idx, trg_idx = header.index('subject'), header.index('relation'), header.index('object')
            typ_idx = header.index('type') if 'type' in header else None
        else:
            src_idx, rel_idx, trg_idx, typ_idx = 0, 1, 2, None
            input_file.seek(0)

        for line in dsv_file:
            src, rel, trg = line[src_idx], line[rel_idx], line[trg_idx]
            typ = line[typ_idx] if typ_idx is not None else default_edge_type
            edge_list.append((src, trg, {'relation': rel, 'type': typ}))

    return edge_list


def validate_init_graph(graph: nx.DiGraph, allowed_edge_types):
    if not type(graph) == nx.DiGraph:
        raise TypeError("Argument graph is not a networkx.Digraph.")

    to_remove = []
    for src, trg, data in graph.edges.data():
        if "relation" not in data:
            warnings.warn("Edge between %s and %s does not have a "
                          "relation specified and will be ignored." % (src, trg))
            to_remove.append((src, trg))
            continue

        if "type" not in data:
            graph[src][trg]["type"] = "infer"
        elif data["type"] not in allowed_edge_types:
            warnings.warn("Unknown type %s of edge %s will be replaced "
                          "with \"infer\"." % (data["type"], str((src, trg))))
            graph[src][trg]["type"] = "infer"

    for src, trg in to_remove:
        graph.remove_edge(src, trg)
        if graph.degree[src] == 0:
            graph.remove_node(src)
        if graph.degree[trg] == 0:
            graph.remove_node(trg)
