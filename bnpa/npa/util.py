from typing import Optional

import numpy as np
import scipy.sparse as sparse

from bnpa.importer.RelationTranslator import RelationTranslator


def enumerate_nodes(backbone_edges: dict, downstream_edges: dict):
    backbone_nodes = {n for nodes in backbone_edges.keys() for n in nodes} | \
                     {nodes[0] for nodes in downstream_edges.keys()}
    downstream_nodes = {nodes[1] for nodes in downstream_edges.keys()}
    node_intersection = backbone_nodes & downstream_nodes
    if len(node_intersection) > 0:
        raise ValueError("The same nodes appear in network backbone and downstream: %s." % str(node_intersection))

    backbone_size = len(backbone_nodes)
    node_idx = {node: idx for idx, node in enumerate(backbone_nodes)} | \
               {node: (backbone_size + idx) for idx, node in enumerate(downstream_nodes)}
    node_name = {idx: node for node, idx in node_idx.items()}
    return node_idx, node_name, backbone_size


def adjacency_matrix(backbone_edges: dict, downstream_edges: dict, node_idx: dict,
                     relation_translator: Optional[RelationTranslator] = None):
    if relation_translator is None:
        relation_translator = RelationTranslator()

    downstream_degree = {src: 0. for src, trg in downstream_edges}
    for (src, trg), rel in downstream_edges.items():
        downstream_degree[src] += abs(relation_translator.translate(rel))

    rows = [node_idx[src] for src, trg in backbone_edges] + [node_idx[src] for src, trg in downstream_edges]
    cols = [node_idx[trg] for src, trg in backbone_edges] + [node_idx[trg] for src, trg in downstream_edges]
    data = [relation_translator.translate(rel) for rel in backbone_edges.values()] + \
           [relation_translator.translate(rel) / downstream_degree[src] for (src, trg), rel in downstream_edges.items()]
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(node_idx), len(node_idx)))


def reduce_to_common_nodes(l2: np.ndarray, network_node_name: dict, fold_change: np.ndarray,
                           t_statistic: np.ndarray, dataset_node_name: dict):
    if l2.ndim != 2:
        raise ValueError("Argument l2 must be two-dimensional.")
    elif fold_change.ndim != 1:
        raise ValueError("Argument fold_change must be one-dimensional.")
    elif t_statistic.ndim != 1:
        raise ValueError("Argument t_statistic must be one-dimensional.")

    backbone_size = l2.shape[0]
    dataset_node_idx = {v: k for k, v in dataset_node_name.items()}

    network_idx = np.array([node_idx - backbone_size for node_idx, node_name in sorted(network_node_name.items())
                            if node_name in dataset_node_idx])
    dataset_idx = np.array([dataset_node_idx[node_name] for node_idx, node_name in sorted(network_node_name.items())
                            if node_name in dataset_node_idx])

    l2_reduced = l2[:, network_idx]
    fold_change_reduced = fold_change[dataset_idx, ]
    t_statistic_reduced = t_statistic[dataset_idx, ]

    return l2_reduced, fold_change_reduced, t_statistic_reduced
