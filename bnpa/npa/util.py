from typing import Optional

import numpy as np
import scipy.sparse as sparse
import pandas as pd

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
    return node_idx, backbone_size


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


def reduce_to_common_nodes(l2: np.ndarray, node_idx: dict, dataset: pd.DataFrame):
    if l2.ndim != 2:
        raise ValueError("Argument l2 must be two-dimensional.")
    elif 'nodeID' not in dataset.columns or 'logFC' not in dataset.columns or 't' not in dataset.columns:
        raise ValueError("Dataset must be contain columns 'nodeID', 'logFC' and 't'.")

    backbone_size = l2.shape[0]
    network_idx = np.array([node_idx[node_name] - backbone_size for node_name in dataset['nodeID'].values
                           if node_name in node_idx])
    l2_reduced = l2[:, network_idx]

    dataset_reduced = dataset[dataset['nodeID'].isin(node_idx)]
    fold_change_reduced = dataset_reduced['logFC'].to_numpy()
    t_statistic_reduced = dataset_reduced['t'].to_numpy()

    return l2_reduced, fold_change_reduced, t_statistic_reduced
