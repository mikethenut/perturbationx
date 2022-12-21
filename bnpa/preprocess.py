import igraph
import numpy as np


def _generate_graphs(backbone_node_count: int, backbone_edges: list,
                     downstream_node_count: int, downstream_edges: list):

    graph_full = igraph.Graph(n=backbone_node_count + downstream_node_count, edges=backbone_edges + downstream_edges)
    graph_backbone = igraph.Graph(n=backbone_node_count, edges=backbone_edges)
    graph_downstream = igraph.Graph(n=backbone_node_count + downstream_node_count, edges=downstream_edges)

    return graph_full, graph_backbone, graph_downstream


def _compute_matrices(backbone_node_count: int, node_type: dict,
                      graph_backbone: igraph.Graph, graph_downstream: igraph.Graph):

    laplacian_backbone = np.array(graph_backbone.laplacian())
    laplacian_backbone_signless = np.abs(laplacian_backbone)

    has_downstream = np.array([1 if node_type[node_idx] == 'dUBE' else 0 for node_idx in range(backbone_node_count)])
    np.fill_diagonal(laplacian_backbone, np.add(np.diagonal(laplacian_backbone), has_downstream))

    laplacian_downstream = np.array(graph_downstream.laplacian(normalized="left"))
    laplacian_downstream = laplacian_downstream[:backbone_node_count, backbone_node_count:]

    return laplacian_backbone, laplacian_backbone_signless, laplacian_downstream
