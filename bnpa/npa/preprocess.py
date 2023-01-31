import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse


def count_downstream_edges(backbone_node_count: int, downstream_edges: list):
    downstream_edge_count = {idx: 0 for idx in range(backbone_node_count)}
    for de in downstream_edges:
        downstream_edge_count[de[0]] += 1
    return downstream_edge_count


def compute_node_degree(node_count: int, backbone_edges: list, downstream_edges: list, downstream_edge_count: dict):
    degree_in, degree_out = [0.] * node_count, [0.] * node_count

    for be in backbone_edges:
        degree_out[be[0]] += 1
        degree_in[be[1]] += 1

    for node, edge_count in downstream_edge_count.items():
        if edge_count > 0:
            degree_out[node] += 1

    for de in downstream_edges:
        degree_in[de[1]] += 1. / downstream_edge_count[de[0]]

    degree_total = [degree_in[idx] + degree_out[idx] for idx in range(node_count)]

    return degree_total, degree_out, degree_in


def compute_adjacency(node_count: int, backbone_edges: list, downstream_edges: list, downstream_edge_count: dict):
    rows = []
    cols = []
    data = []

    for be in backbone_edges:
        rows.append(be[0])
        cols.append(be[1])
        data.append(be[2])

    for de in downstream_edges:
        rows.append(de[0])
        cols.append(de[1])
        data.append(de[2] / downstream_edge_count[de[0]])

    return sparse.csr_matrix((data, (rows, cols)), shape=(node_count, node_count))


def compute_laplacians(backbone_node_count: int, downstream_node_count: int,
                       node_degree: dict, downstream_edge_count: dict, adj_mat: sparse.spmatrix):

    node_count = backbone_node_count + downstream_node_count
    node_degree_backbone = [node_degree[idx] - (idx in downstream_edge_count and downstream_edge_count[idx] > 0)
                            for idx in range(node_count)]

    laplacian = sparse.diags(node_degree) - adj_mat - adj_mat.transpose()
    laplacian_signless = sparse.diags(node_degree_backbone) + adj_mat + adj_mat.transpose()

    laplacian_backbone = laplacian[:backbone_node_count, :backbone_node_count]
    laplacian_downstream = laplacian[:backbone_node_count, backbone_node_count:]
    laplacian_backbone_signless = laplacian_signless[:backbone_node_count, :backbone_node_count]

    lb_inverse = la.inv(laplacian_backbone.todense())
    diffusion_matrix = np.matmul(lb_inverse, laplacian_downstream.todense().A)

    return laplacian_backbone, laplacian_backbone_signless, laplacian_downstream, diffusion_matrix
