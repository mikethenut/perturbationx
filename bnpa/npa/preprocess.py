import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse


def laplacian_matrices(adjacency: sparse.spmatrix, backbone_size: int):
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Argument adjacency is not a square matrix.")
    if backbone_size < 0 or backbone_size >= adjacency.shape[0]:
        raise ValueError("Argument backbone_size is outside of interval [%d, %d]." % (0, adjacency.shape[0]))

    laplacian = - adjacency - adjacency.transpose()
    degree = abs(laplacian).sum(axis=1).A[:, 0]
    laplacian = sparse.diags(degree) + laplacian
    l3 = laplacian[:backbone_size, :backbone_size].todense().A
    l2 = laplacian[:backbone_size, backbone_size:].todense().A

    backbone_adjacency = adjacency[:backbone_size, :backbone_size]
    q = backbone_adjacency + backbone_adjacency.transpose()
    backbone_degree = abs(q).sum(axis=1).A[:, 0]
    q = sparse.diags(backbone_degree) + q
    q = q.todense().A

    return l3, l2, q


def diffusion_matrix(l3: np.ndarray, l2: np.ndarray):
    if l2.ndim != 2:
        raise ValueError("Argument l2 is not two-dimensional.")
    elif l3.ndim != 2 or l3.shape[0] != l3.shape[1]:
        raise ValueError("Argument l3 is not a square matrix.")
    elif l3.shape[0] != l2.shape[0]:
        raise ValueError("Dimensions of l2 and l3 do not match.")

    return np.matmul(la.inv(l3), l2)


def permute_laplacian_k(laplacian: np.ndarray, permutations=500, seed=None):
    if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
        print(laplacian.shape)
        raise ValueError("Argument laplacian is not a square matrix.")
    # WARNING: Some of the generated laplacians might be singular

    if seed is None:
        generator = np.random.default_rng()
    else:
        generator = np.random.default_rng(seed)

    network_size = laplacian.shape[0]
    tril_idx = np.tril_indices(network_size, -1)
    excess_degree = np.sum(np.abs(laplacian), axis=0) - 2 * laplacian.diagonal()

    permuted = []
    for p in range(permutations):
        random_tril = generator.permutation(laplacian[tril_idx])
        random_laplacian = np.zeros((network_size, network_size))
        random_laplacian[tril_idx] = random_tril
        random_laplacian += random_laplacian.transpose()

        isolated_nodes = [idx for idx, deg in enumerate(np.sum(np.abs(random_laplacian), axis=0)) if deg == 0]
        trg_nodes = generator.integers(network_size - 1, size=len(isolated_nodes))
        for n, trg in zip(isolated_nodes, trg_nodes):
            if trg >= n:
                trg += 1
            random_laplacian[n, trg] = 1
            random_laplacian[trg, n] = 1

        np.fill_diagonal(random_laplacian, np.sum(np.abs(random_laplacian), axis=0) + excess_degree)
        permuted.append(random_laplacian)

    return permuted
