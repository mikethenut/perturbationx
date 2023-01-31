import unittest
import pickle
import os

import numpy as np

from bnpa.npa import preprocess

fixtures_path = os.path.join(os.path.dirname(__file__), '../fixtures')


def assert_sparse_equal(sparseMtx1, sparseMtx2):
    return np.array_equal(sparseMtx1.indptr, sparseMtx2.indptr) and \
           np.array_equal(sparseMtx1.indices, sparseMtx2.indices) and \
           np.allclose(sparseMtx1.data, sparseMtx2.data)


class TestNPAPreprocess(unittest.TestCase):

    def testCountDownstreamEdges(self):
        with open(os.path.join(fixtures_path, "edges.pkl"), 'rb') as ef, \
                open(os.path.join(fixtures_path, "downstream_edge_count.pkl"), 'rb') as decf:

            backbone_edges, downstream_edges = pickle.load(ef)
            downstream_edge_count_true = pickle.load(decf)

            downstream_edge_count = preprocess.count_downstream_edges(133, downstream_edges)
            self.assertEqual(downstream_edge_count, downstream_edge_count_true)

    def testComputeNodeDegree(self):
        with open(os.path.join(fixtures_path, "edges.pkl"), 'rb') as ef, \
                open(os.path.join(fixtures_path, "downstream_edge_count.pkl"), 'rb') as decf, \
                open(os.path.join(fixtures_path, "node_degrees.pkl"), 'rb') as ndf:

            backbone_edges, downstream_edges = pickle.load(ef)
            downstream_edge_count_true = pickle.load(decf)

            degree_total_true, degree_out_true, degree_in_true = pickle.load(ndf)
            degree_total, degree_out, degree_in = preprocess.compute_node_degree(5862, backbone_edges, downstream_edges,
                                                                                 downstream_edge_count_true)
            with self.subTest(msg="degreeTotal"):
                self.assertEqual(degree_total, degree_total_true)

            with self.subTest(msg="degreeOut"):
                self.assertEqual(degree_out, degree_out_true)

            with self.subTest(msg="degreeIn"):
                self.assertEqual(degree_in, degree_in_true)

    def testComputeAdjacency(self):
        with open(os.path.join(fixtures_path, "edges.pkl"), 'rb') as ef, \
                open(os.path.join(fixtures_path, "downstream_edge_count.pkl"), 'rb') as decf, \
                open(os.path.join(fixtures_path, "adjacency.pkl"), 'rb') as adjf:

            backbone_edges, downstream_edges = pickle.load(ef)
            downstream_edge_count_true = pickle.load(decf)

            adjacency_true = pickle.load(adjf)
            adjacency = preprocess.compute_adjacency(5862, backbone_edges, downstream_edges, downstream_edge_count_true)

            self.assertTrue(assert_sparse_equal(adjacency, adjacency_true))

    def testComputeLaplacians(self):
        with open(os.path.join(fixtures_path, "node_degrees.pkl"), 'rb') as ndf, \
                open(os.path.join(fixtures_path, "downstream_edge_count.pkl"), 'rb') as decf, \
                open(os.path.join(fixtures_path, "adjacency.pkl"), 'rb') as adjf, \
                open(os.path.join(fixtures_path, "laplacians.pkl"), 'rb') as lf, \
                open(os.path.join(fixtures_path, "diffusion_matrix.pkl"), 'rb') as tmf:

            node_degree, _, _ = pickle.load(ndf)
            downstream_edge_count = pickle.load(decf)
            adjacency = pickle.load(adjf)

            laplacian_backbone_true, laplacian_backbone_signless_true, laplacian_downstream_true = pickle.load(lf)
            diffusion_matrix_true = pickle.load(tmf)
            laplacian_backbone, laplacian_backbone_signless, laplacian_downstream, diffusion_matrix = \
                preprocess.compute_laplacians(133, 5729, node_degree, downstream_edge_count, adjacency)

            with self.subTest(msg="laplacianBackbone"):
                self.assertTrue(assert_sparse_equal(laplacian_backbone, laplacian_backbone_true))

            with self.subTest(msg="laplacianBackboneSignless"):
                self.assertTrue(assert_sparse_equal(laplacian_backbone_signless, laplacian_backbone_signless_true))

            with self.subTest(msg="laplacianDownstream"):
                self.assertTrue(assert_sparse_equal(laplacian_downstream, laplacian_downstream_true))

            with self.subTest(msg="diffusionMatrix"):
                self.assertTrue(np.allclose(diffusion_matrix, diffusion_matrix_true))


if __name__ == '__main__':
    unittest.main()
