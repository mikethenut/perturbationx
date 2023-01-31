import unittest
import pickle
import os

from bnpa.io.network import import_network_npa


fixtures_path = os.path.join(os.path.dirname(__file__), '../fixtures')


class TestImport(unittest.TestCase):

    def testNPANetworkImport(self):
        backbone_node_count, downstream_node_count, node_name, \
            backbone_edges, downstream_edges = \
            import_network_npa(os.path.join(fixtures_path, "npa/network_backbone.tsv"),
                               os.path.join(fixtures_path, "npa/network_downstream.tsv"))

        with self.subTest(msg="nodeCount"):
            self.assertEqual((backbone_node_count, downstream_node_count), (133, 5729))

        with open(os.path.join(fixtures_path, "edges.pkl"), 'rb') as ef, \
                open(os.path.join(fixtures_path, "node_name.pkl"), 'rb') as nnf:

            backbone_edges_true, downstream_edges_true = pickle.load(ef)
            node_name_ref = pickle.load(nnf)

            with self.subTest(msg="backboneEdges"):
                backbone_edges_named = [(node_name[src], node_name[trg], val) for src, trg, val in backbone_edges]
                backbone_edges_named_true = [(node_name_ref[src], node_name_ref[trg], val) for src, trg, val in
                                             backbone_edges_true]
                self.assertCountEqual(backbone_edges_named, backbone_edges_named_true)

            with self.subTest(msg="downstreamEdges"):
                downstream_edges_named = [(node_name[src], node_name[trg], val) for src, trg, val in downstream_edges]
                downstream_edges_named_true = [(node_name_ref[src], node_name_ref[trg], val) for src, trg, val in
                                               downstream_edges_true]
                self.assertCountEqual(downstream_edges_named, downstream_edges_named_true)


if __name__ == '__main__':
    unittest.main()
