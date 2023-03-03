import warnings
from collections.abc import Iterable
from typing import Optional

from bnpa.importer.network import parse_dsv
from bnpa.importer.RelationTranslator import RelationTranslator
from bnpa.Dataset import Dataset
from bnpa.npa.NPAResult import NPAResult

from bnpa.npa.util import enumerate_nodes, adjacency_matrix, reduce_to_common_nodes
from bnpa.npa.preprocess import laplacian_matrices, diffusion_matrix, permute_laplacian_k
from bnpa.npa.core import value_diffusion, perturbation_amplitude, perturbation_amplitude_contributions
from bnpa.npa.statistics import backbone_covariance_matrix, backbone_confidence_interval, \
    perturbation_confidence_interval, permutation_test_o, permutation_test_k


class CausalNetwork:
    def __init__(self, backbone_edges=None, downstream_edges=None,
                 relation_translator: Optional[RelationTranslator] = None, preprocess=False):
        if not issubclass(type(backbone_edges), Iterable):
            raise TypeError("Argument backbone_edges is not iterable.")
        if not issubclass(type(backbone_edges), Iterable):
            raise TypeError("Argument downstream_edges is not iterable.")

        self._backbone_edges = dict()
        if backbone_edges is not None:
            self._backbone_edges = dict()
            for s, r, o in backbone_edges:
                if (s, o) in self._backbone_edges:
                    warnings.warn("Multiple relations between %s and %s. "
                                  "Only the first instance will be kept." % (s, o))
                else:
                    self._backbone_edges[(s, o)] = r

        self._downstream_edges = dict()
        if downstream_edges is not None:
            self._downstream_edges = dict()
            for s, r, o in downstream_edges:
                if (s, o.upper()) in self._downstream_edges:
                    warnings.warn("Multiple relations between %s and %s. "
                                  "Only the first instance will be kept." % (s, o.upper()))
                else:
                    self._downstream_edges[(s, o.upper())] = r

        if relation_translator is None:
            self._relation_translator = RelationTranslator()
        else:
            self._relation_translator = relation_translator

        if preprocess:
            self.preprocess_network()
        else:
            self._is_preprocessed = False
            self._node_name = None
            self._l3, self._l2, self._q = None, None, None
            self._l3_permutations = None

    def set_relation_translator(self, relation_translator: RelationTranslator):
        self._relation_translator = relation_translator
        self._is_preprocessed = False

    @classmethod
    def from_tsv(cls, backbone_network, downstream_network, relation_translator=None):
        backbone_edges = parse_dsv(backbone_network)
        downstream_edges = parse_dsv(downstream_network)
        return cls(backbone_edges, downstream_edges, relation_translator)

    @classmethod
    def from_csv(cls, backbone_network, downstream_network, relation_translator=None):
        backbone_edges = parse_dsv(backbone_network, delimiter=',')
        downstream_edges = parse_dsv(downstream_network, delimiter=',')
        return cls(backbone_edges, downstream_edges, relation_translator)

    def add_edge_backbone(self, src, trg, rel):
        if (src, trg) in self._backbone_edges:
            warnings.warn("Relation between %s and %s already exists "
                          "and will be modified." % (src, trg))
        self._backbone_edges[(src, trg)] = rel
        self._is_preprocessed = False

    def modify_edge_backbone(self, src, trg, rel):
        if (src, trg) not in self._backbone_edges:
            raise KeyError("Relation between %s and %s does not exist." % (src, trg))
        self._backbone_edges[(src, trg)] = rel
        self._is_preprocessed = False

    def remove_edge_backbone(self, src, trg):
        if (src, trg) not in self._backbone_edges:
            raise KeyError("Relation between %s and %s does not exist." % (src, trg))
        del self._backbone_edges[(src, trg)]
        self._is_preprocessed = False

    def add_edge_downstream(self, src, trg, rel):
        if (src, trg.upper()) in self._downstream_edges:
            warnings.warn("Relation between %s and %s already exists "
                          "and will be modified." % (src, trg.upper()))
        self._downstream_edges[(src, trg.upper())] = rel
        self._is_preprocessed = False

    def modify_edge_downstream(self, src, trg, rel):
        if (src, trg.upper()) not in self._downstream_edges:
            raise KeyError("Relation between %s and %s does not exist." % (src, trg.upper()))
        self._downstream_edges[(src, trg.upper())] = rel
        self._is_preprocessed = False

    def remove_edge_downstream(self, src, trg):
        if (src, trg.upper()) not in self._downstream_edges:
            raise KeyError("Relation between %s and %s does not exist." % (src, trg.upper()))
        del self._downstream_edges[(src, trg.upper())]
        self._is_preprocessed = False

    def preprocess_network(self):
        node_idx, self._node_name, bb_size = enumerate_nodes(self._backbone_edges, self._downstream_edges)
        adj_mat = adjacency_matrix(self._backbone_edges, self._downstream_edges, node_idx, self._relation_translator)
        self._l3, self._l2, self._q = laplacian_matrices(adj_mat, bb_size)
        self._l3_permutations = permute_laplacian_k(self._l3)
        self._is_preprocessed = True

    def compute_npa(self, dataset: Dataset, statistics=True):
        if not self._is_preprocessed:
            self.preprocess_network()

        l2_reduced, fold_change_reduced, t_statistic_reduced = reduce_to_common_nodes(
            self._l2, self._node_name, dataset.fold_change, dataset.t_statistic, dataset.node_name)
        diff_mat = diffusion_matrix(self._l3, l2_reduced)
        backbone_values = value_diffusion(diff_mat, fold_change_reduced)

        if not statistics:
            return perturbation_amplitude(self._q, backbone_values, len(self._backbone_edges))
        else:
            perturbation, node_contributions = perturbation_amplitude_contributions(
                self._q, backbone_values, len(self._backbone_edges))
            backbone_covariance = backbone_covariance_matrix(diff_mat, fold_change_reduced, t_statistic_reduced)
            npa_var, npa_ci = perturbation_confidence_interval(
                self._q, backbone_values, backbone_covariance, len(self._backbone_edges))
            bb_var, bb_ci, bb_pv = backbone_confidence_interval(backbone_values, backbone_covariance)
            o_pv, o_distribution = permutation_test_o(
                diff_mat, self._q, fold_change_reduced, len(self._backbone_edges))
            k_pv, k_distribution = permutation_test_k(
                self._l3_permutations, l2_reduced, self._q, fold_change_reduced,
                len(self._backbone_edges), perturbation)
            return NPAResult(perturbation, npa_var, npa_ci, node_contributions, backbone_values, bb_var, bb_ci,
                             bb_pv, o_pv, o_distribution, k_pv, k_distribution)
