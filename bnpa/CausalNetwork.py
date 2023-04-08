import warnings
from collections.abc import Iterable
from typing import Optional

import pandas as pd
import igraph as ig

from bnpa.importer.RelationTranslator import RelationTranslator
from bnpa.importer.network import parse_dsv
from bnpa.npa.core import reduce_to_common_nodes, value_diffusion, perturbation_amplitude_contributions
from bnpa.npa.preprocess import preprocess_network
from bnpa.npa.statistics import backbone_covariance_matrix, backbone_confidence_interval, \
    perturbation_confidence_interval, permutation_test_o, permutation_test_k
from bnpa.output.NPAResultBuilder import NPAResultBuilder


class CausalNetwork:
    def __init__(self, backbone_edges=None, downstream_edges=None,
                 relation_translator: Optional[RelationTranslator] = None):
        if not issubclass(type(backbone_edges), Iterable):
            raise TypeError("Argument backbone_edges is not iterable.")
        if not issubclass(type(backbone_edges), Iterable):
            raise TypeError("Argument downstream_edges is not iterable.")

        # change input parameters
        # modify imports
        # create igraph from tuple list
        # change edge modifications

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
                if (s, o) in self._downstream_edges:
                    warnings.warn("Multiple relations between %s and %s. "
                                  "Only the first instance will be kept." % (s, o))
                else:
                    self._downstream_edges[(s, o)] = r

        self.relation_translator = relation_translator if relation_translator is not None else RelationTranslator()
        self.metadata = dict()

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

    def modify_edge_backbone(self, src, trg, rel):
        if (src, trg) not in self._backbone_edges:
            raise KeyError("Relation between %s and %s does not exist." % (src, trg))
        self._backbone_edges[(src, trg)] = rel

    def remove_edge_backbone(self, src, trg):
        if (src, trg) not in self._backbone_edges:
            raise KeyError("Relation between %s and %s does not exist." % (src, trg))
        del self._backbone_edges[(src, trg)]

    def add_edge_downstream(self, src, trg, rel):
        if (src, trg) in self._downstream_edges:
            warnings.warn("Relation between %s and %s already exists "
                          "and will be modified." % (src, trg))
        self._downstream_edges[(src, trg)] = rel

    def modify_edge_downstream(self, src, trg, rel):
        if (src, trg) not in self._downstream_edges:
            raise KeyError("Relation between %s and %s does not exist." % (src, trg))
        self._downstream_edges[(src, trg)] = rel

    def remove_edge_downstream(self, src, trg):
        if (src, trg) not in self._downstream_edges:
            raise KeyError("Relation between %s and %s does not exist." % (src, trg))
        del self._downstream_edges[(src, trg)]

    def compute_npa(self, datasets: dict):
        node_idx, l3, l2, q, l3_permutations = preprocess_network(
            self._backbone_edges, self._downstream_edges, self.relation_translator)
        bb_edge_count = len(self._backbone_edges)
        backbone_nodes = [node for node in node_idx if node_idx[node] < q.shape[0]]
        result_builder = NPAResultBuilder.new_builder(list(datasets.keys()), backbone_nodes)

        for dataset_id in datasets:
            dataset = datasets[dataset_id]
            if type(dataset) != pd.DataFrame:
                raise ValueError("Dataset %s is not a pandas.DataFrame." % dataset_id)
            if any(col not in dataset.columns for col in ['nodeID', 'logFC', 't']):
                raise ValueError("Dataset %s does not contain columns "
                                 "'nodeID', 'logFC' and 't'." % dataset_id)

            l2_reduced, dataset_reduced = reduce_to_common_nodes(l2, node_idx, dataset)

            backbone_values = value_diffusion(l3, l2_reduced, dataset_reduced['logFC'].to_numpy())
            perturbation, node_contributions = perturbation_amplitude_contributions(
                q, backbone_values, bb_edge_count)
            result_builder.set_global_attributes(dataset_id, ['NPA'], [perturbation])
            result_builder.set_node_attributes(dataset_id, ['contributions', 'coefficients'],
                                               [node_contributions, backbone_values])

            backbone_covariance = backbone_covariance_matrix(
                l3, l2_reduced, dataset_reduced['logFC'].to_numpy(), dataset_reduced['t'].to_numpy())
            npa_var, npa_ci_lower, npa_ci_upper = perturbation_confidence_interval(
                q, backbone_values, backbone_covariance, bb_edge_count)
            result_builder.set_global_attributes(dataset_id, ['variance', 'ci_lower', 'ci_upper'],
                                                 [npa_var, npa_ci_lower, npa_ci_upper])

            node_var, node_ci_lower, node_ci_upper, node_p_value = backbone_confidence_interval(
                backbone_values, backbone_covariance)
            result_builder.set_node_attributes(dataset_id, ['coefficient variance', 'ci_lower', 'ci_upper', 'p_value'],
                                               [node_var, node_ci_lower, node_ci_upper, node_p_value])

            o_pv, o_distribution = permutation_test_o(l3, l2_reduced, q,
                                                      dataset_reduced['logFC'].to_numpy(), bb_edge_count)
            k_pv, k_distribution = permutation_test_k(l3_permutations, l2_reduced, q,
                                                      dataset_reduced['logFC'].to_numpy(), bb_edge_count, perturbation)
            result_builder.set_global_attributes(dataset_id, ['o_value', 'k_value'], [o_pv, k_pv])
            result_builder.set_permutations(dataset_id, ['o_distribution', 'k_distribution'],
                                            [o_distribution, k_distribution])

        return result_builder.build()

    def visualize(self):
        pass
