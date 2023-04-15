import os
import warnings
from collections.abc import Iterable
from typing import Optional

import pandas as pd
import networkx as nx

from bnpa.importer.RelationTranslator import RelationTranslator
from bnpa.importer.network import parse_dsv
from bnpa.npa.core import reduce_to_common_nodes, value_diffusion, perturbation_amplitude_contributions
from bnpa.npa.preprocess import preprocess_network
from bnpa.npa.statistics import backbone_covariance_matrix, backbone_confidence_interval, \
    perturbation_confidence_interval, permutation_test_o, permutation_test_k
from bnpa.output.NPAResultBuilder import NPAResultBuilder


class CausalNetwork:
    __allowed_edge_types = ("core", "boundary", "infer")

    def __init__(self, edges=None, relation_translator: Optional[RelationTranslator] = None,
                 metadata: dict = None):
        if not issubclass(type(edges), Iterable):
            raise TypeError("Argument backbone_edges is not iterable.")

        self._graph = nx.DiGraph()
        self._backbone_edges = dict()
        self._downstream_edges = dict()
        dod = dict()  # Format: {src: {trg: {"relation": "+", "type": "core"}}}
        if edges is not None:
            for edge in edges:
                if not 3 <= len(edge) <= 4:
                    warnings.warn("Edge %s is of invalid length and will be ignored." % str(edge))
                    continue

                src, trg, rel = str(edge[0]), str(edge[1]), str(edge[2])

                if src not in dod:
                    dod[src] = dict()
                if trg in dod[src]:
                    warnings.warn("Multiple edges between %s and %s. "
                                  "Only the first instance will be kept." % (src, trg))
                    continue

                dod[src][trg] = {"relation": rel}
                if len(edge) == 4:
                    typ = str(edge[3]).lower()

                    # TODO: remove this
                    if typ == "core":
                        self._backbone_edges[(src, trg)] = rel
                    else:
                        self._downstream_edges[(src, trg)] = rel

                    if typ in self.__allowed_edge_types:
                        dod[src][trg]["type"] = typ
                    else:
                        warnings.warn("Unknown type %s of edge %s will be "
                                      "replaced with \"infer\"." % (typ, str(edge)))
                        dod[src][trg]["type"] = "infer"
                else:
                    dod[src][trg]["type"] = "infer"
                    # TODO: remove this
                    self._downstream_edges[(src, trg)] = rel

        # modify imports
        # change edge modifications

        self.relation_translator = relation_translator \
            if relation_translator is not None \
            else RelationTranslator()

        # TODO: enter default metadata
        self.metadata = metadata \
            if metadata is not None \
            else dict()

    @classmethod
    def from_dsv(cls, core_filepath, boundary_filepath=None, relation_translator=None, delimiter='\t'):
        core_edges = [(a, b, c, "core") for (a, b, c) in
                      parse_dsv(core_filepath, delimiter=delimiter)]
        boundary_edges = [(a, b, c, "boundary") for (a, b, c) in
                          parse_dsv(boundary_filepath, delimiter=delimiter)] \
            if boundary_filepath is not None else []

        metadata = {"name": os.path.basename(core_filepath)}
        return cls(core_edges + boundary_edges, relation_translator, metadata)

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
            result_builder.set_distribution(dataset_id, 'o_distribution', o_distribution, perturbation)
            result_builder.set_distribution(dataset_id, 'k_distribution', k_distribution, perturbation)

        return result_builder.build()

    def visualize(self):
        pass
