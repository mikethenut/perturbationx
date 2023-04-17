import os
import warnings
from collections.abc import Iterable
from typing import Optional

import pandas as pd
import networkx as nx

from bnpa.importer.RelationTranslator import RelationTranslator
from bnpa.importer.network import parse_dsv
from bnpa.npa.core import value_diffusion, perturbation_amplitude_contributions
from bnpa.npa.preprocess import preprocess_network, reduce_to_common_nodes
from bnpa.npa.statistics import backbone_covariance_matrix, backbone_confidence_interval, \
    perturbation_confidence_interval, permutation_test_o, permutation_test_k
from bnpa.output.NPAResultBuilder import NPAResultBuilder


class CausalNetwork:
    __allowed_edge_types = ("core", "boundary", "infer")

    def __init__(self, graph: nx.DiGraph = None, relation_translator: Optional[RelationTranslator] = None):
        if graph is None:
            graph = nx.DiGraph()
        if relation_translator is None:
            relation_translator = RelationTranslator()

        if not type(graph) == nx.DiGraph:
            raise TypeError("Argument graph is not a networkx.Digraph.")
        if not type(relation_translator) == RelationTranslator:
            raise TypeError("Argument relation_translator is not a RelationTranslator.")

        to_remove = []
        for src, trg, data in graph.edges.data():
            if "relation" not in data:
                warnings.warn("Edge between %s and %s does not have a "
                              "relation specified and will be ignored." % (src, trg))
                to_remove.append((src, trg))
                continue

            if "type" not in data:
                graph[src][trg]["type"] = "infer"
            elif data["type"] not in self.__allowed_edge_types:
                warnings.warn("Unknown type %s of edge %s will be replaced "
                              "with \"infer\"." % (data["type"], str((src, trg))))
                graph[src][trg]["type"] = "infer"

        for src, trg in to_remove:
            graph.remove_edge(src, trg)
            if graph.degree[src] == 0:
                graph.remove_node(src)
            if graph.degree[trg] == 0:
                graph.remove_node(trg)

        self._graph = graph
        self.relation_translator = relation_translator

        # TODO: enter networks stats as metadata and forward them to results
        self.metadata = dict()

    @classmethod
    def from_edge_list(cls, edges: Iterable, relation_translator: Optional[RelationTranslator] = None):
        if not issubclass(type(edges), Iterable):
            raise TypeError("Argument edges is not iterable.")

        dod = dict()  # Format: {src: {trg: {"relation": "+", "type": "core"}}}
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

                if typ in cls.__allowed_edge_types:
                    dod[src][trg]["type"] = typ
                else:
                    warnings.warn("Unknown type %s of edge %s will be "
                                  "replaced with \"infer\"." % (typ, str((src, trg))))
                    dod[src][trg]["type"] = "infer"
            else:
                dod[src][trg]["type"] = "infer"

        return cls(nx.DiGraph(dod), relation_translator)

    @classmethod
    def from_dsv(cls, core_filepath, boundary_filepath=None, relation_translator=None, delimiter='\t'):
        core_edges = [(a, b, c, "core") for (a, b, c) in
                      parse_dsv(core_filepath, delimiter=delimiter)]
        boundary_edges = [(a, b, c, "boundary") for (a, b, c) in
                          parse_dsv(boundary_filepath, delimiter=delimiter)] \
            if boundary_filepath is not None else []

        cn_instance = cls.from_edge_list(core_edges + boundary_edges, relation_translator)
        cn_instance.metadata["name"] = os.path.basename(core_filepath)
        return cn_instance

    @classmethod
    def from_tsv(cls, core_filepath, boundary_filepath=None, relation_translator=None):
        return cls.from_dsv(core_filepath, boundary_filepath, relation_translator)

    @classmethod
    def from_csv(cls, core_filepath, boundary_filepath=None, relation_translator=None):
        return cls.from_dsv(core_filepath, boundary_filepath, relation_translator, delimiter=',')

    def copy(self):
        return CausalNetwork(self._graph.copy(), self.relation_translator.copy())

    def number_of_nodes(self):
        return self._graph.number_of_nodes()

    def nodes(self):
        return list(self._graph.nodes(data=True))

    def number_of_edges(self, typ=None):
        if typ is not None:
            return sum(1 for e in self._graph.edges.data() if e[2]["type"] == typ)
        return self._graph.number_of_edges()

    def edges(self, typ=None):
        if typ is not None:
            return [e for e in self._graph.edges.data() if e[2]["type"] == typ]
        return list(self._graph.edges.data())

    def add_edge(self, src, trg, rel, typ="infer"):
        if self._graph.has_edge(src, trg):
            warnings.warn("Edge between %s and %s already exists "
                          "and will be modified." % (src, trg))
        if typ in self.__allowed_edge_types:
            self._graph.add_edge(src, trg, relation=rel, type=typ)
        else:
            warnings.warn("Unknown type %s of edge %s will be "
                          "replaced with \"infer\"." % (typ, str((src, trg))))
            self._graph.add_edge(src, trg, relation=rel, type=typ)

    def modify_edge(self, src, trg, rel=None, typ=None):
        if not self._graph.has_edge(src, trg):
            raise KeyError("Edge between %s and %s does not exist." % (src, trg))
        if rel is not None:
            self._graph[src][trg]["relation"] = rel
        if typ is not None:
            if typ not in self.__allowed_edge_types:
                self._graph[src][trg]["type"] = typ
            else:
                warnings.warn("Unknown type %s of edge %s "
                              "will be ignored." % (typ, str((src, trg))))

    def remove_edge(self, src, trg):
        if not self._graph.has_edge(src, trg):
            raise KeyError("Edge between %s and %s does not exist." % (src, trg))
        self._graph.remove_edge(src, trg)

        if self._graph.degree[src] == 0:
            self._graph.remove_node(src)
        if self._graph.degree[trg] == 0:
            self._graph.remove_node(trg)

    def compute_npa(self, datasets: dict):
        prograph, lps, lperms = preprocess_network(self._graph.copy(), self.relation_translator)
        core_edge_count = sum(1 for src, trg in prograph.edges if prograph[src][trg]["type"] == "core")
        result_builder = NPAResultBuilder.new_builder(prograph, list(datasets.keys()))

        for dataset_id in datasets:
            dataset = datasets[dataset_id]
            if type(dataset) != pd.DataFrame:
                raise ValueError("Dataset %s is not a pandas.DataFrame." % dataset_id)
            if any(col not in dataset.columns for col in ['nodeID', 'logFC', 't']):
                raise ValueError("Dataset %s does not contain columns "
                                 "'nodeID', 'logFC' and 't'." % dataset_id)

            lb_reduced, dataset_reduced = reduce_to_common_nodes(lps['b'], prograph, dataset)

            backbone_values = value_diffusion(lps['c'], lb_reduced, dataset_reduced['logFC'].to_numpy())
            perturbation, node_contributions = perturbation_amplitude_contributions(
                lps['q'], backbone_values, core_edge_count
            )
            result_builder.set_global_attributes(dataset_id, ['NPA'], [perturbation])
            result_builder.set_node_attributes(
                dataset_id, ['contributions', 'coefficients'], [node_contributions, backbone_values]
            )

            backbone_covariance = backbone_covariance_matrix(
                lps['c'], lb_reduced, dataset_reduced['logFC'].to_numpy(), dataset_reduced['t'].to_numpy()
            )
            npa_var, npa_ci_lower, npa_ci_upper = perturbation_confidence_interval(
                lps['q'], backbone_values, backbone_covariance, core_edge_count
            )
            result_builder.set_global_attributes(
                dataset_id, ['variance', 'ci_lower', 'ci_upper'], [npa_var, npa_ci_lower, npa_ci_upper]
            )

            node_var, node_ci_lower, node_ci_upper, node_p_value = backbone_confidence_interval(
                backbone_values, backbone_covariance)
            result_builder.set_node_attributes(
                dataset_id, ['coefficient variance', 'ci_lower', 'ci_upper', 'p_value'],
                [node_var, node_ci_lower, node_ci_upper, node_p_value]
            )

            o_pv, o_distribution = permutation_test_o(
                lps['c'], lb_reduced, lps['q'], dataset_reduced['logFC'].to_numpy(), core_edge_count
            )
            k_pv, k_distribution = permutation_test_k(
                lperms['k'], lb_reduced, lps['q'], dataset_reduced['logFC'].to_numpy(), core_edge_count, perturbation
            )
            result_builder.set_global_attributes(dataset_id, ['o_value', 'k_value'], [o_pv, k_pv])
            result_builder.set_distribution(dataset_id, 'o_distribution', o_distribution, perturbation)
            result_builder.set_distribution(dataset_id, 'k_distribution', k_distribution, perturbation)

        return result_builder.build()

    def display(self):
        # TODO
        pass
