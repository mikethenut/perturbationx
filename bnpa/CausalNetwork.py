import os
import warnings
import logging
from typing import Optional

import pandas as pd
import networkx as nx
import py4cytoscape as p4c
from py4cytoscape.py4cytoscape_utils import DEFAULT_BASE_URL

from bnpa.importer.RelationTranslator import RelationTranslator
from bnpa.importer.network import parse_dsv, validate_init_graph
from bnpa.npa.core import value_inference, perturbation_amplitude_contributions
from bnpa.npa.preprocess import preprocess_network, preprocess_dataset, infer_graph_attributes
from bnpa.npa.statistics import compute_variances, confidence_interval
from bnpa.npa.permutations import compute_permutations, p_value
from bnpa.output.NPAResultBuilder import NPAResultBuilder


class CausalNetwork:
    __allowed_edge_types = ("core", "boundary", "infer")

    def __init__(self, graph: nx.DiGraph = None, relation_translator: Optional[RelationTranslator] = None):
        if relation_translator is None:
            relation_translator = RelationTranslator()
        if type(relation_translator) != RelationTranslator:
            raise TypeError("Argument relation_translator is not a RelationTranslator.")
        self.relation_translator = relation_translator

        if graph is None:
            graph = nx.DiGraph()
        validate_init_graph(graph, self.__allowed_edge_types)
        self._graph = graph

        self.metadata = self._graph.graph.copy()
        self.initialize_metadata()

    @classmethod
    def from_dsv(cls, core_filepath=None, boundary_filepath=None, delimiter='\t', relation_translator=None):
        graph = nx.DiGraph()
        if core_filepath is not None:
            graph.add_edges_from(parse_dsv(core_filepath, delimiter=delimiter, default_edge_type="core"))
        if boundary_filepath is not None:
            graph.add_edges_from(parse_dsv(boundary_filepath, delimiter=delimiter, default_edge_type="boundary"))

        graph.graph["title"] = os.path.basename(core_filepath)
        graph.graph["collection"] = os.path.basename(core_filepath)
        cn_instance = cls(graph, relation_translator)
        return cn_instance

    @classmethod
    def from_tsv(cls, core_filepath, boundary_filepath=None, relation_translator=None):
        return cls.from_dsv(core_filepath, boundary_filepath, relation_translator=relation_translator)

    @classmethod
    def from_csv(cls, core_filepath, boundary_filepath=None, relation_translator=None):
        return cls.from_dsv(core_filepath, boundary_filepath, delimiter=',', relation_translator=relation_translator)

    def initialize_metadata(self):
        # TODO: enter networks stats as metadata and forward them to results
        if "title" not in self.metadata:
            self.metadata["title"] = "Untitled network"
        if "collection" not in self.metadata:
            self.metadata["collection"] = "Untitled collection"

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

    def infer_graph_attributes(self, verbose=True):
        infer_graph_attributes(self._graph, verbose)

    def compute_npa(self, datasets: dict, alpha=0.95, permutations=('o', 'k'), p_iters=500, seed=None, verbose=True):
        if verbose:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s -- %(message)s")
            logging.info("PREPROCESSING NETWORK")

        prograph, lap, lperms = preprocess_network(self._graph.copy(), self.relation_translator,
                                                   permutations, p_iters, seed, verbose)
        lb_original = lap['b']
        core_edge_count = sum(1 for src, trg in prograph.edges if prograph[src][trg]["type"] == "core")
        result_builder = NPAResultBuilder.new_builder(prograph, list(datasets.keys()))

        for dataset_id in datasets:
            if verbose:
                logging.info("COMPUTING NPA FOR DATASET '%s'" % dataset_id)

            dataset = datasets[dataset_id]
            if type(dataset) != pd.DataFrame:
                raise ValueError("Dataset %s is not a pandas.DataFrame." % dataset_id)
            if any(col not in dataset.columns for col in ['nodeID', 'logFC', 't']):
                raise ValueError("Dataset %s does not contain columns "
                                 "'nodeID', 'logFC' and 't'." % dataset_id)

            lap['b'], dataset_reduced = preprocess_dataset(lb_original, prograph, dataset, verbose)

            core_coefficients = value_inference(lap['b'], lap['c'], dataset_reduced['logFC'].to_numpy())
            npa, node_contributions = perturbation_amplitude_contributions(
                lap['q'], core_coefficients, core_edge_count
            )
            result_builder.set_global_attributes(dataset_id, ['NPA'], [npa])
            result_builder.set_node_attributes(dataset_id, ['contr'], [node_contributions])
            result_builder.set_node_attributes(dataset_id, ['coeff'], [core_coefficients])

            npa_var, node_var = compute_variances(lap, dataset_reduced['stderr'].to_numpy(),
                                                  core_coefficients, core_edge_count)

            npa_ci_lower, npa_ci_upper, _ = confidence_interval(npa, npa_var, alpha)
            result_builder.set_global_attributes(
                dataset_id, ['var', 'ci_lower', 'ci_upper'], [npa_var, npa_ci_lower, npa_ci_upper]
            )

            node_ci_lower, node_ci_upper, node_p_value = confidence_interval(core_coefficients, node_var, alpha)
            result_builder.set_node_attributes(
                dataset_id, ['var', 'ci_lower', 'ci_upper', 'p_value'],
                [node_var, node_ci_lower, node_ci_upper, node_p_value]
            )

            distributions = compute_permutations(lap, lperms, core_edge_count, dataset_reduced['logFC'].to_numpy(),
                                                 permutations, p_iters, seed)
            for p in distributions:
                pv = p_value(npa, distributions[p])
                result_builder.set_global_attributes(dataset_id, ["%s_value" % p], [pv])
                result_builder.set_distribution(dataset_id, "%s_distribution" % p, distributions[p], npa)

        return result_builder.build()

    def display(self, cytoscape_url=DEFAULT_BASE_URL, display_boundary=False):
        self.initialize_metadata()

        network_suid = p4c.networks.create_network_from_networkx(
            self._graph, base_url=cytoscape_url,
            title=self.metadata['title'],
            collection=self.metadata['collection']
        )

        # TODO: make directed
        # TODO: update with changes

        if not display_boundary:
            boundary_nodes = {trg for src, trg in self._graph.edges if self._graph[src][trg]["type"] == "boundary"}
            boundary_nodes = list(boundary_nodes)
            p4c.hide_nodes(boundary_nodes, network=network_suid, base_url=cytoscape_url)

        return network_suid
