import sys
import warnings
import logging
from pathlib import Path
from typing import Optional
import json

import pandas as pd
import networkx as nx

from bnpa.resources.resources import DEFAULT_DATA_COLS
from bnpa.io.RelationTranslator import RelationTranslator
from bnpa.io.network import read_dsv, validate_nx_graph, write_dsv
from bnpa.npa.core import value_inference, perturbation_amplitude_contributions
from bnpa.npa.preprocess import preprocess_network, preprocess_dataset, infer_graph_attributes
from bnpa.npa.statistics import compute_variances, confidence_interval
from bnpa.npa.permutations import compute_permutations, p_value
from bnpa.result.NPAResultBuilder import NPAResultBuilder


class CausalNetwork:
    __allowed_edge_types = ("core", "boundary", "infer")

    def __init__(self, graph: Optional[nx.DiGraph] = None,
                 relation_translator: Optional[RelationTranslator] = None,
                 inplace=False):
        if relation_translator is None:
            relation_translator = RelationTranslator()
        if type(relation_translator) != RelationTranslator:
            raise TypeError("Argument relation_translator is not a RelationTranslator.")
        self.relation_translator = relation_translator

        if graph is None:
            graph = nx.DiGraph()
        elif type(graph) != nx.DiGraph:
            warnings.warn("Argument graph is not a networkx.DiGraph. It will be converted.")
            graph = nx.DiGraph(graph)
        elif not inplace:
            graph = graph.copy()

        validate_nx_graph(graph, self.__allowed_edge_types)
        self._graph = graph

        # Copy metadata from nx attributes, clear attributes and set default values
        self.metadata = self._graph.graph.copy()
        self._graph.graph.clear()
        self.initialize_metadata()

        self._cytoscape_suid = dict()

    @classmethod
    def from_networkx(cls, graph: nx.DiGraph,
                      relation_translator: Optional[RelationTranslator] = None,
                      inplace=False):
        return cls(graph, relation_translator, inplace)

    @classmethod
    def from_dsv(cls, filepath, edge_type="infer", delimiter='\t',
                 header_cols=DEFAULT_DATA_COLS, relation_translator=None):
        graph = nx.DiGraph()
        graph.add_edges_from(read_dsv(filepath, default_edge_type=edge_type,
                                      delimiter=delimiter, header_cols=header_cols))

        graph.graph["title"] = Path(filepath).stem
        graph.graph["collection"] = Path(filepath).parent.name
        return cls(graph, relation_translator, inplace=True)

    @classmethod
    def from_tsv(cls, filepath, edge_type="infer",
                 header_cols=DEFAULT_DATA_COLS, relation_translator=None):
        return cls.from_dsv(filepath, edge_type, header_cols=header_cols,
                            relation_translator=relation_translator)

    @classmethod
    def from_csv(cls, filepath, edge_type="infer",
                 header_cols=DEFAULT_DATA_COLS, relation_translator=None):
        return cls.from_dsv(filepath, edge_type, delimiter=',', header_cols=header_cols,
                            relation_translator=relation_translator)

    @classmethod
    def from_cyjs_json(cls, filepath, relation_translator=None):
        with open(filepath) as f:
            graph = nx.cytoscape_graph(json.load(f))
            graph.graph["title"] = Path(filepath).stem
            graph.graph["collection"] = Path(filepath).parent.name
            return cls(graph, relation_translator, inplace=True)

    def add_edges_from_dsv(self, filepath, edge_type="infer", delimiter='\t',
                           header_cols=DEFAULT_DATA_COLS):
        self._graph.add_edges_from(read_dsv(filepath, default_edge_type=edge_type,
                                            delimiter=delimiter, header_cols=header_cols))
        validate_nx_graph(self._graph, self.__allowed_edge_types)

    def add_edges_from_tsv(self, filepath, edge_type="infer",
                           header_cols=DEFAULT_DATA_COLS):
        self.add_edges_from_dsv(filepath, edge_type, header_cols=header_cols)

    def add_edges_from_csv(self, filepath, edge_type="infer",
                           header_cols=DEFAULT_DATA_COLS):
        self.add_edges_from_dsv(filepath, edge_type, delimiter=',', header_cols=header_cols)

    def initialize_metadata(self):
        # The user can delete metadata dict
        if self.metadata is None:
            self.metadata = dict()
        # Title and collection are required for Cytoscape
        if "title" not in self.metadata:
            self.metadata["title"] = "Untitled network"
        if "collection" not in self.metadata:
            self.metadata["collection"] = "Untitled collection"

    def copy(self):
        return CausalNetwork(self._graph, self.relation_translator.copy(), inplace=False)

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
        infer_graph_attributes(self._graph, self.relation_translator, verbose)

    def compute_npa(self, datasets: dict, alpha=0.95, permutations=('o', 'k'), p_iters=500, seed=None, verbose=True):
        if verbose:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format="%(asctime)s %(levelname)s -- %(message)s")
            logging.info("PREPROCESSING NETWORK")

        # Copy the graph and set metadata
        prograph = self._graph.copy()
        self.initialize_metadata()
        prograph.graph.update(self.metadata)

        # Preprocess the graph
        prograph, lap, lperms = preprocess_network(prograph, self.relation_translator,
                                                   permutations, p_iters, seed, verbose)
        lb_original = lap['b']
        core_edge_count = sum(1 for src, trg in prograph.edges if prograph[src][trg]["type"] == "core")
        result_builder = NPAResultBuilder.new_builder(prograph, list(datasets.keys()))

        for dataset_id in datasets:
            if verbose:
                logging.info("COMPUTING NPA FOR DATASET '%s'" % dataset_id)

            # Preprocess the dataset
            dataset = datasets[dataset_id]
            if type(dataset) != pd.DataFrame:
                raise ValueError("Dataset %s is not a pandas.DataFrame." % dataset_id)
            if any(col not in dataset.columns for col in ['nodeID', 'logFC', 't']):
                raise ValueError("Dataset %s does not contain columns "
                                 "'nodeID', 'logFC' and 't'." % dataset_id)
            lap['b'], dataset_reduced = preprocess_dataset(lb_original, prograph, dataset, verbose)

            # Compute NPA
            core_coefficients = value_inference(lap['b'], lap['c'], dataset_reduced['logFC'].to_numpy())
            npa, node_contributions = perturbation_amplitude_contributions(
                lap['q'], core_coefficients, core_edge_count
            )
            result_builder.set_global_attributes(dataset_id, ['NPA'], [npa])
            result_builder.set_node_attributes(dataset_id, ['contribution'], [node_contributions])
            result_builder.set_node_attributes(dataset_id, ['coefficient'], [core_coefficients])

            # Compute variances and confidence intervals
            npa_var, node_var = compute_variances(lap, dataset_reduced['stderr'].to_numpy(),
                                                  core_coefficients, core_edge_count)
            npa_ci_lower, npa_ci_upper, _ = confidence_interval(npa, npa_var, alpha)
            result_builder.set_global_attributes(
                dataset_id, ['variance', 'ci_lower', 'ci_upper'], [npa_var, npa_ci_lower, npa_ci_upper]
            )
            node_ci_lower, node_ci_upper, node_p_value = confidence_interval(core_coefficients, node_var, alpha)
            result_builder.set_node_attributes(
                dataset_id, ['variance', 'ci_lower', 'ci_upper', 'p_value'],
                [node_var, node_ci_lower, node_ci_upper, node_p_value]
            )

            # Compute permutation test statistics
            distributions = compute_permutations(lap, lperms, core_edge_count, dataset_reduced['logFC'].to_numpy(),
                                                 permutations, p_iters, seed)
            for p in distributions:
                pv = p_value(npa, distributions[p])
                result_builder.set_global_attributes(dataset_id, ["%s_value" % p], [pv])
                result_builder.set_distribution(dataset_id, "%s_distribution" % p, distributions[p], npa)

        return result_builder.build()

    def to_networkx(self):
        return self._graph.copy()

    def to_dsv(self, filepath, edge_type="all", delimiter='\t',
               data_cols=DEFAULT_DATA_COLS, header=None):
        write_dsv(self._graph, filepath, edge_type, delimiter, data_cols, header)

    def to_tsv(self, filepath, edge_type="all",
               data_cols=DEFAULT_DATA_COLS, header=None):
        self.to_dsv(filepath, edge_type, data_cols=data_cols, header=header)

    def to_csv(self, filepath, edge_type="all",
               data_cols=DEFAULT_DATA_COLS, header=None):
        self.to_dsv(filepath, edge_type, delimiter=",", data_cols=data_cols, header=header)

    def to_cyjs_json(self, filepath, indent=4):
        with open(filepath, 'w') as f:
            json.dump(nx.cytoscape_data(self._graph), f, ensure_ascii=False, indent=indent)
