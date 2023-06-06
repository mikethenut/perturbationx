import sys
import warnings
import logging
from pathlib import Path
from typing import Optional
import json

import networkx as nx

from bnpa.resources.resources import DEFAULT_DATA_COLS
from bnpa.io.RelationTranslator import RelationTranslator
from bnpa.io.network_io import read_dsv, validate_nx_graph, write_dsv
from bnpa.npa.preprocess import preprocess_network, preprocess_dataset, network_matrices, permute_adjacency
from bnpa.npa import core, statistics, permutation_tests
from bnpa.result.NPAResultBuilder import NPAResultBuilder


class CausalNetwork:
    __allowed_edge_types = ("core", "boundary", "infer")

    def __init__(self, graph: Optional[nx.DiGraph] = None,
                 relation_translator: Optional[RelationTranslator] = None,
                 inplace=False):
        if relation_translator is None:
            relation_translator = RelationTranslator()
        if not isinstance(relation_translator, RelationTranslator):
            raise TypeError("Argument relation_translator is not a RelationTranslator.")
        self.relation_translator = relation_translator

        if graph is None:
            graph = nx.DiGraph()
        elif not isinstance(graph, nx.DiGraph):
            warnings.warn("Argument graph is not a networkx.DiGraph, and will be converted.")
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
        preprocess_network.infer_graph_attributes(self._graph, self.relation_translator, verbose)

    def compute_npa(self, datasets: dict, legacy=False, strict_pruning=False, alpha=0.95,
                    permutations=('o', 'k2'), p_iters=500, p_rate=1., seed=None, verbose=True):

        # Preprocess the datasets
        for dataset_id in datasets:
            datasets[dataset_id] = preprocess_dataset.format_dataset(datasets[dataset_id])

        if verbose:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format="%(asctime)s %(levelname)s -- %(message)s")
            logging.info("PREPROCESSING NETWORK")

        # Copy the graph and set metadata
        prograph = self._graph.to_directed()
        self.initialize_metadata()
        prograph.graph.update(self.metadata)

        # Preprocess the graph
        preprocess_network.infer_graph_attributes(prograph, self.relation_translator, verbose)
        core_edge_count = sum(1 for src, trg in prograph.edges if prograph[src][trg]["type"] == "core")
        adj_b, adj_c = network_matrices.generate_adjacency(prograph)
        adj_perms = permute_adjacency.permute_adjacency(
            adj_c, permutations=permutations, permutation_rate=p_rate, iterations=p_iters, seed=seed
        )

        result_builder = NPAResultBuilder.new_builder(prograph, list(datasets.keys()))
        for dataset_id in datasets:
            dataset = datasets[dataset_id]
            if verbose:
                logging.info("COMPUTING NPA FOR DATASET '%s'" % dataset_id)

            # Prepare data
            if legacy:
                lap_b, lap_c, lap_q, lap_perms = network_matrices.generate_laplacians(adj_b, adj_c, adj_perms)
                lap_b, dataset = preprocess_dataset.prune_network_dataset(
                    prograph, lap_b, dataset, dataset_id, strict_pruning, verbose
                )
            else:
                lap_b, dataset = preprocess_dataset.prune_network_dataset(
                    prograph, adj_b, dataset, dataset_id, strict_pruning, verbose
                )
                lap_b, lap_c, lap_q, lap_perms = network_matrices.generate_laplacians(lap_b, adj_c, adj_perms)

            # Compute NPA
            core_coefficients = core.value_inference(lap_b, lap_c, dataset["logFC"].to_numpy())
            npa, node_contributions = core.perturbation_amplitude_contributions(
                lap_q, core_coefficients, core_edge_count
            )
            result_builder.set_global_attributes(dataset_id, ["NPA"], [npa])
            result_builder.set_node_attributes(dataset_id, ["contribution"], [node_contributions])
            result_builder.set_node_attributes(dataset_id, ["coefficient"], [core_coefficients])

            # Compute variances and confidence intervals
            npa_var, node_var = statistics.compute_variances(
                lap_b, lap_c, lap_q, dataset["stderr"].to_numpy(), core_coefficients, core_edge_count)
            npa_ci_lower, npa_ci_upper, _ = statistics.confidence_interval(npa, npa_var, alpha)
            result_builder.set_global_attributes(
                dataset_id, ["variance", "ci_lower", "ci_upper"], [npa_var, npa_ci_lower, npa_ci_upper]
            )
            node_ci_lower, node_ci_upper, node_p_value = \
                statistics.confidence_interval(core_coefficients, node_var, alpha)
            result_builder.set_node_attributes(
                dataset_id, ["variance", "ci_lower", "ci_upper", "p_value"],
                [node_var, node_ci_lower, node_ci_upper, node_p_value]
            )

            # Compute permutation test statistics
            distributions = permutation_tests.permutation_tests(
                lap_b, lap_c, lap_q, lap_perms, core_edge_count,
                dataset["logFC"].to_numpy(), permutations,
                permutation_rate=p_rate, iterations=p_iters, seed=seed
            )
            for p in distributions:
                pv = statistics.p_value(npa, distributions[p])
                result_builder.set_global_attributes(dataset_id, ["%s_value" % p], [pv])
                result_builder.set_distribution(dataset_id, p, distributions[p], npa)

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
