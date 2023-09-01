import sys
import warnings
import logging
from pathlib import Path
from typing import Optional
import json

import numpy as np
import networkx as nx

import perturbationx.io as px_io
from perturbationx.io import RelationTranslator
import perturbationx.toponpa as toponpa
from perturbationx.resources import DEFAULT_DATA_COLS, DEFAULT_LOGGING_KWARGS


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

        px_io.validate_nx_graph(graph, self.__allowed_edge_types)
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
        graph.add_edges_from(px_io.read_dsv(
            filepath, default_edge_type=edge_type, delimiter=delimiter, header_cols=header_cols
        ))
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

    @classmethod
    def from_pandas(cls, df, default_edge_type="infer",
                    header_cols=DEFAULT_DATA_COLS, relation_translator=None):
        edge_list = px_io.parse_pandas(
            df, default_edge_type=default_edge_type, header_cols=header_cols
        )
        graph = nx.DiGraph()
        graph.add_edges_from(edge_list)
        return cls(graph, relation_translator, inplace=True)

    def add_edges_from_dsv(self, filepath, edge_type="infer", delimiter='\t',
                           header_cols=DEFAULT_DATA_COLS):
        self._graph.add_edges_from(px_io.read_dsv(
            filepath, default_edge_type=edge_type, delimiter=delimiter, header_cols=header_cols
        ))
        px_io.validate_nx_graph(self._graph, self.__allowed_edge_types)

    def add_edges_from_tsv(self, filepath, edge_type="infer",
                           header_cols=DEFAULT_DATA_COLS):
        self.add_edges_from_dsv(filepath, edge_type, header_cols=header_cols)

    def add_edges_from_csv(self, filepath, edge_type="infer",
                           header_cols=DEFAULT_DATA_COLS):
        self.add_edges_from_dsv(filepath, edge_type, delimiter=',', header_cols=header_cols)

    def add_edges_from_pandas(self, df, default_edge_type="infer",
                              header_cols=DEFAULT_DATA_COLS):
        self._graph.add_edges_from(px_io.parse_pandas(
            df, default_edge_type=default_edge_type, header_cols=header_cols
        ))
        px_io.validate_nx_graph(self._graph, self.__allowed_edge_types)

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

    def number_of_nodes(self, typ=None):
        if typ is not None:
            return sum(1 for n, d in self._graph.nodes(data=True) if d["type"] == typ)
        return self._graph.number_of_nodes()

    def nodes(self, typ=None, data=True):
        if typ is not None:
            if data:
                return [(n, d) for n, d in self._graph.nodes(data=True) if d["type"] == typ]
            else:
                return [n for n, d in self._graph.nodes(data=True) if d["type"] == typ]

        return list(self._graph.nodes(data=data))

    def number_of_edges(self, typ=None):
        if typ is not None:
            return sum(1 for e in self._graph.edges.data() if e[2]["type"] == typ)
        return self._graph.number_of_edges()

    def edges(self, typ=None, data=True):
        if typ is not None:
            if data:
                return [(src, trg, d) for src, trg, d in self._graph.edges.data() if d["type"] == typ]
            else:
                return [(src, trg) for src, trg, d in self._graph.edges.data() if d["type"] == typ]

        if data:
            return list(self._graph.edges.data())
        else:
            return list(self._graph.edges)

    def add_edge(self, src, trg, rel, typ="infer"):
        if self._graph.has_edge(src, trg):
            warnings.warn("Edge between %s and %s already exists "
                          "and will be modified." % (src, trg))
        if typ in self.__allowed_edge_types:
            self._graph.add_edge(src, trg, relation=rel, type=typ)
        else:
            warnings.warn("Unknown type %s of edge %s will be "
                          "replaced with \"infer\"." % (typ, str((src, trg))))
            self._graph.add_edge(src, trg, relation=rel, type="infer")

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

    def modify_network(self, edge_list):
        for src, trg, rel, typ in edge_list:
            if rel is None:
                self.remove_edge(src, trg)
            elif typ in self.__allowed_edge_types:
                self._graph.add_edge(src, trg, relation=rel, type=typ)
            elif self._graph.has_edge(src, trg):
                warnings.warn("Unknown type %s of edge %s "
                              "will be ignored." % (typ, str((src, trg))))
                self._graph[src][trg]["relation"] = rel
            else:
                warnings.warn("Unknown type %s of edge %s will be "
                              "replaced with \"infer\"." % (typ, str((src, trg))))
                self._graph.add_edge(src, trg, relation=rel, type="infer")

    def rewire_edges(self, nodes, iterations, datasets, method='k1', p_rate=1.,
                     missing_value_pruning_mode="remove", opposing_value_pruning_mode="remove",
                     boundary_edge_minimum=6, exact_boundary_outdegree=True, seed=None, verbose=True,
                     logging_kwargs=DEFAULT_LOGGING_KWARGS):
        if verbose and logging_kwargs is not None:
            logging.basicConfig(**logging_kwargs)
            logging.info("REWIRING EDGES")

        # Find existing edges
        existing_edges = []
        for src, trg in self._graph.edges(nodes):
            if trg in nodes:
                existing_edges.append((src, trg, self._graph[src][trg]["relation"]))

        # Permute edges
        modifications = toponpa.permute_edge_list(
            np.array(existing_edges), nodes, iterations,
            method=method, permutation_rate=p_rate, seed=seed
        )
        for idx in range(len(modifications)):
            modifications[idx] = [(*edge, "core") for edge in modifications[idx]]

        # If no datasets were given, return the permutations
        if datasets is None:
            return modifications

        # Otherwise, compute NPAs for each dataset
        return self.evaluate_modifications(
            modifications, nodes, datasets,
            missing_value_pruning_mode, opposing_value_pruning_mode,
            boundary_edge_minimum, exact_boundary_outdegree, seed, verbose
        )

    def wire_edges(self, number_of_edges, nodes, edge_relations, iterations, datasets,
                   missing_value_pruning_mode="remove", opposing_value_pruning_mode="remove",
                   boundary_edge_minimum=6, exact_boundary_outdegree=True, seed=None, verbose=True,
                   logging_kwargs=DEFAULT_LOGGING_KWARGS):
        if verbose and logging_kwargs is not None:
            logging.basicConfig(**logging_kwargs)
            logging.info("WIRING EDGES")

        rng = np.random.default_rng(seed)
        modifications = []
        for _ in range(iterations):
            src_nodes = rng.choice(nodes, size=number_of_edges, replace=True)
            trg_nodes = rng.choice(nodes, size=number_of_edges, replace=True)
            for idx, (src, trg) in enumerate(zip(src_nodes, trg_nodes)):
                while trg == src:
                    trg = rng.choice(nodes)
                trg_nodes[idx] = trg

            relations = rng.choice(edge_relations, size=number_of_edges, replace=True)
            modifications.append([(src, trg, rel, "core") for src, trg, rel in zip(src_nodes, trg_nodes, relations)])

        # If no datasets were given, return the modifications
        if datasets is None:
            return modifications

        # Otherwise, compute NPAs for each dataset
        return self.evaluate_modifications(
            modifications, nodes, datasets,
            missing_value_pruning_mode, opposing_value_pruning_mode,
            boundary_edge_minimum, exact_boundary_outdegree, seed, verbose
        )

    def evaluate_modifications(self, modifications, nodes, datasets, missing_value_pruning_mode="remove",
                               opposing_value_pruning_mode="remove", boundary_edge_minimum=6,
                               exact_boundary_outdegree=True, seed=None, verbose=True,
                               logging_kwargs=DEFAULT_LOGGING_KWARGS):
        if verbose and logging_kwargs is not None:
            logging.basicConfig(**logging_kwargs)

        # Copy the graph
        prograph = self._graph.to_directed()

        return toponpa.evaluate_modifications(
            prograph, self.relation_translator, modifications, nodes, datasets,
            missing_value_pruning_mode, opposing_value_pruning_mode,
            boundary_edge_minimum, exact_boundary_outdegree, seed, verbose
        )

    def infer_graph_attributes(self, inplace=False, verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS):
        if verbose and logging_kwargs is not None:
            logging.basicConfig(**logging_kwargs)

        if inplace:
            toponpa.infer_graph_attributes(self._graph, self.relation_translator, verbose)
            return self
        else:
            graph = self._graph.to_directed()
            toponpa.infer_graph_attributes(graph, self.relation_translator, verbose)
            return CausalNetwork(graph, self.relation_translator, inplace=True)

    def get_adjacencies(self, verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS):
        if verbose and logging_kwargs is not None:
            logging.basicConfig(**logging_kwargs)
            logging.info("PREPROCESSING NETWORK")

        # Preprocess the graph
        prograph = self._graph.to_directed()
        toponpa.infer_graph_attributes(prograph, self.relation_translator, verbose)

        # Construct adjacency matrices
        adj_b, adj_c = toponpa.generate_adjacencies(prograph)

        # Determine node ordering
        node_ordering = [None] * len(prograph.nodes)
        for node in prograph.nodes:
            node_ordering[prograph.nodes[node]["idx"]] = node

        return adj_b, adj_c, node_ordering

    def get_laplacians(self, boundary_outdegree_minimum=6, exact_boundary_outdegree=True, verbose=True):
        adj_b, adj_c, node_ordering = self.get_adjacencies(verbose)
        lap_b = - toponpa.generate_boundary_laplacian(
            adj_b, boundary_edge_minimum=boundary_outdegree_minimum
        )
        lap_c, lap_q, _ = toponpa.generate_core_laplacians(
            lap_b, adj_c, {},
            exact_boundary_outdegree=exact_boundary_outdegree
        )
        return lap_b, lap_c, lap_q, node_ordering

    def toponpa(self, datasets: dict, missing_value_pruning_mode="remove", opposing_value_pruning_mode="remove",
                boundary_edge_minimum=6, exact_boundary_outdegree=True, compute_statistics=True, alpha=0.95,
                permutations=('o', 'k2'), p_iters=500, p_rate=1., seed=None, verbose=True,
                logging_kwargs=DEFAULT_LOGGING_KWARGS):
        if verbose and logging_kwargs is not None:
            logging.basicConfig(**logging_kwargs)

        # Copy the graph and set metadata
        prograph = self._graph.to_directed()
        self.initialize_metadata()
        prograph.graph.update(self.metadata)

        return toponpa.toponpa(
            prograph, self.relation_translator, datasets,
            missing_value_pruning_mode, opposing_value_pruning_mode,
            boundary_edge_minimum, exact_boundary_outdegree,
            compute_statistics, alpha, permutations, p_iters, p_rate,
            seed, verbose
        )

    def to_networkx(self):
        return self._graph.copy()

    def to_edge_list(self, edge_type="all", data_cols=DEFAULT_DATA_COLS,):
        return px_io.to_edge_list(self._graph, edge_type, data_cols)

    def to_dsv(self, filepath, edge_type="all", delimiter='\t',
               data_cols=DEFAULT_DATA_COLS, header=None):
        px_io.write_dsv(self._graph, filepath, edge_type, delimiter, data_cols, header)

    def to_tsv(self, filepath, edge_type="all",
               data_cols=DEFAULT_DATA_COLS, header=None):
        self.to_dsv(filepath, edge_type, data_cols=data_cols, header=header)

    def to_csv(self, filepath, edge_type="all",
               data_cols=DEFAULT_DATA_COLS, header=None):
        self.to_dsv(filepath, edge_type, delimiter=",", data_cols=data_cols, header=header)

    def to_cyjs_json(self, filepath, indent=4):
        with open(filepath, 'w') as f:
            json.dump(nx.cytoscape_data(self._graph), f, ensure_ascii=False, indent=indent)
