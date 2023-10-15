import warnings
import logging
from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
import networkx as nx

import perturbationx.io as px_io
from perturbationx.io import RelationTranslator
import perturbationx.toponpa as toponpa
from perturbationx.resources import DEFAULT_DATA_COLS, DEFAULT_LOGGING_KWARGS


class CausalNetwork:
    """Class for storing and evaluating causal networks. The network is stored as a networkx.DiGraph. The class
    provides methods for reading and writing networks from and to various file formats, as well as for evaluating
    the network using the TopoNPA algorithm. The class also provides methods for rewiring and wiring edges in the
    network, as well as for evaluating the rewired and wired networks.

    :param graph: The network graph. Defaults to None (empty graph).
    :type graph: networkx.DiGraph, optional
    :param relation_translator: The relation translator. Defaults to None (default relation translator).
    :type relation_translator: RelationTranslator, optional
    :param inplace: Whether to initialize the network in place. If False, the network is copied. Defaults to False.
    :type inplace: bool, optional
    :raises TypeError: If the relation translator is not a RelationTranslator.
    """
    __allowed_edge_types = ("core", "boundary", "infer")

    def __init__(self, graph: Optional[nx.DiGraph] = None,
                 relation_translator: Optional[RelationTranslator] = None,
                 inplace=False):
        """Construct a new CausalNetwork.
        """
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
        """Construct a new CausalNetwork from a networkx.DiGraph.

        :param graph: The network graph.
        :type graph: networkx.DiGraph
        :param relation_translator: The relation translator. Defaults to None (default relation translator).
        :type relation_translator: RelationTranslator, optional
        :param inplace: Whether to initialize the network in place. If False, the network is copied. Defaults to False.
        :type inplace: bool, optional
        :raises TypeError: If the relation translator is not a RelationTranslator.
        """
        return cls(graph, relation_translator, inplace)

    @classmethod
    def from_dsv(cls, filepath: str, edge_type="infer", delimiter='\t',
                 header_cols=DEFAULT_DATA_COLS, relation_translator=None):
        """Construct a new CausalNetwork from a delimited file.

        :param filepath: The path to the file.
        :type filepath: str
        :param edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
        :type edge_type: str, optional
        :param delimiter: The delimiter to use. Defaults to '\t'.
        :type delimiter: str, optional
        :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
            "type"].
        :type header_cols: list, optional
        :param relation_translator: The relation translator. Defaults to None (default relation translator).
        :type relation_translator: RelationTranslator, optional
        :raises TypeError: If the relation translator is not a RelationTranslator.
        :return: The constructed CausalNetwork.
        :rtype: CausalNetwork
        """
        graph = nx.DiGraph()
        graph.add_edges_from(px_io.read_dsv(
            filepath, default_edge_type=edge_type, delimiter=delimiter, header_cols=header_cols
        ))
        graph.graph["title"] = Path(filepath).stem
        graph.graph["collection"] = Path(filepath).parent.name
        return cls(graph, relation_translator, inplace=True)

    @classmethod
    def from_tsv(cls, filepath: str, edge_type="infer",
                 header_cols=DEFAULT_DATA_COLS, relation_translator=None):
        """Construct a new CausalNetwork from a tab-delimited file.

        :param filepath: The path to the file.
        :type filepath: str
        :param edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
        :type edge_type: str, optional
        :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
            "type"].
        :type header_cols: list, optional
        :param relation_translator: The relation translator. Defaults to None (default relation translator).
        :type relation_translator: RelationTranslator, optional
        :raises TypeError: If the relation translator is not a RelationTranslator.
        :return: The constructed CausalNetwork.
        :rtype: CausalNetwork
        """
        return cls.from_dsv(filepath, edge_type, header_cols=header_cols,
                            relation_translator=relation_translator)

    @classmethod
    def from_csv(cls, filepath: str, edge_type="infer",
                 header_cols=DEFAULT_DATA_COLS, relation_translator=None):
        """Construct a new CausalNetwork from a comma-delimited file.

        :param filepath: The path to the file.
        :type filepath: str
        :param edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
        :type edge_type: str, optional
        :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
            "type"].
        :type header_cols: list, optional
        :param relation_translator: The relation translator. Defaults to None (default relation translator).
        :type relation_translator: RelationTranslator, optional
        :raises TypeError: If the relation translator is not a RelationTranslator.
        :return: The constructed CausalNetwork.
        :rtype: CausalNetwork
        """
        return cls.from_dsv(filepath, edge_type, delimiter=',', header_cols=header_cols,
                            relation_translator=relation_translator)

    @classmethod
    def from_cyjs_json(cls, filepath: str, relation_translator=None):
        """Construct a new CausalNetwork from a Cytoscape.js JSON file.

        :param filepath: The path to the file.
        :type filepath: str
        :param relation_translator: The relation translator. Defaults to None (default relation translator).
        :type relation_translator: RelationTranslator, optional
        :raises TypeError: If the relation translator is not a RelationTranslator.
        :return: The constructed CausalNetwork.
        :rtype: CausalNetwork
        """
        with open(filepath) as f:
            graph = nx.cytoscape_graph(json.load(f))
            graph.graph["title"] = Path(filepath).stem
            graph.graph["collection"] = Path(filepath).parent.name
            return cls(graph, relation_translator, inplace=True)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, default_edge_type="infer",
                    header_cols=DEFAULT_DATA_COLS, relation_translator=None):
        """Construct a new CausalNetwork from a pandas DataFrame.

        :param df: The DataFrame to parse.
        :type df: pd.DataFrame
        :param default_edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
        :type default_edge_type: str, optional
        :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
            "type"].
        :type header_cols: list, optional
        :param relation_translator: The relation translator. Defaults to None (default relation translator).
        :type relation_translator: RelationTranslator, optional
        :raises TypeError: If the relation translator is not a RelationTranslator.
        :return: The constructed CausalNetwork.
        :rtype: CausalNetwork
        """
        edge_list = px_io.parse_pandas(
            df, default_edge_type=default_edge_type, header_cols=header_cols
        )
        graph = nx.DiGraph()
        graph.add_edges_from(edge_list)
        return cls(graph, relation_translator, inplace=True)

    def add_edges_from_dsv(self, filepath: str, edge_type="infer", delimiter='\t',
                           header_cols=DEFAULT_DATA_COLS):
        """Add edges from a delimited file.

        :param filepath: The path to the file.
        :type filepath: str
        :param edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
        :type edge_type: str, optional
        :param delimiter: The delimiter to use. Defaults to '\t'.
        :type delimiter: str, optional
        :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
            "type"].
        :type header_cols: list, optional
        """
        self._graph.add_edges_from(px_io.read_dsv(
            filepath, default_edge_type=edge_type, delimiter=delimiter, header_cols=header_cols
        ))
        px_io.validate_nx_graph(self._graph, self.__allowed_edge_types)

    def add_edges_from_tsv(self, filepath: str, edge_type="infer",
                           header_cols=DEFAULT_DATA_COLS):
        """Add edges from a tab-delimited file.

        :param filepath: The path to the file.
        :type filepath: str
        :param edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
        :type edge_type: str, optional
        :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
            "type"].
        :type header_cols: list, optional
        """
        self.add_edges_from_dsv(filepath, edge_type, header_cols=header_cols)

    def add_edges_from_csv(self, filepath: str, edge_type="infer",
                           header_cols=DEFAULT_DATA_COLS):
        """Add edges from a comma-delimited file.

        :param filepath: The path to the file.
        :type filepath: str
        :param edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
        :type edge_type: str, optional
        :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
            "type"].
        :type header_cols: list, optional
        """
        self.add_edges_from_dsv(filepath, edge_type, delimiter=',', header_cols=header_cols)

    def add_edges_from_pandas(self, df: pd.DataFrame, default_edge_type="infer",
                              header_cols=DEFAULT_DATA_COLS):
        """Add edges from a pandas DataFrame.

        :param df: The DataFrame to parse.
        :type df: pd.DataFrame
        :param default_edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
        :type default_edge_type: str, optional
        :param header_cols: The column names of the columns in the file. Defaults to ["subject", "object", "relation",
            "type"].
        :type header_cols: list, optional
        """
        self._graph.add_edges_from(px_io.parse_pandas(
            df, default_edge_type=default_edge_type, header_cols=header_cols
        ))
        px_io.validate_nx_graph(self._graph, self.__allowed_edge_types)

    def initialize_metadata(self):
        """Initialize the metadata dictionary with default values. These values are "Untitled network" for the title
        and "Untitled collection" for the collection. The metadata dictionary is created if it does not exist.
        """
        # The user can delete metadata dict
        if self.metadata is None:
            self.metadata = dict()
        # Title and collection are required for Cytoscape
        if "title" not in self.metadata:
            self.metadata["title"] = "Untitled network"
        if "collection" not in self.metadata:
            self.metadata["collection"] = "Untitled collection"

    def copy(self):
        """Return a copy of the CausalNetwork.

        :return: The copy.
        :rtype: CausalNetwork
        """
        return CausalNetwork(self._graph, self.relation_translator.copy(), inplace=False)

    def number_of_nodes(self, typ=None):
        """Return the number of nodes in the network.

        :param typ: The node type to count. If None, all nodes are counted. Defaults to None.
        :type typ: str, optional
        :return: The number of nodes.
        :rtype: int
        """
        if typ is not None:
            return sum(1 for n, d in self._graph.nodes(data=True) if d["type"] == typ)
        return self._graph.number_of_nodes()

    def nodes(self, typ=None, data=True):
        """Return a list of nodes in the network.

        :param typ: The node type to return. If None, all nodes are returned. Defaults to None.
        :type typ: str, optional
        :param data: Whether to return the node data. Defaults to True.
        :type data: bool, optional
        :return: The list of nodes.
        :rtype: list
        """
        if typ is not None:
            if data:
                return [(n, d) for n, d in self._graph.nodes(data=True) if d["type"] == typ]
            else:
                return [n for n, d in self._graph.nodes(data=True) if d["type"] == typ]

        return list(self._graph.nodes(data=data))

    def number_of_edges(self, typ=None):
        """Return the number of edges in the network.

        :param typ: The edge type to count. If None, all edges are counted. Defaults to None.
        :type typ: str, optional
        :return: The number of edges.
        :rtype: int
        """
        if typ is not None:
            return sum(1 for e in self._graph.edges.data() if e[2]["type"] == typ)
        return self._graph.number_of_edges()

    def edges(self, typ=None, data=True):
        """Return a list of edges in the network.

        :param typ: The edge type to return. If None, all edges are returned. Defaults to None.
        :type typ: str, optional
        :param data: Whether to return the edge data. Defaults to True.
        :type data: bool, optional
        :return: The list of edges.
        :rtype: list
        """
        if typ is not None:
            if data:
                return [(src, trg, d) for src, trg, d in self._graph.edges.data() if d["type"] == typ]
            else:
                return [(src, trg) for src, trg, d in self._graph.edges.data() if d["type"] == typ]

        if data:
            return list(self._graph.edges.data())
        else:
            return list(self._graph.edges)

    def add_edge(self, src: str, trg: str, rel: str, typ="infer", confidence=None):
        """Add an edge to the network. If the edge already exists, it is modified. If the nodes do not exist, they are
        created.

        :param src: The source node.
        :type src: str
        :param trg: The target node.
        :type trg: str
        :param rel: The causal relation of the edge.
        :type rel: str
        :param typ: The type of the edge. Allowed values are "core", "boundary" and "infer". Defaults to "infer".
        :type typ: str, optional
        :param confidence: The confidence of the edge. Defaults to None, which sets the confidence to 1.0.
        :type confidence: float, optional
        """
        if self._graph.has_edge(src, trg):
            warnings.warn("Edge between %s and %s already exists "
                          "and will be modified." % (src, trg))
        if typ in self.__allowed_edge_types:
            self._graph.add_edge(src, trg, relation=rel, type=typ)
        else:
            warnings.warn("Unknown type %s of edge %s will be "
                          "replaced with \"infer\"." % (typ, str((src, trg))))
            self._graph.add_edge(src, trg, relation=rel, type="infer")
        if confidence is not None:
            self._graph[src][trg]["confidence"] = float(confidence)

    def modify_edge(self, src: str, trg: str, rel=None, typ=None, confidence=None):
        """Modify an edge in the network. If the edge does not exist, a KeyError is raised.

        :param src: The source node.
        :type src: str
        :param trg: The target node.
        :type trg: str
        :param rel: The new causal relation of the edge. Defaults to None, which does not modify the relation.
        :type rel: str, optional
        :param typ: The new type of the edge. Allowed values are "core", "boundary" and "infer". Defaults to None,
            which does not modify the type.
        :type typ: str, optional
        :param confidence: The new confidence of the edge. Defaults to None, which does not modify the confidence.
        :type confidence: float, optional
        :raises KeyError: If the edge does not exist.
        """
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
        if confidence is not None:
            self._graph[src][trg]["confidence"] = float(confidence)

    def remove_edge(self, src: str, trg: str):
        """Remove an edge from the network. If the edge does not exist, a KeyError is raised.

        :param src: The source node.
        :type src: str
        :param trg: The target node.
        :type trg: str
        :raises KeyError: If the edge does not exist.
        """
        if not self._graph.has_edge(src, trg):
            raise KeyError("Edge between %s and %s does not exist." % (src, trg))
        self._graph.remove_edge(src, trg)

        if self._graph.degree[src] == 0:
            self._graph.remove_node(src)
        if self._graph.degree[trg] == 0:
            self._graph.remove_node(trg)

    def modify_network(self, edge_list):
        """Modify the network using a list of edges. The list of edges is a list of tuples of the form
        (source, target, relation, type). If the relation is None, the edge is removed. If the type is None, the type
        is not modified. If the type is not one of the allowed types, it is ignored or replaced with "infer".

        :param edge_list: The list of edges.
        :type edge_list: list
        """
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

    def rewire_edges(self, nodes: list, iterations: int, datasets: list, method='k1', p_rate=1.,
                     missing_value_pruning_mode="nullify", opposing_value_pruning_mode=None,
                     opposing_value_minimum_amplitude=1., boundary_edge_minimum=6,
                     exact_boundary_outdegree=True, sparse=True, seed=None,
                     verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS):
        """Rewire edges in the network. The rewiring is performed on the given nodes, and the given number of
        iterations. Datasets can be provided to evaluate the rewired networks. If no datasets are provided, the
        rewired networks are returned as a list of modifications. Otherwise, the NPAs of the rewired networks are
        computed and returned.

        :param nodes: The nodes to rewire.
        :type nodes: list
        :param iterations: The number of rewiring iterations.
        :type iterations: int
        :param datasets: The datasets to use for evaluation. If None, the rewired networks are returned as a list of
            modifications. Otherwise, the NPAs of the rewired networks are computed and returned.
        :type datasets: list
        :param method: The rewiring method to use. May be "k1" or "k2". Defaults to "k1".
        :type method: str, optional
        :param p_rate: The fraction of edges to rewire. Defaults to 1.0.
        :type p_rate: float, optional
        :param missing_value_pruning_mode: The mode to use for pruning nodes with missing values. Must be one of
        'remove' or 'nullify'. Defaults to 'nullify'.
        :type missing_value_pruning_mode: str, optional
        :param opposing_value_pruning_mode: The mode to use for pruning edges with opposing values. Must be one of
        'remove', 'nullify', or 'none'. Defaults to None.
        :type opposing_value_pruning_mode: str, optional
        :param opposing_value_minimum_amplitude: The minimum amplitude of the dataset values to consider. Values with
        an absolute value smaller than this threshold are ignored. Defaults to 1.
        :type opposing_value_minimum_amplitude: float, optional
        :param boundary_edge_minimum: The minimum number of boundary edges a core node must have to be included
        in the pruned network. If a core node has fewer boundary edges after 'remove' pruning, all of its edges are
        removed. This parameter is ignored if 'nullify' pruning is used. Defaults to 6.
        :type boundary_edge_minimum: int, optional
        :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
        is set to 1 for all core nodes with boundary edges. Defaults to True.
        :type exact_boundary_outdegree: bool, optional
        :param sparse: Whether to use sparse computation. Defaults to True.
        :type sparse: bool, optional
        :param seed: The seed to use for the random number generator. Defaults to None.
        :type seed: int, optional
        :param verbose: Whether to log progress and network statistics. Defaults to True.
        :type verbose: bool, optional
        :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to "stream": sys.stdout,
         "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
        :type logging_kwargs: dict, optional
        :raises ValueError: If the permutation method is unknown. If the same node appears in both
        the core and boundary network.
        :return: The list of modifications, or a list of tuples of the form (modification, npa), where modification
        is the modification and npa is a dictionary of the form {dataset_id: npa}.
        :rtype: list
        """
        if verbose and logging_kwargs is not None:
            logging.basicConfig(**logging_kwargs)
            logging.info("REWIRING EDGES")

        # Find existing edges
        existing_edges = []
        for src, trg in self._graph.edges(nodes):
            if trg in nodes:
                confidence = self._graph[src][trg].get("confidence", 1.0)
                existing_edges.append((src, trg, self._graph[src][trg]["relation"], confidence))

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
            missing_value_pruning_mode=missing_value_pruning_mode,
            opposing_value_pruning_mode=opposing_value_pruning_mode,
            opposing_value_minimum_amplitude=opposing_value_minimum_amplitude,
            boundary_edge_minimum=boundary_edge_minimum,
            exact_boundary_outdegree=exact_boundary_outdegree,
            sparse=sparse, seed=seed, verbose=verbose
        )

    def wire_edges(self, nodes: list, iterations: int, datasets: list, number_of_edges: int, edge_relations: list,
                   missing_value_pruning_mode="nullify", opposing_value_pruning_mode=None,
                   opposing_value_minimum_amplitude=1., boundary_edge_minimum=6,
                   exact_boundary_outdegree=True, sparse=True, seed=None,
                   verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS):
        """Wire edges in the network. The wiring is performed on the given nodes, and the given number of
        iterations. Datasets can be provided to evaluate the wired networks. If no datasets are provided, the
        wired networks are returned as a list of modifications. Otherwise, the NPAs of the wired networks are
        computed and returned.

        :param nodes: The nodes to wire.
        :type nodes: list
        :param iterations: The number of wiring iterations.
        :type iterations: int
        :param datasets: The datasets to use for evaluation. If None, the wired networks are returned as a list of
            modifications. Otherwise, the NPAs of the wired networks are computed and returned.
        :type datasets: list
        :param number_of_edges: The number of edges to wire.
        :type number_of_edges: int
        :param edge_relations: The relations to use for wiring.
        :type edge_relations: list
        :param missing_value_pruning_mode: The mode to use for pruning nodes with missing values. Must be one of
        'remove' or 'nullify'. Defaults to 'nullify'.
        :type missing_value_pruning_mode: str, optional
        :param opposing_value_pruning_mode: The mode to use for pruning edges with opposing values. Must be one of
        'remove', 'nullify', or 'none'. Defaults to None.
        :type opposing_value_pruning_mode: str, optional
        :param opposing_value_minimum_amplitude: The minimum amplitude of the dataset values to consider. Values with
        an absolute value smaller than this threshold are ignored. Defaults to 1.
        :type opposing_value_minimum_amplitude: float, optional
        :param boundary_edge_minimum: The minimum number of boundary edges a core node must have to be included
        in the pruned network. If a core node has fewer boundary edges after 'remove' pruning, all of its edges are
        removed. This parameter is ignored if 'nullify' pruning is used. Defaults to 6.
        :type boundary_edge_minimum: int, optional
        :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
        is set to 1 for all core nodes with boundary edges. Defaults to True.
        :type exact_boundary_outdegree: bool, optional
        :param sparse: Whether to use sparse computation. Defaults to True.
        :type sparse: bool, optional
        :param seed: The seed to use for the random number generator. Defaults to None.
        :type seed: int, optional
        :param verbose: Whether to log progress and network statistics. Defaults to True.
        :type verbose: bool, optional
        :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to "stream": sys.stdout,
         "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
        :type logging_kwargs: dict, optional
        :raises ValueError: If the same node appears in both the core and boundary network.
        :return: The list of modifications, or a list of tuples of the form (modification, npa), where modification
        is the modification and npa is a dictionary of the form {dataset_id: npa}.
        :rtype: list
        """
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
            modifications.append([
                (src, trg, rel, "core") for src, trg, rel
                in zip(src_nodes, trg_nodes, relations)
            ])

        # If no datasets were given, return the modifications
        if datasets is None:
            return modifications

        # Otherwise, compute NPAs for each dataset
        return self.evaluate_modifications(
            modifications, nodes, datasets,
            missing_value_pruning_mode=missing_value_pruning_mode,
            opposing_value_pruning_mode=opposing_value_pruning_mode,
            opposing_value_minimum_amplitude=opposing_value_minimum_amplitude,
            boundary_edge_minimum=boundary_edge_minimum,
            exact_boundary_outdegree=exact_boundary_outdegree,
            sparse=sparse, seed=seed, verbose=verbose
        )

    def evaluate_modifications(self, modifications: list, nodes, datasets, missing_value_pruning_mode="nullify",
                               opposing_value_pruning_mode=None, opposing_value_minimum_amplitude=1.,
                               boundary_edge_minimum=6, exact_boundary_outdegree=True, sparse=True,
                               seed=None, verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS):
        """Evaluate modifications of the network. The modifications are evaluated using the given datasets. The
        modifications are returned as a list of tuples of the form (modification, npa), where modification is the
        modification and npa is a dictionary of the form {dataset_id: npa}. Modifications can be generated using
        the rewire_edges or wire_edges methods.

        :param modifications: The modifications to evaluate. The modifications are a list of lists of tuples of the
        form (source, target, relation, type).
        :type modifications: list
        :param nodes: The nodes used in the modifications.
        :type nodes: list
        :param datasets: The datasets to use for evaluation.
        :type datasets: list
        :param missing_value_pruning_mode: The mode to use for pruning nodes with missing values. Must be one of
        'remove' or 'nullify'. Defaults to 'nullify'.
        :type missing_value_pruning_mode: str, optional
        :param opposing_value_pruning_mode: The mode to use for pruning edges with opposing values. Must be one of
        'remove', 'nullify', or 'none'. Defaults to None.
        :type opposing_value_pruning_mode: str, optional
        :param opposing_value_minimum_amplitude: The minimum amplitude of the dataset values to consider. Values with
        an absolute value smaller than this threshold are ignored. Defaults to 1.
        :type opposing_value_minimum_amplitude: float, optional
        :param boundary_edge_minimum: The minimum number of boundary edges a core node must have to be included
        in the pruned network. If a core node has fewer boundary edges after 'remove' pruning, all of its edges are
        removed. This parameter is ignored if 'nullify' pruning is used. Defaults to 6.
        :type boundary_edge_minimum: int, optional
        :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
        is set to 1 for all core nodes with boundary edges. Defaults to True.
        :type exact_boundary_outdegree: bool, optional
        :param sparse: Whether to use sparse computation. Defaults to True.
        :type sparse: bool, optional
        :param seed: The seed to use for the random number generator. Defaults to None.
        :type seed: int, optional
        :param verbose: Whether to log progress and network statistics. Defaults to True.
        :type verbose: bool, optional
        :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to "stream": sys.stdout,
         "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
        :type logging_kwargs: dict, optional
        :raises ValueError: If the same node appears in both the core and boundary network.
        :return: The list of modifications, or a list of tuples of the form (modification, npa), where modification
        is the modification and npa is a dictionary of the form {dataset_id: npa}.
        :rtype: list
        """
        if verbose and logging_kwargs is not None:
            logging.basicConfig(**logging_kwargs)

        # Copy the graph
        prograph = self._graph.to_directed()

        return toponpa.evaluate_modifications(
            prograph, self.relation_translator, modifications, nodes, datasets,
            missing_value_pruning_mode=missing_value_pruning_mode,
            opposing_value_pruning_mode=opposing_value_pruning_mode,
            opposing_value_minimum_amplitude=opposing_value_minimum_amplitude,
            boundary_edge_minimum=boundary_edge_minimum,
            exact_boundary_outdegree=exact_boundary_outdegree,
            sparse=sparse, seed=seed, verbose=verbose
        )

    def infer_graph_attributes(self, inplace=False, verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS):
        """Infer graph attributes required for NPA computation. The attributes are inferred from the relations in the
        network. The attributes are added to the graph as node and edge attributes. If inplace is True, the attributes
        are added to the graph and the graph is returned. Otherwise, the attributes are added to a copy of the graph
        and the copy is returned.

        :param inplace: Whether to add the attributes to the graph inplace. Defaults to False.
        :type inplace: bool, optional
        :param verbose: Whether to log progress and network statistics. Defaults to True.
        :type verbose: bool, optional
        :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to  "stream": sys.stdout,
        "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
        :type logging_kwargs: dict, optional
        :raises ValueError: If the same node appears in both the core and boundary network.
        :return: The graph with the inferred attributes.
        :rtype: CausalNetwork
        """
        if verbose and logging_kwargs is not None:
            logging.basicConfig(**logging_kwargs)

        if inplace:
            toponpa.infer_graph_attributes(self._graph, self.relation_translator, verbose)
            return self
        else:
            graph = self._graph.to_directed()
            toponpa.infer_graph_attributes(graph, self.relation_translator, verbose)
            return CausalNetwork(graph, self.relation_translator, inplace=True)

    def get_adjacencies(self, sparse=False, verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS):
        """Get the adjacency matrices of the network.

        :param sparse: Whether to use sparse matrices. Defaults to False.
        :type sparse: bool, optional
        :param verbose: Whether to log progress and network statistics. Defaults to True.
        :type verbose: bool, optional
        :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to  "stream": sys.stdout,
        "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
        :type logging_kwargs: dict, optional
        :raises ValueError: If the same node appears in both the core and boundary network.
        :return: The boundary and core adjacency matrices and the node ordering.
        :rtype: (np.ndarray | sp.sparray, np.ndarray | sp.sparray, list)
        """
        if verbose and logging_kwargs is not None:
            logging.basicConfig(**logging_kwargs)
            logging.info("PREPROCESSING NETWORK")

        # Preprocess the graph
        prograph = self._graph.to_directed()
        toponpa.infer_graph_attributes(prograph, self.relation_translator, verbose)

        # Construct adjacency matrices
        adj_b, adj_c = toponpa.generate_adjacencies(prograph, sparse=sparse)

        # Determine node ordering
        node_ordering = [None] * len(prograph.nodes)
        for node in prograph.nodes:
            node_ordering[prograph.nodes[node]["idx"]] = node

        return adj_b, adj_c, node_ordering

    def get_laplacians(self, boundary_outdegree_minimum=6, exact_boundary_outdegree=True, sparse=False,
                       verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS):
        """Get the Laplacian matrices of the network.

        :param boundary_outdegree_minimum: The minimum number of boundary edges a core node must have to be included
        in the boundary network. Defaults to 6.
        :type boundary_outdegree_minimum: int, optional
        :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
        is set to 1 for all core nodes with boundary edges. Defaults to True.
        :type exact_boundary_outdegree: bool, optional
        :param sparse: Whether to use sparse matrices. Defaults to False.
        :type sparse: bool, optional
        :param verbose: Whether to log progress and network statistics. Defaults to True.
        :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to  "stream": sys.stdout,
        "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
        :raises ValueError: If the same node appears in both the core and boundary network.
        :return: The Lb boundary Laplacian, Lc core Laplacian and Q core Laplacian and the node ordering.
        :rtype: (np.ndarray | sp.sparray, np.ndarray | sp.sparray, np.ndarray | sp.sparray, list)
        """
        adj_b, adj_c, node_ordering = self.get_adjacencies(
            sparse=sparse, verbose=verbose, logging_kwargs=logging_kwargs)
        lap_b = - toponpa.generate_boundary_laplacian(
            adj_b, boundary_edge_minimum=boundary_outdegree_minimum
        )
        lap_c, lap_q, = toponpa.generate_core_laplacians(
            lap_b, adj_c, exact_boundary_outdegree=exact_boundary_outdegree
        )
        return lap_b, lap_c, lap_q, node_ordering

    def toponpa(self, datasets: dict, missing_value_pruning_mode="nullify", opposing_value_pruning_mode=None,
                opposing_value_minimum_amplitude=1., boundary_edge_minimum=6, exact_boundary_outdegree=True,
                compute_statistics=True, alpha=0.95, permutations=('o', 'k2'), full_core_permutation=True,
                p_iters=500, p_rate=1., sparse=True, seed=None, verbose=True, logging_kwargs=DEFAULT_LOGGING_KWARGS):
        """Compute the Network Perturbation Amplitude (NPA) for a given network and datasets.

        :param datasets: The datasets to use. The keys are the dataset IDs and the values are the datasets, which are
        pandas DataFrames.
        :type datasets: dict
        :param missing_value_pruning_mode: The mode to use for pruning nodes with missing values. Must be one of
        'remove' or 'nullify'. Defaults to 'nullify'.
        :type missing_value_pruning_mode: str, optional
        :param opposing_value_pruning_mode: The mode to use for pruning edges with opposing values. Must be one of
        'remove', 'nullify', or 'none'. Defaults to None.
        :type opposing_value_pruning_mode: str, optional
        :param opposing_value_minimum_amplitude: The minimum amplitude of the dataset values to consider. Values with an
        absolute value smaller than this threshold are ignored. Defaults to 1.
        :type opposing_value_minimum_amplitude: float, optional
        :param boundary_edge_minimum: The minimum number of boundary edges a core node must have to be included in
        the pruned network. If a core node has fewer boundary edges after 'remove' pruning, all of its edges are
        removed. This parameter is ignored if 'nullify' pruning is used. Defaults to 6.
        :type boundary_edge_minimum: int, optional
        :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
        is set to 1 for all core nodes with boundary edges. Defaults to True.
        :type exact_boundary_outdegree: bool, optional
        :param compute_statistics: Whether to compute variances and confidence intervals. Defaults to True.
        :type compute_statistics: bool, optional
        :param alpha: The confidence level for the confidence intervals. Defaults to 0.95.
        :type alpha: float, optional
        :param permutations: The permutations to test. May contain 'o', 'k1', and 'k2' in any order.
        Defaults to ('o', 'k2').
        :type permutations: list, optional
        :param full_core_permutation: Whether to use the full permutation matrix for each core permutation. Partial
        permutations sample core coefficients, while full permutations sample perturbation scores. Defaults to True.
        :type full_core_permutation: bool, optional
        :param p_iters: The number of permutations to perform. Defaults to 500.
        :type p_iters: int, optional
        :param p_rate: The fraction of boundary coefficients to permute. Defaults to 1.
        :type p_rate: float, optional
        :param sparse: Whether to use sparse computation. Defaults to True.
        :type sparse: bool, optional
        :param seed: The seed for the random number generator. Defaults to None.
        :type seed: int, optional
        :param verbose: Whether to log progress and network statistics. Defaults to True.
        :type verbose: bool, optional
        :param logging_kwargs: The keyword arguments to pass to logging.basicConfig. Defaults to  "stream": sys.stdout,
        "level": logging.INFO, "format": "%(asctime)s %(levelname)s -- %(message)s".
        :type logging_kwargs: dict, optional
        :raises ValueError: If the same node appears in both the core and boundary network.
        :return: The NPA result.
        :rtype: NPAResult
        """
        if verbose and logging_kwargs is not None:
            logging.basicConfig(**logging_kwargs)

        # Copy the graph and set metadata
        prograph = self._graph.to_directed()
        self.initialize_metadata()
        prograph.graph.update(self.metadata)

        return toponpa.toponpa(
            prograph, self.relation_translator, datasets,
            missing_value_pruning_mode=missing_value_pruning_mode,
            opposing_value_pruning_mode=opposing_value_pruning_mode,
            opposing_value_minimum_amplitude=opposing_value_minimum_amplitude,
            boundary_edge_minimum=boundary_edge_minimum,
            exact_boundary_outdegree=exact_boundary_outdegree,
            compute_statistics=compute_statistics, alpha=alpha,
            permutations=permutations, full_core_permutation=full_core_permutation,
            p_iters=p_iters, p_rate=p_rate, sparse=sparse, seed=seed, verbose=verbose
        )

    def to_networkx(self):
        """Return a copy of the network as a NetworkX graph.

        :return: The NetworkX graph.
        :rtype: nx.DiGraph
        """
        return self._graph.copy()

    def to_edge_list(self, edge_type="all", data_cols=DEFAULT_DATA_COLS,):
        """Convert the network to a list of edges.

        :param edge_type: List of edge types to include. If "all", all edges will be included. Defaults to "all".
        :type edge_type: list | str, optional
        :param data_cols: data_cols: List of data columns to include. Defaults to ["subject", "object",
        "relation", "type"].
        :type data_cols: list, optional
        :return: A list of edges.
        :rtype: list
        """
        return px_io.to_edge_list(self._graph, edge_type, data_cols)

    def to_dsv(self, filepath, edge_type="all", delimiter='\t',
               data_cols=DEFAULT_DATA_COLS, header=None):
        """Write the network to a delimited file.

        :param filepath: The path to write the file to.
        :type filepath: str
        :param edge_type: List of edge types to include. If "all", all edges will be included. Defaults to "all".
        :type edge_type: list | str, optional
        :param delimiter: The delimiter to use in the DSV file. Defaults to "\t".
        :type delimiter: str, optional
        :param data_cols: List of data columns to include. Columns not in ["subject", "object", "relation", "type"] will
            be ignored. Defaults to ["subject", "object", "relation", "type"].
        :type data_cols: list, optional
        :param header: List of header values to use. Must be given in the same order as data_cols. Defaults to None.
        :type header: list, optional
        :raises ValueError: If the length of the header list does not match the length of the data_cols list.
        """
        px_io.write_dsv(self._graph, filepath, edge_type, delimiter, data_cols, header)

    def to_tsv(self, filepath, edge_type="all",
               data_cols=DEFAULT_DATA_COLS, header=None):
        """Write the network to a tab-separated file.

        :param filepath: The path to write the file to.
        :type filepath: str
        :param edge_type: List of edge types to include. If "all", all edges will be included. Defaults to "all".
        :type edge_type: list | str, optional
        :param data_cols: List of data columns to include. Columns not in ["subject", "object", "relation", "type"] will
            be ignored. Defaults to ["subject", "object", "relation", "type"].
        :type data_cols: list, optional
        :param header: List of header values to use. Must be given in the same order as data_cols. Defaults to None.
        :type header: list, optional
        :raises ValueError: If the length of the header list does not match the length of the data_cols list.
        """
        self.to_dsv(filepath, edge_type, data_cols=data_cols, header=header)

    def to_csv(self, filepath, edge_type="all",
               data_cols=DEFAULT_DATA_COLS, header=None):
        """Write the network to a comma-separated file.

        :param filepath: The path to write the file to.
        :type filepath: str
        :param edge_type: List of edge types to include. If "all", all edges will be included. Defaults to "all".
        :type edge_type: list | str, optional
        :param data_cols: List of data columns to include. Columns not in ["subject", "object", "relation", "type"] will
            be ignored. Defaults to ["subject", "object", "relation", "type"].
        :type data_cols: list, optional
        :param header: List of header values to use. Must be given in the same order as data_cols. Defaults to None.
        :type header: list, optional
        :raises ValueError: If the length of the header list does not match the length of the data_cols list.
        """
        self.to_dsv(filepath, edge_type, delimiter=",", data_cols=data_cols, header=header)

    def to_cyjs_json(self, filepath, indent=4):
        """Write the network to a Cytoscape.js JSON file.

        :param filepath: The path to write the file to.
        :type filepath: str
        :param indent: The indentation to use in the JSON file. Defaults to 4.
        :type indent: int, optional
        """
        with open(filepath, 'w') as f:
            json.dump(nx.cytoscape_data(self._graph), f, ensure_ascii=False, indent=indent)
