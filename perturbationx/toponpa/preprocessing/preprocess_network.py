import logging
import warnings
import itertools
from typing import Optional

import networkx as nx

from perturbationx.io import RelationTranslator


def infer_node_type(graph: nx.DiGraph):
    """Infer the type of each node in the network (core or boundary).

    :param graph: The network to process.
    :type graph: nx.DiGraph
    :raises ValueError: If the same node appears in both the core and boundary sets.
    :return: A tuple with the sets of boundary and core nodes.
    :rtype: (set, set)
    """
    # Select nodes with outgoing edges and targets of core edges as core nodes
    core_nodes = {src for src, trg in graph.edges} | \
                 {trg for src, trg in graph.edges if graph[src][trg]["type"] == "core"}

    # Select targets of boundary edges as boundary nodes
    boundary_nodes = {trg for src, trg in graph.edges if graph[src][trg]["type"] == "boundary"}

    # Check that there isn't overlap between the two sets
    node_intersection = core_nodes & boundary_nodes
    if len(node_intersection) > 0:
        raise ValueError("The same nodes appear in network core and boundary: %s." % str(node_intersection))

    # Infer targets of unknown edges; their type defaults to boundary if they have no outgoing links
    for src, trg in graph.edges:
        if graph[src][trg]["type"] not in ("core", "boundary"):
            if trg in core_nodes:
                graph[src][trg]["type"] = "core"
            else:
                graph[src][trg]["type"] = "boundary"
                boundary_nodes.add(trg)

    return boundary_nodes, core_nodes


def enumerate_nodes(graph: nx.DiGraph, boundary_nodes: set, core_nodes: set):
    """Assign an index and type to each node in the network.

    :param graph: The network to process.
    :type graph: nx.DiGraph
    :param boundary_nodes: The set of boundary nodes.
    :type boundary_nodes: set
    :param core_nodes: The set of core nodes.
    :type core_nodes: set
    """
    core_size = len(core_nodes)
    node_idx = {node: idx for idx, node in enumerate(core_nodes)} | \
               {node: (core_size + idx) for idx, node in enumerate(boundary_nodes)}
    for n in graph.nodes:
        graph.nodes[n]["idx"] = node_idx[n]
        graph.nodes[n]["type"] = "core" if n in core_nodes else "boundary"


def remove_invalid_graph_elements(graph: nx.DiGraph):
    """Remove invalid elements from the graph. This function removes self-loops and opposing edges, and ensures that the
    core graph is weakly connected.

    :param graph: The network to process.
    :type graph: nx.DiGraph
    """
    core_graph = graph.subgraph([n for n in graph.nodes if graph.nodes[n]["type"] == "core"])

    # Check that the core graph is weakly connected
    if not nx.is_weakly_connected(core_graph):
        warnings.warn("The network core is not weakly connected. Automatically selecting the largest component.")
        weak_components = sorted(nx.weakly_connected_components(core_graph), key=len, reverse=True)
        core_nodes_to_remove = set(itertools.chain.from_iterable(weak_components[1:]))
        removed_edge_count = sum(1 for src, trg in graph.edges
                                 if src in core_nodes_to_remove
                                 or trg in core_nodes_to_remove)
        graph.remove_nodes_from(core_nodes_to_remove)

        # Some boundary nodes may have become isolated
        boundary_nodes_to_remove = set(nx.isolates(graph))
        graph.remove_nodes_from(boundary_nodes_to_remove)
        warnings.warn("Removed %d nodes and %d associated edges: %s"
                      % (len(core_nodes_to_remove) + len(boundary_nodes_to_remove),
                         removed_edge_count, str(core_nodes_to_remove.union(boundary_nodes_to_remove))))

    # Check that the graph has no self-loops
    self_loops = list(nx.selfloop_edges(graph))
    if len(self_loops) > 0:
        warnings.warn("The network contains self-loops. "
                      "They cannot be processed by the algorithm and will be removed. "
                      "Found %d self-loops on nodes: %s" %
                      (len(self_loops), str(set(loop[0] for loop in self_loops))))
        graph.remove_edges_from(self_loops)

    # Check that the graph has no opposing edges
    opposing_edges = set((src, trg) for src, trg in graph.edges
                         if (trg, src) in graph.edges and src >= trg)
    if len(opposing_edges) > 0:
        warnings.warn("The network contains opposing edges. "
                      "They will be collapsed into a single edge with their weights added. "
                      "Found %d pairs of opposing edges: %s" % (len(opposing_edges), str(opposing_edges)))


def infer_edge_attributes(graph: nx.DiGraph, relation_translator: Optional[RelationTranslator] = None):
    """Infer the attributes of each edge in the network.

    :param graph: The network to process.
    :type graph: nx.DiGraph
    :param relation_translator: The relation translator to use. If None, a new instance will be created.
    :type relation_translator: RelationTranslator, optional
    """
    rt = relation_translator if relation_translator is not None else RelationTranslator()
    for src, trg in graph.edges:
        edge_weight = rt.translate(graph[src][trg]["relation"])
        graph[src][trg]["weight"] = edge_weight
        if edge_weight > 0:
            graph[src][trg]["interaction"] = "directlyIncreases"
        elif edge_weight < 0:
            graph[src][trg]["interaction"] = "directlyDecreases"
        else:
            graph[src][trg]["interaction"] = "causesNoChange"


def infer_metadata(graph: nx.DiGraph, verbose=True):
    """Infer metadata about the network and add it to the graph instance.

    :param graph: The network to process.
    :type graph: nx.DiGraph
    :param verbose: Whether to log network statistics.
    :type verbose: bool, optional
    """
    boundary_nodes = {trg for src, trg in graph.edges if graph[src][trg]["type"] == "boundary"}
    core_nodes = {src for src, trg in graph.edges} | \
                 {trg for src, trg in graph.edges if graph[src][trg]["type"] == "core"}
    inner_boundary_nodes = {src for src, trg in graph.edges if graph[src][trg]["type"] == "boundary"}

    core_edge_count = sum(1 for e in graph.edges.data() if e[2]["type"] == "core")
    boundary_edge_count = sum(1 for e in graph.edges.data() if e[2]["type"] == "boundary")

    graph.graph["core_edges"] = core_edge_count
    graph.graph["boundary_edges"] = boundary_edge_count
    graph.graph["core_nodes"] = len(core_nodes)
    graph.graph["outer_boundary_nodes"] = len(boundary_nodes)
    graph.graph["inner_boundary_nodes"] = len(inner_boundary_nodes)

    if verbose:  # Log network statistics
        logging.info("core edges: %d" % core_edge_count)
        logging.info("boundary edges: %d" % boundary_edge_count)
        logging.info("core nodes: %d" % len(core_nodes))
        logging.info("outer boundary nodes: %d" % len(boundary_nodes))
        logging.info("core nodes with boundary edges (inner boundary): %d" % len(inner_boundary_nodes))

