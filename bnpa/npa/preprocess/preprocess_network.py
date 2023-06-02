import logging
import warnings
import itertools

import networkx as nx

from bnpa.io.RelationTranslator import RelationTranslator
from typing import Optional


def infer_node_type(graph: nx.DiGraph):
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


def enumerate_nodes(graph: nx.DiGraph, boundary_nodes, core_nodes):
    core_size = len(core_nodes)
    node_idx = {node: idx for idx, node in enumerate(core_nodes)} | \
               {node: (core_size + idx) for idx, node in enumerate(boundary_nodes)}
    for n in graph.nodes:
        graph.nodes[n]["idx"] = node_idx[n]
        graph.nodes[n]["type"] = "core" if n in core_nodes else "boundary"


def remove_invalid_graph_elements(graph: nx.DiGraph):
    core_graph = graph.subgraph([n for n in graph.nodes if graph.nodes[n]["type"] == "core"])

    # Check that the core graph is weakly connected
    if not nx.is_weakly_connected(core_graph):
        warnings.warn("The network core is not weakly connected. Automatically selecting the largest component.")
        weak_components = sorted(nx.weakly_connected_components(core_graph), key=len, reverse=True)
        core_nodes_to_remove = set(itertools.chain.from_iterable(weak_components[1:]))
        graph.remove_nodes_from(core_nodes_to_remove)

        # Some boundary nodes may have become isolated
        boundary_nodes_to_remove = set(nx.isolates(graph))
        graph.remove_nodes_from(boundary_nodes_to_remove)
        warnings.warn("Removed nodes and all associated edges: %s"
                      % str(core_nodes_to_remove.union(boundary_nodes_to_remove)))

    # Check that the graph has no self-loops
    self_loops = list(nx.selfloop_edges(graph))
    if len(self_loops) > 0:
        warnings.warn("The network contains self-loops. "
                      "They cannot be processed by the algorithm and will be removed.")
        graph.remove_edges_from(self_loops)
        warnings.warn("Removed self-loops on nodes: %s" % str(set(loop[0] for loop in self_loops)))

    # Check that the graph has no opposing edges
    opposing_edges = set({(src, trg), (trg, src)} for src, trg in graph.edges if (trg, src) in graph.edges)
    if len(opposing_edges) > 0:
        warnings.warn("The network contains opposing edges. "
                      "They will be collapsed into a single edge with their weights added.")
        warnings.warn("Opposing edges found: %s" % str(opposing_edges))


def infer_edge_attributes(graph: nx.DiGraph, relation_translator: Optional[RelationTranslator] = None):
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
    boundary_nodes = {trg for src, trg in graph.edges if graph[src][trg]["type"] == "boundary"}
    core_nodes = {src for src, trg in graph.edges} | \
                 {trg for src, trg in graph.edges if graph[src][trg]["type"] == "core"}
    inner_boundary_nodes = {src for src, trg in graph.edges if graph[src][trg]["type"] == "boundary"}

    core_edge_count = sum(1 for e in graph.edges.data() if e[2]["type"] == "core")
    boundary_edge_count = sum(1 for e in graph.edges.data() if e[2]["type"] == "boundary")

    graph.graph["core_nodes"] = len(core_nodes)
    graph.graph["outer_boundary_nodes"] = len(boundary_nodes)
    graph.graph["inner_boundary_nodes"] = len(inner_boundary_nodes)
    graph.graph["core_edges"] = core_edge_count
    graph.graph["boundary_edges"] = boundary_edge_count

    if verbose:  # Log network statistics
        logging.info("core edges: %d, boundary edges: %d" % (core_edge_count, boundary_edge_count))
        logging.info("core nodes: %d, (outer) boundary nodes: %d" % (len(core_nodes), len(boundary_nodes)))
        logging.info("core nodes with boundary edges: %d" % len(inner_boundary_nodes))


def infer_graph_attributes(graph: nx.DiGraph, relation_translator: Optional[RelationTranslator] = None, verbose=True):
    # Quietly remove nodes without edges
    graph.remove_nodes_from(list(nx.isolates(graph)))

    # Partition core and boundary nodes
    boundary_nodes, core_nodes = infer_node_type(graph)
    if len(core_nodes) == 0:
        raise ValueError("The network does not contain any core nodes.")
    if len(boundary_nodes) == 0:
        raise ValueError("The network does not contain any boundary nodes.")

    # Compute node type and indices, add data to graph instance
    enumerate_nodes(graph, boundary_nodes, core_nodes)

    remove_invalid_graph_elements(graph)

    # Compute edge weight and interaction type
    infer_edge_attributes(graph, relation_translator)

    # Add stats to metadata
    infer_metadata(graph, verbose)

    return graph
