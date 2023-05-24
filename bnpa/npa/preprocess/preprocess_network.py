import logging

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
    # TODO: Check that the core is connected and validate graph structure

    boundary_nodes, core_nodes = infer_node_type(graph)

    # Check that the network core and boundary are not empty
    if len(core_nodes) == 0:
        raise ValueError("The network does not contain any core nodes.")
    if len(boundary_nodes) == 0:
        raise ValueError("The network does not contain any boundary nodes.")

    # Compute indices and add data to graph instance
    enumerate_nodes(graph, boundary_nodes, core_nodes)

    # Compute edge weight and interaction type
    infer_edge_attributes(graph, relation_translator)

    # Add stats to metadata
    infer_metadata(graph, verbose)

    return graph
