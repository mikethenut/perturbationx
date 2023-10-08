import networkx as nx

__all__ = ["get_neighborhood_components"]


def get_all_neighbors(graph: nx.Graph, source_nodes: set, max_distance=0):
    """Get all neighbors of a set of nodes in a graph.

    :param graph: The graph to search for neighbors in.
    :type graph: nx.Graph
    :param source_nodes: The nodes to find neighbors of.
    :type source_nodes: set
    :param max_distance: The maximum distance from leading nodes that neighbors can be. If 0, no neighbors
            will be included. If 1, only the neighbors of the nodes will be included. If 2, the neighbors of the
            neighbors of the nodes will be included, and so on. Defaults to 0.
    :type max_distance: int, optional
    :return: The neighbors of the source nodes.
    :rtype: set
    """
    if max_distance < 1:
        return source_nodes

    neighbors = set()
    for n in source_nodes:
        for m in graph.neighbors(n):
            if m not in source_nodes:
                neighbors.add(m)

    return get_all_neighbors(graph, neighbors, max_distance - 1).union(source_nodes)


def get_common_neighbors(graph: nx.Graph, source_nodes: set, max_distance=0):
    """Get the common neighbors of a set of nodes in a graph.

    :param graph: The graph to search for neighbors in.
    :type graph: nx.Graph
    :param source_nodes: The nodes to find neighbors of.
    :type source_nodes: set
    :param max_distance: The maximum distance from leading nodes that neighbors can be. If 0, no neighbors
            will be included. If 1, only the neighbors of the nodes will be included. If 2, the neighbors of the
            neighbors of the nodes will be included, and so on. Defaults to 0.
    :type max_distance: int, optional
    :return: The common neighbors of the source nodes.
    :rtype: set
    """

    neighborhoods = {n: get_all_neighbors(graph, {n}, max_distance) for n in source_nodes}

    neighbors = None
    for n in neighborhoods:
        if neighbors is None:
            neighbors = neighborhoods[n]
        else:
            neighbors = neighbors.intersection(neighborhoods[n])

    return neighbors


def get_neighborhood_components(graph: nx.DiGraph, source_nodes: set, max_distance=0, neighborhood_type="union"):
    """Get the neighborhood of a set of nodes in a graph.

    :param graph: The graph to search for neighbors in.
    :type graph: nx.DiGraph
    :param source_nodes: The nodes to find neighbors of.
    :type source_nodes: set
    :param max_distance: The maximum distance from leading nodes that neighbors can be. If 0, no neighbors
            will be included. If 1, only the neighbors of the nodes will be included. If 2, the neighbors of the
            neighbors of the nodes will be included, and so on. Defaults to 0.
    :type max_distance: int, optional
    :param neighborhood_type: The type of neighborhood to find. Can be one of "union" or "intersection".
            If "union", all nodes within the maximum distance from any leading node are returned. If "intersection",
            only nodes within the maximum distance from all leading nodes are returned. Defaults to "union".
    :type neighborhood_type: str, optional
    :return: The nodes and edges in the neighborhood. They are returned as a pair of lists.
    :rtype: tuple
    """

    if max_distance < 0.:
        raise ValueError("Argument max_distance must be >= 0.")
    elif max_distance == 0:
        return source_nodes, set()

    match neighborhood_type:
        case "union":
            nodes = get_all_neighbors(graph.to_undirected(), source_nodes, max_distance)
        case "intersection":
            nodes = get_common_neighbors(graph.to_undirected(), source_nodes, max_distance)
        case _:
            raise ValueError("Argument neighborhood_type must be 'union' or 'intersection'.")

    edges = set()
    for src, trg in graph.edges:
        if src in nodes and trg in nodes:
            edges.add((src, trg, graph[src][trg]["interaction"]))

    return nodes, edges
