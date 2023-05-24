import networkx as nx


def get_all_neighbors(graph: nx.Graph, source_nodes: set, max_distance=0):
    if max_distance < 1:
        return source_nodes

    neighbors = set()
    for n in source_nodes:
        for m in graph.neighbors(n):
            if m not in source_nodes:
                neighbors.add(m)

    return get_all_neighbors(graph, neighbors, max_distance - 1).union(source_nodes)


def get_common_neighbors(graph: nx.Graph, source_nodes: set, max_distance=0):
    neighborhoods = {n: get_all_neighbors(graph, {n}, max_distance) for n in source_nodes}

    neighbors = None
    for n in neighborhoods:
        if neighbors is None:
            neighbors = neighborhoods[n]
        else:
            neighbors = neighbors.intersection(neighborhoods[n])

    return neighbors


def get_neighborhood(graph: nx.DiGraph, source_nodes: set, max_distance=0, neighborhood_type="union"):
    """
    Get the neighbors of a set of nodes in a graph.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to search for neighbors in.
    source_nodes : set
        The nodes to find neighbors of.
    max_distance : int
        The maximum distance from source nodes that neighbors can be.
    neighborhood_type : str
        The type of neighborhood to find. Can be one of "union" or "intersection". If "union", all nodes within the
        maximum distance from any source node are returned. If "intersection", only nodes within the maximum distance
        from all source nodes are returned.

    Returns
    -------
    nodes : set
        The nodes in the neighborhood.
    edges : set
        The edges in the neighborhood.
    """

    if max_distance < 0.:
        raise ValueError("Argument max_distance must be >= 0.")

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
