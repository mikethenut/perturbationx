import networkx as nx

__all__ = ["get_shortest_path_components"]


def get_shortest_path_components(graph: nx.DiGraph, endpoints: set, directed=False, length_tolerance=0):
    """Find paths between nodes in a graph and return the nodes and edges in those paths.

    :param graph: The graph to search for paths in.
    :type graph: nx.DiGraph
    :param endpoints: The nodes to find paths between.
    :type endpoints: set
    :param directed: Whether to search for directed or undirected shortest paths.
    :type directed: bool, optional
    :param length_tolerance: The tolerance for the length of the shortest paths. If 0, only the shortest paths
            are returned. If length_tolerance is an integer, it is interpreted as an absolute length. If
            length_tolerance is a float, it is interpreted as a percentage of the length of the shortest path.
            Defaults to 0.
    :type length_tolerance: int | float, optional
    :raises ValueError: If length_tolerance is not a number or is negative.
    :return: The nodes and edges in the paths. They are returned as a pair of lists.
    :rtype: tuple
    """

    try:
        if type(length_tolerance) != int and type(length_tolerance) != float:
            length_tolerance = float(length_tolerance)
            if length_tolerance.is_integer():
                length_tolerance = int(length_tolerance)
    except ValueError:
        raise ValueError("Argument length_tolerance must be a number.")

    if length_tolerance < 0.:
        raise ValueError("Argument length_tolerance must be >= 0.")

    dir_graph = graph
    if not directed:
        graph = graph.to_undirected()

    endpoint_pairs = set()
    for n in endpoints:
        for m in endpoints:
            # If undirected graph, we only need to consider one ordering
            if n != m and (directed or (m, n) not in endpoint_pairs):
                endpoint_pairs.add((n, m))

    nodes, edges = set(), set()
    for src, trg in endpoint_pairs:
        try:
            shortest_len = nx.shortest_path_length(graph, source=src, target=trg)

            if type(length_tolerance) == int:
                max_len = shortest_len + length_tolerance
            else:
                max_len = shortest_len * (1. + length_tolerance)

            paths = nx.all_simple_paths(graph, source=src, target=trg, cutoff=max_len)
            for p in paths:
                nodes.update(p)
                for n, m in zip(p, p[1:]):
                    if dir_graph.has_edge(n, m):
                        edges.add((n, m, dir_graph[n][m]["interaction"]))
                    elif dir_graph.has_edge(m, n):
                        edges.add((m, n, dir_graph[m][n]["interaction"]))

        except nx.NetworkXNoPath:
            pass

    return nodes, edges
