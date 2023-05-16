import networkx as nx


def get_shortest_path_components(graph, endpoints, directed):
    if not directed:
        graph = graph.to_undirected()

    endpoint_pairs = set()
    for n in endpoints:
        for m in endpoints:
            # If undirected graph, we only need to consider one direction
            if n != m and (directed or (m, n) not in endpoint_pairs):
                endpoint_pairs.add((n, m))

    nodes, edges = set(), set()
    for src, trg in endpoint_pairs:
        try:
            paths = nx.all_shortest_paths(graph, source=src, target=trg)

            for p in paths:
                nodes.update(p)
                for n, m in zip(p, p[1:]):
                    edges.add((n, m, graph[n][m]["interaction"]))

        except nx.NetworkXNoPath:
            pass

    return nodes, edges


def add_node_ancestors(node_set, node, parents):
    node_set.add(node)
    for p in parents[node]:
        if p in node_set:
            continue
        add_node_ancestors(node_set, p, parents)


def get_all_path_components(graph, endpoints, directed):
    # TODO: Rewrite this to handle directed and undirected graphs separately
    # Use bidirectional search in directed graphs
    # Use biconnected components in undirected graphs

    directed_edges = graph.edges
    if not directed:
        graph = graph.to_undirected()

    nodes = endpoints.copy()
    stack = list(endpoints)
    visited = set()
    parents = {n: set() for n in graph.nodes}

    while len(stack) > 0:
        n = stack.pop()
        visited.add(n)

        for m in graph.neighbors(n):
            # Node is trying to connect back to the only node it was found from
            if m in parents[n] and len(parents[n]) < 2:
                continue

            # If node connects to a node in target set, add it to the target set
            if m in nodes:
                add_node_ancestors(nodes, n, parents)
                continue

            if m not in visited:
                parents[m].add(n)
                if m not in stack:
                    stack.append(m)

    edges = set()
    for src, trg in directed_edges:
        if src in nodes and trg in nodes:
            edges.add((src, trg, graph[src][trg]["interaction"]))

    return nodes, edges


def get_path_components(graph: nx.DiGraph, endpoints: set, path_type="shortest", directed=False):
    """
    Finds paths between nodes in a graph and returns the nodes and edges in the paths.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to search for paths in.
    endpoints : set
        The nodes to find paths between.
    path_type : str
        The type of paths to find. Can be one of "shortest" or "all".
    directed : bool
        Whether to only consider directed paths.

    Returns
    -------
    nodes : set
        The nodes in the paths.
    edges : set
        The edges in the paths.
    """

    match path_type:
        case "shortest":
            return get_shortest_path_components(graph, endpoints, directed)
        case "all":
            return get_all_path_components(graph, endpoints, directed)
        case "none":
            return endpoints, set()
        case _:
            raise ValueError("Argument path_type must be 'none', 'shortest' or 'all'.")
