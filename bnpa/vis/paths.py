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


def get_all_directed_path_nodes(graph: nx.DiGraph, endpoints: set):
    descendants = endpoints.copy()
    out_stack = list(endpoints)
    while len(out_stack) > 0:
        parent = out_stack.pop()
        for child in graph.successors(parent):
            if child not in descendants:
                descendants.add(child)
                out_stack.append(child)

    ancestors = endpoints.copy()
    in_stack = list(endpoints)
    while len(in_stack) > 0:
        parent = in_stack.pop()
        for child in graph.predecessors(parent):
            if child not in ancestors:
                ancestors.add(child)
                in_stack.append(child)

    return ancestors.intersection(descendants)


def get_all_undirected_path_nodes(graph: nx.Graph, endpoints: set):
    bridges = list(nx.bridges(graph))
    for b in bridges:
        graph.remove_edge(*b)
    components = list(nx.connected_components(graph))

    bridge_components = {b: [] for b in bridges}
    component_neighbors = {}
    marked_components = set()
    root = None
    for idx, c in enumerate(components):
        if len(endpoints.intersection(c)) > 0:
            marked_components.add(idx)
            if root is None:
                root = idx

        component_neighbors[idx] = set()
        for b in bridges:
            if len(c.intersection(b)) > 0:
                bridge_components[b].append(idx)

    for bridge, c in bridge_components.items():
        component_neighbors[c[0]].add(c[1])
        component_neighbors[c[1]].add(c[0])

    visited = set()
    stack = [root]
    parent = {root: None}
    while len(stack) > 0:
        component = stack.pop()
        visited.add(component)
        for n in component_neighbors[component]:
            if n not in visited and n not in stack:
                stack.append(n)
                parent[n] = component

        if component in marked_components:
            component = parent[component]
            while component is not None and component not in marked_components:
                marked_components.add(component)
                component = parent[component]

    nodes = set()
    for idx in marked_components:
        nodes.update(components[idx])
    return nodes


def get_all_path_components(graph, endpoints, directed):
    if directed:
        nodes = get_all_directed_path_nodes(graph, endpoints)
    else:
        nodes = get_all_undirected_path_nodes(graph.to_undirected(), endpoints)

    edges = set()
    for src, trg in graph.edges:
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
