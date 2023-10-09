import sys
import io
from typing import Optional
from contextlib import redirect_stderr

import pandas as pd
import py4cytoscape as p4c
import networkx as nx
from py4cytoscape.exceptions import CyError
from py4cytoscape.py4cytoscape_utils import DEFAULT_BASE_URL
from py4cytoscape.tables import load_table_data

from perturbationx.resources import *


def edge_to_p4c_format(src: str, trg: str, interaction: str):
    """Convert an edge to the format used by py4cytoscape.

    :param src: Source node name.
    :type src: str
    :param trg: Target node name.
    :type trg: str
    :param interaction: Interaction type.
    :type interaction: str
    :return: Edge name.
    :rtype: str
    """
    # Edge name format used by p4c is 'source (interaction) target'
    return "%s (%s) %s" % (src, interaction, trg)


def init_cytoscape(graph: nx.Graph, title: str, collection: str, init_boundary: Optional[bool] = False,
                   network_suid=None, cytoscape_url=DEFAULT_BASE_URL):
    """Initialize a Cytoscape network from a NetworkX graph.

    :param graph: NetworkX graph.
    :type graph: nx.Graph
    :param title: Network title.
    :type title: str
    :param collection: Network collection.
    :type collection: str
    :param init_boundary: Whether to initialize boundary nodes. Defaults to False.
    :type init_boundary: bool, optional
    :param network_suid: Network SUID to update. Defaults to None. If None or invalid, a new network is created.
    :type network_suid: int, optional
    :param cytoscape_url: Cytoscape URL. Defaults to DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
    :type cytoscape_url: str, optional
    :return: Network SUID.
    :rtype: int
    """
    if network_suid is not None:
        try:
            p4c.set_current_network(network_suid, base_url=cytoscape_url)
        except CyError:
            # Network does not exist anymore
            network_suid = None

    if network_suid is None:
        # If boundary nodes should not be included, remove them from the graph
        if not init_boundary:
            graph = graph.copy()
            boundary_nodes = [n for n in graph.nodes if graph.nodes[n]["type"] == "boundary"]
            graph.remove_nodes_from(boundary_nodes)
        network_suid = p4c.networks.create_network_from_networkx(
            graph, base_url=cytoscape_url, title=title, collection=collection
        )

    # Import default style if not already present
    if DEFAULT_STYLE not in p4c.styles.get_visual_style_names(base_url=cytoscape_url):
        p4c.styles.import_visual_styles(filename=get_style_xml_path(), base_url=cytoscape_url)

    return network_suid


def load_network_data(dataframe: pd.DataFrame, table: str, network_suid: int, cytoscape_url=DEFAULT_BASE_URL):
    """Load a pandas DataFrame into a Cytoscape table.

    :param dataframe: The DataFrame to load.
    :type dataframe: pd.DataFrame
    :param table: The table to update.
    :type table: str
    :param network_suid: The network to update.
    :type network_suid: int
    :param cytoscape_url: Cytoscape URL. Defaults to DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
    :type cytoscape_url: str, optional
    """
    dataframe = dataframe.copy()
    # Rename MultiIndex columns
    dataframe.columns = [col if isinstance(col, str) else ' '.join(col)
                         for col in dataframe.columns]
    for c in dataframe:
        column = dataframe[[c]].dropna().reset_index()
        if column.shape[0] > 0:
            load_table_data(
                column, data_key_column="index", table=table,
                network=network_suid, base_url=cytoscape_url
            )


def set_boundary_display(graph: nx.Graph, show_boundary: bool, network_suid, cytoscape_url=DEFAULT_BASE_URL):
    """Set the display of boundary nodes.

    :param graph: The graph to load boundary nodes from.
    :type graph: nx.Graph
    :param show_boundary: Whether to show boundary nodes. If boundary nodes were not loaded during initialization,
                            they can be loaded in here. If boundary nodes have already been loaded, they are hidden.
    :type show_boundary: bool
    :param network_suid: The network to update.
    :type network_suid: int
    :param cytoscape_url: Cytoscape URL. Defaults to DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
    :type cytoscape_url: str, optional
    """
    node_table = p4c.tables.get_table_columns(table="node", columns=["name", "type"],
                                              network=network_suid, base_url=cytoscape_url)
    boundary_nodes = node_table.loc[node_table["type"] == "boundary"]["name"].tolist()

    if len(boundary_nodes) > 0 and show_boundary is False:
        p4c.style_bypasses.hide_nodes(boundary_nodes, network=network_suid, base_url=cytoscape_url)
    elif len(boundary_nodes) > 0 and show_boundary is True:
        # The unhide function deletes the bypass, which throws an error if the bypass does not exist
        p4c.style_bypasses.set_node_property_bypass(
            boundary_nodes, new_values="true", visual_property="NODE_VISIBLE",
            network=network_suid, base_url=cytoscape_url
        )
    elif show_boundary is True:
        # No boundary nodes in Cytoscape :( We need to load them in
        boundary_nodes = [n for n in graph.nodes if graph.nodes[n]["type"] == "boundary"]
        p4c.networks.add_cy_nodes(boundary_nodes, network=network_suid, base_url=cytoscape_url)

        # Find boundary edges and all interaction types
        boundary_edges = [(src, trg) for src, trg in graph.edges
                          if graph[src][trg]["type"] == "boundary"]
        interactions = {graph[src][trg]["interaction"] for src, trg in boundary_edges}

        # Load edges per interaction
        for interaction in interactions:
            boundary_edge_subset = [(src, trg) for src, trg in boundary_edges
                                    if graph[src][trg]["interaction"] == interaction]
            if len(boundary_edge_subset) > 0:
                p4c.networks.add_cy_edges(boundary_edge_subset, edge_type=interaction, directed=True,
                                          network=network_suid, base_url=cytoscape_url)

        # Load node data
        node_data = pd.DataFrame.from_dict({
            n: graph.nodes[n] for n in boundary_nodes
        }, orient="index")
        load_network_data(node_data, "node", network_suid, cytoscape_url)

        # Load edge data
        edge_data = pd.DataFrame.from_dict({
            edge_to_p4c_format(src, trg, graph[src][trg]["interaction"]): graph[src][trg]
            for src, trg in boundary_edges
        }, orient="index")
        load_network_data(edge_data, "edge", network_suid, cytoscape_url)


def color_nodes_by_column(data_column: str, network_suid: int, gradient=DEFAULT_GRADIENT,
                          default_color=DEFAULT_NODE_COLOR, style=DEFAULT_STYLE, cytoscape_url=DEFAULT_BASE_URL):
    """Color nodes by a column in the node table.

    :param data_column: The column to color by.
    :type data_column: str
    :param network_suid: The network to update.
    :type network_suid: int
    :param gradient: The gradient to use. Defaults to DEFAULT_GRADIENT ("#2B80EF", "#EF3B2C").
    :type gradient: tuple, optional
    :param default_color: The default node color to use. Defaults to DEFAULT_NODE_COLOR ("#FEE391").
    :type default_color: str, optional
    :param style: The style to use. Defaults to DEFAULT_STYLE ("perturbationx-default").
    :type style: str, optional
    :param cytoscape_url: Cytoscape URL. Defaults to DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
    :type cytoscape_url: str, optional
    """
    node_table = p4c.tables.get_table_columns(
        table="node", columns=[data_column], network=network_suid, base_url=cytoscape_url
    )
    data_range = [node_table.min()[data_column], node_table.max()[data_column]]

    p4c.style_mappings.set_node_color_mapping(
        data_column, data_range, colors=list(gradient), default_color=default_color,
        style_name=style, network=network_suid, base_url=cytoscape_url
    )


def highlight_subgraph(nodes: list, edges: list, network_suid: int, highlight_factor=3, cytoscape_url=DEFAULT_BASE_URL):
    """Highlight a subgraph.

    :param nodes: The nodes to highlight.
    :type nodes: list
    :param edges: The edges to highlight.
    :type edges: list
    :param network_suid: The network to update.
    :type network_suid: int
    :param highlight_factor: The factor to multiply the default width by. Defaults to 3.
    :type highlight_factor: float, optional
    :param cytoscape_url: Cytoscape URL. Defaults to DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
    :type cytoscape_url: str, optional
    """
    p4c.style_bypasses.set_node_border_width_bypass(
        nodes, DEFAULT_NODE_BORDER_WIDTH * highlight_factor,
        network=network_suid, base_url=cytoscape_url
    )
    p4c.style_bypasses.set_edge_line_width_bypass(
        edges, DEFAULT_EDGE_WIDTH * highlight_factor,
        network=network_suid, base_url=cytoscape_url
    )


def isolate_subgraph(graph: nx.Graph, nodes: list, edges: list, network_suid: int, cytoscape_url=DEFAULT_BASE_URL):
    """Isolate a subgraph.

    :param graph: The graph displayed in Cytoscape.
    :type graph: nx.Graph
    :param nodes: The nodes to isolate.
    :type nodes: list
    :param edges: The edges to isolate.
    :type edges: list
    :param network_suid: The network to update.
    :type network_suid: int
    :param cytoscape_url: Cytoscape URL. Defaults to DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
    :type cytoscape_url: str, optional
    """
    # Set edge visibility
    core_edges = [edge_to_p4c_format(src, trg, graph[src][trg]["interaction"])
                  for src, trg in graph.edges if graph[src][trg]["type"] == "core"]
    edge_visibility = ["true" if e in edges else "false" for e in core_edges]
    p4c.style_bypasses.set_edge_property_bypass(
        core_edges, new_values=edge_visibility, visual_property="EDGE_VISIBLE",
        network=network_suid, base_url=cytoscape_url
    )

    # Set node visibility
    core_nodes = [n for n in graph.nodes if graph.nodes[n]["type"] == "core"]
    node_visibility = ["true" if n in nodes else "false" for n in core_nodes]
    p4c.style_bypasses.set_node_property_bypass(
        core_nodes, new_values=node_visibility, visual_property="NODE_VISIBLE",
        network=network_suid, base_url=cytoscape_url
    )


def extract_subgraph(nodes: list, edges: list, network_suid: int, cytoscape_url=DEFAULT_BASE_URL):
    """Extract a subgraph.

    :param nodes: The nodes to extract.
    :type nodes: list
    :param edges: The edges to extract.
    :type edges: list
    :param network_suid: The network to update.
    :type network_suid: int
    :param cytoscape_url: Cytoscape URL. Defaults to DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
    :type cytoscape_url: str, optional
    :return: The SUID of the extracted subnetwork.
    :rtype: int
    """
    return p4c.networks.create_subnetwork(
        nodes=nodes, edges=edges, exclude_edges=True,
        network=network_suid, base_url=cytoscape_url
    )


def clear_bypass(components: list, component_type: str, visual_property: str, network_suid: int,
                 cytoscape_url=DEFAULT_BASE_URL):
    """Clear a bypass.

    :param components: The components to clear the bypass for.
    :type components: list
    :param component_type: The component type. Must be 'node' or 'edge'.
    :type component_type: str
    :param visual_property: The visual property to clear the bypass for.
    :type visual_property: str
    :param network_suid: The network to update.
    :type network_suid: int
    :param cytoscape_url: Cytoscape URL. Defaults to DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
    :type cytoscape_url: str, optional
    :raises ValueError: If the component type is not 'node' or 'edge'.
    :raises CyError: If a CyREST error occurs.
    """
    fake_logger = io.StringIO()
    with redirect_stderr(fake_logger):
        try:
            match component_type:
                case "node":
                    p4c.style_bypasses.clear_node_property_bypass(
                        components, visual_property, network_suid, base_url=cytoscape_url
                    )
                case "edge":
                    p4c.style_bypasses.clear_edge_property_bypass(
                        components, visual_property, network_suid, base_url=cytoscape_url
                    )
                case _:
                    raise ValueError("Invalid component type: '%s'. Must be 'node' or 'edge'." % component_type)
        except CyError as e:
            if "Bypass Visual Property does not exist" not in str(e):
                print(fake_logger.getvalue(), file=sys.stderr)
                raise e
