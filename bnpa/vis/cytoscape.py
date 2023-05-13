from typing import Optional

import pandas as pd
import py4cytoscape as p4c
from py4cytoscape.exceptions import CyError
from py4cytoscape.py4cytoscape_utils import DEFAULT_BASE_URL
from py4cytoscape.tables import load_table_data

from bnpa.resources.resources import DEFAULT_STYLE, get_style_xml_path


def init_cytoscape(graph, title, collection, init_boundary: Optional[bool] = False,
                   network_suid=None, cytoscape_url=DEFAULT_BASE_URL):
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


def load_network_data(dataframe, table, network_suid, cytoscape_url=DEFAULT_BASE_URL):
    dataframe = dataframe.copy()
    # Rename MultiIndex columns
    dataframe.columns = [' '.join(col) if type(col) != str else col
                         for col in dataframe.columns]
    for c in dataframe:
        column = dataframe[[c]].dropna().reset_index()
        if column.shape[0] > 0:
            load_table_data(
                column, data_key_column="index", table=table,
                network=network_suid, base_url=cytoscape_url
            )


def set_boundary_display(graph, show_boundary, network_suid, cytoscape_url=DEFAULT_BASE_URL):
    node_table = p4c.tables.get_table_columns(table="node", columns=["name", "type"],
                                              network=network_suid, base_url=cytoscape_url)
    boundary_nodes = node_table.loc[node_table["type"] == "boundary"]["name"].tolist()

    if len(boundary_nodes) > 0 and show_boundary is False:
        p4c.style_bypasses.hide_nodes(boundary_nodes, network=network_suid, base_url=cytoscape_url)
    elif len(boundary_nodes) > 0 and show_boundary is True:
        # The unhide function deletes the bypass, which throws an error if the bypass does not exist
        p4c.style_bypasses.set_node_property_bypass(
            boundary_nodes, new_values='true', visual_property='NODE_VISIBLE',
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
            # Edge name format used by p4c is 'source (interaction) target'
            "%s (%s) %s" % (src, graph[src][trg]["interaction"], trg): graph[src][trg]
            for src, trg in boundary_edges
        }, orient="index")
        load_network_data(edge_data, "edge", network_suid, cytoscape_url)
