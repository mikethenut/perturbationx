import py4cytoscape as p4c
from py4cytoscape.exceptions import CyError
from py4cytoscape.py4cytoscape_utils import DEFAULT_BASE_URL
from py4cytoscape.tables import load_table_data

from bnpa.resources.resources import DEFAULT_STYLE, get_style_xml_path


def init_cytoscape(graph, title, collection, node_data, network_suid=None, cytoscape_url=DEFAULT_BASE_URL):
    if network_suid is not None:
        try:
            p4c.set_current_network(network_suid, base_url=cytoscape_url)
        except CyError:
            # Network does not exist anymore
            network_suid = None

    if network_suid is None:
        network_suid = p4c.networks.create_network_from_networkx(
            graph, base_url=cytoscape_url, title=title, collection=collection
        )
        load_node_data(node_data, network_suid, cytoscape_url)

    # Import default style if not already present
    if DEFAULT_STYLE not in p4c.styles.get_visual_style_names(base_url=cytoscape_url):
        p4c.styles.import_visual_styles(filename=get_style_xml_path(), base_url=cytoscape_url)

    return network_suid


def load_node_data(dataframe, network_suid, cytoscape_url=DEFAULT_BASE_URL):
    dataframe.columns = [' '.join(col) for col in dataframe.columns]
    dataframe.reset_index(inplace=True)
    load_table_data(dataframe, data_key_column='index', network=network_suid, base_url=cytoscape_url)


