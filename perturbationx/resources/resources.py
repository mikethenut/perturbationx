import sys
import os
import logging

__all__ = [
    "DEFAULT_DATA_COLS", "DEFAULT_LOGGING_KWARGS", "STYLE_XML", "DEFAULT_STYLE", "DEFAULT_NODE_COLOR",
    "DEFAULT_GRADIENT", "DEFAULT_EDGE_WIDTH", "DEFAULT_NODE_BORDER_WIDTH", "get_style_xml_path"
]


# io
DEFAULT_DATA_COLS = ("subject", "object", "relation", "type")
DEFAULT_LOGGING_KWARGS = {
    "stream": sys.stdout,
    "level": logging.INFO,
    "format": "%(asctime)s %(levelname)s -- %(message)s",
}


# visualisation
STYLE_XML = "default-style.xml"
DEFAULT_STYLE = "perturbationx-default"
DEFAULT_NODE_COLOR = "#FEE391"
DEFAULT_GRADIENT = ("#2B80EF", "#EF3B2C")
DEFAULT_EDGE_WIDTH = 2.0
DEFAULT_NODE_BORDER_WIDTH = 4.0


def get_style_xml_path():
    """Get the path to the default style xml file.
    :return: The path to the default style xml file.
    :rtype: str
    """
    module_path = os.path.abspath(__file__)
    resource_path = os.path.join(os.path.dirname(module_path), STYLE_XML)
    return resource_path


def get_osmotic_stress_core_path():
    """Get the path to the osmotic stress core network file. The network is a subset of the osmotic stress network
    from the Causal Biological Networks database (https://www.causalbionet.com/, network ID osmotic_stress_2.0_hs).
    All non-causal edges were removed and the largest connected component was then selected.

    :return: The path to the osmotic stress core network file.
    :rtype: str
    """
    module_path = os.path.abspath(__file__)
    resource_path = os.path.join(os.path.dirname(module_path), "osmotic_stress_core.tsv")
    return resource_path


def get_osmotic_stress_boundary_path():
    """Get the path to the osmotic stress boundary network file. The network was generated using the Barabási-Albert
    model and is not biological in nature. It contains 100 outer boundary nodes and 500 boundary edges.

    :return: The path to the osmotic stress boundary network file.
    :rtype: str
    """
    module_path = os.path.abspath(__file__)
    resource_path = os.path.join(os.path.dirname(module_path), "osmotic_stress_boundary.tsv")
    return resource_path


def get_osmotic_stress_dataset_paths():
    """Get the paths to the osmotic stress dataset files. The datasets were generated using a genetic algorithm and
    are not biological in nature.

    :return: The paths to the osmotic stress dataset files.
    :rtype: list
    """
    module_path = os.path.abspath(__file__)
    resource_paths = [
        os.path.join(os.path.dirname(module_path), "osmotic_stress_dataset_" + idx + ".csv")
        for idx in ["1", "2", "3"]
    ]
    return resource_paths
