import os

__all__ = [
    "DEFAULT_DATA_COLS", "STYLE_XML", "DEFAULT_STYLE", "DEFAULT_NODE_COLOR", "DEFAULT_GRADIENT",
    "DEFAULT_EDGE_WIDTH", "DEFAULT_NODE_BORDER_WIDTH", "get_style_xml_path"
]


# io
DEFAULT_DATA_COLS = ("subject", "object", "relation", "type")


# visualisation
STYLE_XML = "default-style.xml"
DEFAULT_STYLE = "bnpa-default"
DEFAULT_NODE_COLOR = "#FEE391"
DEFAULT_GRADIENT = ("#2B80EF", "#EF3B2C")
DEFAULT_EDGE_WIDTH = 2.0
DEFAULT_NODE_BORDER_WIDTH = 4.0


def get_style_xml_path():
    module_path = os.path.abspath(__file__)
    resource_path = os.path.join(os.path.dirname(module_path), STYLE_XML)
    return resource_path
