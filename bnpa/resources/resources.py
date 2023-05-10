import os

DEFAULT_STYLE = "bnpa-default"
STYLE_XML = "default-style.xml"


def get_style_xml_path():
    module_path = os.path.abspath(__file__)
    resource_path = os.path.join(os.path.dirname(module_path), STYLE_XML)
    return resource_path
