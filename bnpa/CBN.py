import igraph

from bnpa.io import _import_from_npa, _write_edge_list, _write_matrix
from bnpa.preprocess import _generate_graphs, _compute_matrices


class CBN:
    def __init__(self, input_format='NPA', backbone_network=None, downstream_network=None):
        match input_format:
            case 'NPA':
                if backbone_network is None or not isinstance(backbone_network, str):
                    raise TypeError("Argument backbone_network is not a file path.")
                if downstream_network is None or not isinstance(downstream_network, str):
                    raise TypeError("Argument downstream_network is not a file path.")

                self._backbone_node_count, self._downstream_node_count, \
                    self._backbone_edges,  self._downstream_edges, \
                    self._node_name, self._node_type, self._edge_type = \
                    _import_from_npa(backbone_network, downstream_network)

                # Note: graph_full might not be needed

                self._graph_full, self._graph_backbone, self._graph_downstream = \
                    _generate_graphs(self._backbone_node_count, self._backbone_edges,
                                     self._downstream_node_count, self._downstream_edges)

                # Note: this should likely be factored out of match statement

                self._laplacian_backbone, self._laplacian_backbone_signless, self._laplacian_downstream = \
                    _compute_matrices(self._backbone_node_count, self._node_type,
                                      self._graph_backbone, self._graph_downstream)

            case 'igraph':
                if backbone_network is None or not isinstance(backbone_network, igraph.Graph):
                    raise TypeError("Argument backbone_network is not an igraph.Graph instance.")
                if downstream_network is None or not isinstance(downstream_network, igraph.Graph):
                    raise TypeError("Argument downstream_network is not an igraph.Graph instance.")

                # TODO: Initialize attributes

                raise ValueError("Input format %s is not currently supported." % input_format)

            case '_':
                raise ValueError("Input format %s is not supported." % input_format)

    def write_edge_list(self, output_file="./output/network_edges.tsv", edges='all', name_nodes=True, headings=True):
        mode = 'w'
        node_names = None
        if name_nodes:
            node_names = self._node_name

        if edges in ['all', 'backbone']:
            if headings:
                with open(output_file, mode) as f:
                    f.write("# BACKBONE NETWORK\n")
                mode = 'a'

            _write_edge_list(self._backbone_edges, self._edge_type, node_names, output_file, mode)
            mode = 'a'

        if edges in ['all', 'downstream']:
            if headings:
                with open(output_file, mode) as f:
                    f.write("# DOWNSTREAM NETWORK\n")
                mode = 'a'

            _write_edge_list(self._downstream_edges, self._edge_type, node_names, output_file, mode)

    def write_laplacians(self, lb_file=None, ld_file=None, lbs_file=None, fmt='array2string'):
        if lb_file is not None:
            if not isinstance(lb_file, str):
                raise TypeError("Argument lb_file is not a file path.")
            _write_matrix(self._laplacian_backbone, output_file=lb_file, fmt=fmt)

        if ld_file is not None:
            if not isinstance(ld_file, str):
                raise TypeError("Argument lb_file is not a file path.")
            _write_matrix(self._laplacian_downstream, output_file=ld_file, fmt=fmt)

        if lbs_file is not None:
            if not isinstance(lbs_file, str):
                raise TypeError("Argument lb_file is not a file path.")
            _write_matrix(self._laplacian_backbone_signless, output_file=lbs_file, fmt=fmt)
