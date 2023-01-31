import igraph

from .io.network import import_network_npa, write_edge_list, write_matrix
from .npa.preprocess import count_downstream_edges, compute_node_degree, compute_adjacency, compute_laplacians
from .npa.core import npa_value_diffusion, network_perturbation_amplitude

from .Dataset import ContrastDataset


class CBN:
    def __init__(self, input_format='NPA', backbone_network=None, downstream_network=None):
        match input_format:
            case 'NPA':
                if backbone_network is None or not isinstance(backbone_network, str):
                    raise TypeError("Argument backbone_network is not a file path.")
                if downstream_network is None or not isinstance(downstream_network, str):
                    raise TypeError("Argument downstream_network is not a file path.")

                self._backbone_node_count, self._downstream_node_count, self._node_name, \
                    self._backbone_edges, self._downstream_edges = \
                    import_network_npa(backbone_network, downstream_network)

                # Note: this should likely be factored out of match statement

                self._downstream_edge_count = count_downstream_edges(self._backbone_node_count, self._downstream_edges)

                self._node_degree, _, _ = compute_node_degree(self._backbone_node_count + self._downstream_node_count,
                                                              self._backbone_edges, self._downstream_edges,
                                                              self._downstream_edge_count)

                self._adjacency = compute_adjacency(self._backbone_node_count + self._downstream_node_count,
                                                    self._backbone_edges, self._downstream_edges,
                                                    self._downstream_edge_count)

                self._laplacian_backbone, self._laplacian_backbone_signless, \
                    self._laplacian_downstream, self._diffusion_matrix = \
                    compute_laplacians(self._backbone_node_count, self._downstream_node_count,
                                       self._node_degree, self._downstream_edge_count, self._adjacency)

            case 'igraph':
                if backbone_network is None or not isinstance(backbone_network, igraph.Graph):
                    raise TypeError("Argument backbone_network is not an igraph.Graph instance.")
                if downstream_network is None or not isinstance(downstream_network, igraph.Graph):
                    raise TypeError("Argument downstream_network is not an igraph.Graph instance.")

                # TODO: Initialize attributes

                raise ValueError("Input format %s is not currently supported." % input_format)

            case '_':
                raise ValueError("Input format %s is not supported." % input_format)

    def diffuse_values(self, dataset: ContrastDataset, diffusion_method="NPA"):
        match diffusion_method:
            case 'NPA':
                return npa_value_diffusion(self._diffusion_matrix, self._node_name,
                                           dataset.get_fold_changes(), dataset.get_node_names())
            case '_':
                raise ValueError("Diffusion method %s is not supported." % diffusion_method)

    def compute_perturbation(self, dataset: ContrastDataset, diffusion_method="NPA", perturbation_method="NPA"):
        match diffusion_method:
            case 'NPA':
                diffused_values = npa_value_diffusion(self._diffusion_matrix, self._node_name,
                                                      dataset.get_fold_changes(), dataset.get_node_names())
            case '_':
                raise ValueError("Diffusion method %s is not supported." % diffusion_method)

        match perturbation_method:
            case 'NPA':
                return network_perturbation_amplitude(len(self._backbone_edges), self._laplacian_backbone_signless,
                                                      diffused_values)
            case '_':
                raise ValueError("Perturbation method %s is not supported." % perturbation_method)

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

            write_edge_list(self._backbone_edges, node_names, output_file, mode)
            mode = 'a'

        if edges in ['all', 'downstream']:
            if headings:
                with open(output_file, mode) as f:
                    f.write("# DOWNSTREAM NETWORK\n")
                mode = 'a'

            write_edge_list(self._downstream_edges, node_names, output_file, mode)

    def write_laplacians(self, lb_file=None, ld_file=None, lbs_file=None, fmt='array2string'):
        if lb_file is not None:
            if not isinstance(lb_file, str):
                raise TypeError("Argument lb_file is not a file path.")
            write_matrix(self._laplacian_backbone, output_file=lb_file, fmt=fmt)

        if ld_file is not None:
            if not isinstance(ld_file, str):
                raise TypeError("Argument lb_file is not a file path.")
            write_matrix(self._laplacian_downstream, output_file=ld_file, fmt=fmt)

        if lbs_file is not None:
            if not isinstance(lbs_file, str):
                raise TypeError("Argument lb_file is not a file path.")
            write_matrix(self._laplacian_backbone_signless, output_file=lbs_file, fmt=fmt)
