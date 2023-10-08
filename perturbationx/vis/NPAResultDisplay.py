from typing import Optional
import logging

import py4cytoscape as p4c
from py4cytoscape.py4cytoscape_utils import DEFAULT_BASE_URL

from perturbationx.resources import DEFAULT_NODE_COLOR, DEFAULT_GRADIENT
from .cytoscape import set_boundary_display, edge_to_p4c_format, \
    color_nodes_by_column, highlight_subgraph, isolate_subgraph, extract_subgraph, clear_bypass


class NPAResultDisplay:
    def __init__(self, graph, results, network_style, network_suid, cytoscape_url=DEFAULT_BASE_URL):
        self._graph = graph
        self._results = results
        self._network_suid = network_suid
        self._network_style = network_style
        self._cytoscape_url = cytoscape_url

    def reset_display(self, display_boundary: Optional[bool] = None, reset_color=False,
                      reset_highlight=False, reset_visibility=False):
        logging.getLogger().handlers.clear()  # Block logging to stdout

        p4c.set_current_network(self._network_suid, base_url=self._cytoscape_url)

        # Adjust boundary display as requested
        if display_boundary is not None:
            set_boundary_display(self._graph, display_boundary, self._network_suid, self._cytoscape_url)

        # Get core nodes and edges
        core_nodes = [n for n in self._graph.nodes if self._graph.nodes[n]["type"] == "core"]
        core_edges = [edge_to_p4c_format(src, trg, self._graph[src][trg]["interaction"])
                      for src, trg in self._graph.edges if self._graph[src][trg]["type"] == "core"]

        # Note: CyREST is very fussy with bypasses
        # you cannot remove a bypass if all given nodes/edges don't have it

        # Reset visibility
        if reset_visibility:
            isolate_subgraph(self._graph, core_nodes, core_edges,
                             network_suid=self._network_suid, cytoscape_url=self._cytoscape_url)
            clear_bypass(core_edges, "edge", "EDGE_VISIBLE",
                         network_suid=self._network_suid, cytoscape_url=self._cytoscape_url)
            clear_bypass(core_nodes, "node", "NODE_VISIBLE",
                         network_suid=self._network_suid, cytoscape_url=self._cytoscape_url)

        # Reset highlight
        if reset_highlight:
            highlight_subgraph(core_nodes, core_edges, highlight_factor=1,
                               network_suid=self._network_suid, cytoscape_url=self._cytoscape_url)
            clear_bypass(core_nodes, "node", "NODE_BORDER_WIDTH",
                         network_suid=self._network_suid, cytoscape_url=self._cytoscape_url)
            clear_bypass(core_edges, "edge", "EDGE_WIDTH",
                         network_suid=self._network_suid, cytoscape_url=self._cytoscape_url)

        # Reset color
        if reset_color:
            p4c.style_mappings.delete_style_mapping(
                style_name=self._network_style, visual_prop="NODE_FILL_COLOR", base_url=self._cytoscape_url
            )

        return self._network_suid

    def color_nodes(self, dataset, attribute, gradient=DEFAULT_GRADIENT, default_color=DEFAULT_NODE_COLOR):
        self.reset_display(reset_color=True)

        data_column = dataset + ' ' + attribute
        color_nodes_by_column(data_column, self._network_suid, gradient, default_color,
                              self._network_style, self._cytoscape_url)

        return self._network_suid

    def highlight_leading_nodes(self, dataset, cutoff=0.8, attr="contribution", abs_value=True,
                                include_shortest_paths="none", path_length_tolerance=0,
                                include_neighbors=0, neighborhood_type="union"):
        self.reset_display(reset_highlight=True)

        # Get subgraph
        leading_nodes = self._results.get_leading_nodes(dataset, cutoff=cutoff, attr=attr, abs_value=abs_value)
        nodes, edges = self._results.get_node_subgraph(
            leading_nodes, include_shortest_paths, path_length_tolerance, include_neighbors, neighborhood_type
        )
        edges = [edge_to_p4c_format(*e) for e in edges]

        # Pass nodes and edges to highlight function
        highlight_subgraph(
            nodes, edges,
            network_suid=self._network_suid,
            cytoscape_url=self._cytoscape_url
        )
        return self._network_suid

    def extract_leading_nodes(self, dataset, cutoff=0.8, attr="contribution", abs_value=True,
                              inplace=True, include_shortest_paths="none", path_length_tolerance=0,
                              include_neighbors=0, neighborhood_type="union"):
        self.reset_display(reset_visibility=True)

        # Get subgraph
        leading_nodes = self._results.get_leading_nodes(dataset, cutoff=cutoff, attr=attr, abs_value=abs_value)
        nodes, edges = self._results.get_node_subgraph(
            leading_nodes, include_shortest_paths, path_length_tolerance, include_neighbors, neighborhood_type
        )
        edges = [edge_to_p4c_format(*e) for e in edges]

        # Set visibility
        if inplace:
            isolate_subgraph(
                self._graph, nodes, edges,
                network_suid=self._network_suid,
                cytoscape_url=self._cytoscape_url
            )
            return self._network_suid
        else:
            return extract_subgraph(
                nodes, edges,
                network_suid=self._network_suid,
                cytoscape_url=self._cytoscape_url
            )

    def get_results(self):
        return self._results