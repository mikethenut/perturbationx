from typing import Optional
import logging

import networkx as nx
import py4cytoscape as p4c
from py4cytoscape.py4cytoscape_utils import DEFAULT_BASE_URL

from perturbationx.result import NPAResult
from perturbationx.resources import DEFAULT_NODE_COLOR, DEFAULT_GRADIENT
from .cytoscape import set_boundary_display, edge_to_p4c_format, \
    color_nodes_by_column, highlight_subgraph, isolate_subgraph, extract_subgraph, clear_bypass


class NPAResultDisplay:
    """Class to display results from NPA analysis.

    :param graph: NetworkX graph object
    :type graph: nx.Graph
    :param results: NPA results object
    :type results: NPAResults
    :param network_style: Name of the network style to use
    :type network_style: str
    :param network_suid: SUID of the network to display results in.
    :type network_suid: int
    :param cytoscape_url: Cytoscape URL. Defaults to DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
    :type cytoscape_url: str, optional
    """

    def __init__(self, graph: nx.Graph, results: NPAResult, network_style: str, network_suid: int,
                 cytoscape_url=DEFAULT_BASE_URL):
        """Construct a new NPAResultDisplay object.
        """
        self._graph = graph
        self._results = results
        self._network_suid = network_suid
        self._network_style = network_style
        self._cytoscape_url = cytoscape_url

    def reset_display(self, display_boundary: Optional[bool] = None, reset_color=False,
                      reset_highlight=False, reset_visibility=False):
        """Reset the display of the network.

        :param display_boundary: Whether to display the boundary of the network. Defaults to None, which does not
            change the current setting.
        :type display_boundary: bool, optional
        :param reset_color: Whether to reset the color of the nodes. Defaults to False.
        :type reset_color: bool, optional
        :param reset_highlight: Whether to reset the highlight of the nodes and edges. Defaults to False.
        :type reset_highlight: bool, optional
        :param reset_visibility: Whether to reset the visibility of the nodes and edges. Defaults to False.
        :type reset_visibility: bool, optional
        :return: SUID of the network.
        :rtype: int
        """

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

    def color_nodes(self, dataset: str, attribute: str, gradient=DEFAULT_GRADIENT, default_color=DEFAULT_NODE_COLOR):
        """Color nodes by a given attribute.

        :param dataset: The dataset to color nodes by.
        :type dataset: str
        :param attribute: The attribute to color nodes by.
        :type attribute: str
        :param gradient: The gradient to use. Defaults to DEFAULT_GRADIENT ("#2B80EF", "#EF3B2C").
        :type gradient: (str, str), optional
        :param default_color: The default color to use. Defaults to DEFAULT_NODE_COLOR ("#FEE391").
        :type default_color: str, optional
        :return: SUID of the network.
        :rtype: int
        """
        self.reset_display(reset_color=True)

        data_column = dataset + ' ' + attribute
        color_nodes_by_column(data_column, self._network_suid, gradient, default_color,
                              self._network_style, self._cytoscape_url)

        return self._network_suid

    def highlight_leading_nodes(self, dataset: str, cutoff=0.8, attr="contribution", abs_value=True,
                                include_shortest_paths="none", path_length_tolerance=0,
                                include_neighbors=0, neighborhood_type="union"):
        """Highlight leading nodes.

        :param dataset: The dataset to highlight leading nodes for.
        :type dataset: str
        :param cutoff: The cutoff to use when determining leading nodes. Defaults to 0.8.
        :type cutoff: float, optional
        :param attr: The attribute to use when determining leading nodes. Defaults to "contribution".
        :type attr: str, optional
        :param abs_value: Whether to use the absolute value of the attribute. Defaults to True.
        :type abs_value: bool, optional
        :param include_shortest_paths: If "directed", the directed shortest paths between the nodes will be included.
            If "undirected", the undirected shortest paths between the nodes will be included. If "none",
            no shortest paths will be included. Defaults to "none".
        :type include_shortest_paths: str, optional
        :param path_length_tolerance: The tolerance for the length of the shortest paths. If 0, only the shortest paths
            are returned. If length_tolerance is an integer, it is interpreted as an absolute length. If
            length_tolerance is a float, it is interpreted as a percentage of the length of the shortest path.
            Defaults to 0.
        :type path_length_tolerance: int | float, optional
        :param include_neighbors: The maximum distance from leading nodes that neighbors can be. If 0, no neighbors
            will be included. If 1, only the neighbors of the nodes will be included. If 2, the neighbors of the
            neighbors of the nodes will be included, and so on. Defaults to 0.
        :type include_neighbors: int, optional
        :param neighborhood_type: The type of neighborhood to include. Can be one of "union" or "intersection".
            If "union", all nodes within the maximum distance from any leading node are returned. If "intersection",
            only nodes within the maximum distance from all leading nodes are returned. Defaults to "union".
        :type neighborhood_type: str, optional
        :return: SUID of the network.
        :rtype: int
        """
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

    def extract_leading_nodes(self, dataset: str, cutoff=0.8, attr="contribution", abs_value=True,
                              inplace=True, include_shortest_paths="none", path_length_tolerance=0,
                              include_neighbors=0, neighborhood_type="union"):
        """Extract leading nodes.

        :param dataset: The dataset to extract leading nodes for.
                :type dataset: str
        :param cutoff: The cutoff to use when determining leading nodes. Defaults to 0.8.
        :type cutoff: float, optional
        :param attr: The attribute to use when determining leading nodes. Defaults to "contribution".
        :type attr: str, optional
        :param abs_value: Whether to use the absolute value of the attribute. Defaults to True.
        :type abs_value: bool, optional
        :param inplace: Whether to extract the leading nodes in-place. Defaults to True. If True, the network will be
            modified by hiding all nodes and edges that are not leading nodes. If False, a new network will be created
            with only the leading nodes.
        :type inplace: bool, optional
        :param include_shortest_paths: If "directed", the directed shortest paths between the nodes will be included.
            If "undirected", the undirected shortest paths between the nodes will be included. If "none",
            no shortest paths will be included. Defaults to "none".
        :type include_shortest_paths: str, optional
        :param path_length_tolerance: The tolerance for the length of the shortest paths. If 0, only the shortest paths
            are returned. If length_tolerance is an integer, it is interpreted as an absolute length. If
            length_tolerance is a float, it is interpreted as a percentage of the length of the shortest path.
            Defaults to 0.
        :type path_length_tolerance: int | float, optional
        :param include_neighbors: The maximum distance from leading nodes that neighbors can be. If 0, no neighbors
            will be included. If 1, only the neighbors of the nodes will be included. If 2, the neighbors of the
            neighbors of the nodes will be included, and so on. Defaults to 0.
        :type include_neighbors: int, optional
        :param neighborhood_type: The type of neighborhood to include. Can be one of "union" or "intersection".
            If "union", all nodes within the maximum distance from any leading node are returned. If "intersection",
            only nodes within the maximum distance from all leading nodes are returned. Defaults to "union".
        :type neighborhood_type: str, optional
        :return: SUID of the network.
        :rtype: int
        """
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
        """Retrieve the results object for this display.

        :return: The results object.
        :rtype: NPAResults
        """
        return self._results
