:py:mod:`perturbationx.vis.cytoscape`
=====================================

.. py:module:: perturbationx.vis.cytoscape


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.vis.cytoscape.edge_to_p4c_format
   perturbationx.vis.cytoscape.init_cytoscape
   perturbationx.vis.cytoscape.load_network_data
   perturbationx.vis.cytoscape.set_boundary_display
   perturbationx.vis.cytoscape.color_nodes_by_column
   perturbationx.vis.cytoscape.highlight_subgraph
   perturbationx.vis.cytoscape.isolate_subgraph
   perturbationx.vis.cytoscape.extract_subgraph
   perturbationx.vis.cytoscape.clear_bypass



.. py:function:: edge_to_p4c_format(src: str, trg: str, interaction: str)

   Convert an edge to the format used by py4cytoscape.

   :param src: Source node name.
   :type src: str
   :param trg: Target node name.
   :type trg: str
   :param interaction: Interaction type.
   :type interaction: str
   :return: Edge name.
   :rtype: str


.. py:function:: init_cytoscape(graph: networkx.Graph, title: str, collection: str, init_boundary: Optional[bool] = False, network_suid=None, cytoscape_url=DEFAULT_BASE_URL)

   Initialize a Cytoscape network from a NetworkX graph.

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


.. py:function:: load_network_data(dataframe: pandas.DataFrame, table: str, network_suid: int, cytoscape_url=DEFAULT_BASE_URL)

   Load a pandas DataFrame into a Cytoscape table.

   :param dataframe: The DataFrame to load.
   :type dataframe: pd.DataFrame
   :param table: The table to update.
   :type table: str
   :param network_suid: The network to update.
   :type network_suid: int
   :param cytoscape_url: Cytoscape URL. Defaults to DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
   :type cytoscape_url: str, optional


.. py:function:: set_boundary_display(graph: networkx.Graph, show_boundary: bool, network_suid, cytoscape_url=DEFAULT_BASE_URL)

   Set the display of boundary nodes.

   :param graph: The graph to load boundary nodes from.
   :type graph: nx.Graph
   :param show_boundary: Whether to show boundary nodes. If boundary nodes were not loaded during initialization,
                           they can be loaded in here. If boundary nodes have already been loaded, they are hidden.
   :type show_boundary: bool
   :param network_suid: The network to update.
   :type network_suid: int
   :param cytoscape_url: Cytoscape URL. Defaults to DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
   :type cytoscape_url: str, optional


.. py:function:: color_nodes_by_column(data_column: str, network_suid: int, gradient=DEFAULT_GRADIENT, default_color=DEFAULT_NODE_COLOR, style=DEFAULT_STYLE, cytoscape_url=DEFAULT_BASE_URL)

   Color nodes by a column in the node table.

   :param data_column: The column to color by.
   :type data_column: str
   :param network_suid: The network to update.
   :type network_suid: int
   :param gradient: The gradient to use. Defaults to DEFAULT_GRADIENT ("#2B80EF", "#EF3B2C").
   :type gradient: (str, str), optional
   :param default_color: The default node color to use. Defaults to DEFAULT_NODE_COLOR ("#FEE391").
   :type default_color: str, optional
   :param style: The style to use. Defaults to DEFAULT_STYLE ("perturbationx-default").
   :type style: str, optional
   :param cytoscape_url: Cytoscape URL. Defaults to DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
   :type cytoscape_url: str, optional


.. py:function:: highlight_subgraph(nodes: list, edges: list, network_suid: int, highlight_factor=3, cytoscape_url=DEFAULT_BASE_URL)

   Highlight a subgraph.

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


.. py:function:: isolate_subgraph(graph: networkx.Graph, nodes: list, edges: list, network_suid: int, cytoscape_url=DEFAULT_BASE_URL)

   Isolate a subgraph.

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


.. py:function:: extract_subgraph(nodes: list, edges: list, network_suid: int, cytoscape_url=DEFAULT_BASE_URL)

   Extract a subgraph.

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


.. py:function:: clear_bypass(components: list, component_type: str, visual_property: str, network_suid: int, cytoscape_url=DEFAULT_BASE_URL)

   Clear a bypass.

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


