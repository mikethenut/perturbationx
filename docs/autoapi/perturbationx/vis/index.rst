:py:mod:`perturbationx.vis`
===========================

.. py:module:: perturbationx.vis


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   NPAResultDisplay/index.rst
   cytoscape/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   perturbationx.vis.NPAResultDisplay




.. py:class:: NPAResultDisplay(graph: networkx.Graph, results: perturbationx.result.NPAResult, network_style: str, network_suid: int, cytoscape_url=DEFAULT_BASE_URL)
   :no-index:


   Class to display results from NPA analysis.

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

   .. py:method:: reset_display(display_boundary: Optional[bool] = None, reset_color=False, reset_highlight=False, reset_visibility=False)

      Reset the display of the network.

      :param display_boundary: Whether to display the boundary of the network. Defaults to None, which does not
          change the current setting.
      :type display_boundary: bool, optional
      :param reset_color: Whether to reset the color of the nodes. Defaults to False.
      :type reset_color: bool, optional
      :param reset_highlight: Whether to reset the highlight of the nodes and edges. Defaults to False.
      :type reset_highlight: bool, optional
      :param reset_visibility: Whether to reset the visibility of the nodes and edges. Defaults to False.
      :type reset_visibility: bool, optional
      :raises CyError: If a CyREST error occurs.
      :return: SUID of the network.
      :rtype: int


   .. py:method:: color_nodes(dataset: str, attribute: str, gradient=DEFAULT_GRADIENT, default_color=DEFAULT_NODE_COLOR)

      Color nodes by a given attribute.

      :param dataset: The dataset to color nodes by.
      :type dataset: str
      :param attribute: The attribute to color nodes by.
      :type attribute: str
      :param gradient: The gradient to use. Defaults to DEFAULT_GRADIENT ("#2B80EF", "#EF3B2C").
      :type gradient: (str, str), optional
      :param default_color: The default color to use. Defaults to DEFAULT_NODE_COLOR ("#FEE391").
      :type default_color: str, optional
      :raises CyError: If a CyREST error occurs.
      :return: SUID of the network.
      :rtype: int


   .. py:method:: highlight_leading_nodes(dataset: str, cutoff=0.8, attr='contribution', abs_value=True, include_shortest_paths='none', path_length_tolerance=0, include_neighbors=0, neighborhood_type='union')

      Highlight leading nodes.

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
      :raises ValueError: If include_shortest_paths is not "directed", "undirected", or "none".
          If max_distance is less than 0 or neighborhood_type is not "union" or "intersection".
          If length_tolerance is not a number or is negative.
      :raises CyError: If a CyREST error occurs.
      :return: SUID of the network.
      :rtype: int


   .. py:method:: extract_leading_nodes(dataset: str, cutoff=0.8, attr='contribution', abs_value=True, inplace=True, include_shortest_paths='none', path_length_tolerance=0, include_neighbors=0, neighborhood_type='union')

      Extract leading nodes.

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
      :raises ValueError: If include_shortest_paths is not "directed", "undirected", or "none".
          If max_distance is less than 0 or neighborhood_type is not "union" or "intersection".
          If length_tolerance is not a number or is negative.
      :raises CyError: If a CyREST error occurs.
      :return: SUID of the network.
      :rtype: int


   .. py:method:: get_results()

      Retrieve the results object for this display.

      :return: The results object.
      :rtype: NPAResults



