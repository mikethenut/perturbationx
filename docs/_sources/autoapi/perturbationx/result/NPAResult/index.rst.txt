:py:mod:`perturbationx.result.NPAResult`
========================================

.. py:module:: perturbationx.result.NPAResult


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   perturbationx.result.NPAResult.NPAResult




.. py:class:: NPAResult(graph: networkx.DiGraph, datasets: list, global_info: pandas.DataFrame, node_info: pandas.DataFrame, distributions: dict, metadata=None)


   Class for storing and accessing the results of a Network Perturbation Analysis (NPA). It is recommended
   to build an NPAResult object using NPAResultBuilder to ensure correct formatting. Metadata is prefixed with
   "network\_" to avoid conflicts, unless the metadata key already starts with "network" or "dataset". By default,
   the following metadata is added: datetime_utc, python_implementation, python_version, system_name,
   system_release, system_version, network_title, network_collection, perturbationx_version, numpy_version,
   networkx_version, pandas_version, scipy_version, matplotlib_version, seaborn_version, and py4cytoscape_version.


   :param graph: The network graph.
   :type graph: networkx.DiGraph
   :param datasets: The datasets used for the analysis.
   :type datasets: list
   :param global_info: The global information for each dataset.
   :type global_info: pandas.DataFrame
   :param node_info: The node information for each dataset.
   :type node_info: pandas.DataFrame
   :param distributions: The distributions for each permutation.
   :type distributions: dict
   :param metadata: Additional metadata to store with the result.
   :type metadata: dict, optional

   .. py:method:: metadata()

      Get the metadata for this result.

      :return: The metadata for this result.
      :rtype: dict


   .. py:method:: datasets()

      Get the datasets used for this result.

      :return: The datasets used for this result.
      :rtype: list


   .. py:method:: node_attributes()

      Get the node attributes for this result.

      :return: The node attributes for this result.
      :rtype: list


   .. py:method:: distributions()

      Get the distributions for this result.

      :return: The distributions for this result.
      :rtype: list


   .. py:method:: global_info()

      Get the global information for this result.

      :return: The global information for this result.
      :rtype: pandas.DataFrame


   .. py:method:: node_info(accessor: str)

      Get the node information for this result.

      :param accessor: The dataset or node attribute to get the information for.
      :type accessor: str
      :return: The node information for this result.
      :rtype: pandas.DataFrame


   .. py:method:: get_distribution(distribution: str, dataset: str, include_reference=False)

      Get the distribution for a permutation.

      :param distribution: The permutation to get the distribution for.
      :type distribution: str
      :param dataset: The dataset to get the distribution for.
      :type dataset: str
      :param include_reference: If True, the reference value will be included in the distribution. Defaults to False.
      :type include_reference: bool, optional
      :return: The distribution for the permutation. If include_reference is True, a tuple of the distribution and
          the reference value will be returned.
      :rtype: list | (list, float)


   .. py:method:: plot_distribution(distribution: str, datasets=None, show=True)

      Plot the distribution for a permutation.

      :param distribution: The permutation to plot the distribution for.
      :type distribution: str
      :param datasets: The datasets to plot the distribution for. If None, all datasets will be plotted.
      :type datasets: list, optional
      :param show: If True, the plot will be shown. Defaults to True.
      :type show: bool, optional
      :return: The axes of the plot.
      :rtype: matplotlib.axes.Axes


   .. py:method:: get_leading_nodes(dataset: str, cutoff=0.8, attr='contribution', abs_value=True)

      Get the leading nodes for a dataset. The leading nodes are the nodes that contribute the most
      to a selected attribute, up to a certain cutoff.

      :param dataset: The dataset to get the leading nodes for.
      :type dataset: str
      :param cutoff: The cutoff for the cumulative distribution. Defaults to 0.8.
      :type cutoff: float, optional
      :param attr: The node attribute to get the leading nodes for. Defaults to "contribution".
      :type attr: str, optional
      :param abs_value: If True, the absolute value of the attribute will be used. Defaults to True.
      :type abs_value: bool, optional
      :return: The leading nodes for the dataset.
      :rtype: set


   .. py:method:: get_node_subgraph(nodes, include_shortest_paths='none', path_length_tolerance=0, include_neighbors=0, neighborhood_type='union')

      Get the subgraph for a set of nodes. The subgraph can include the shortest paths between the nodes,
      the neighborhood of the nodes, or both.

      :param nodes: The nodes to get the subgraph for.
      :type nodes: set
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
      :return: The nodes and edges in the subgraph. They are returned as a pair of lists.
      :rtype: (list, list)


   .. py:method:: display_network(display_boundary=False, style=DEFAULT_STYLE, cytoscape_url=DEFAULT_BASE_URL)

      Display the network in Cytoscape.

      :param display_boundary: If True, boundary nodes will be displayed. Defaults to False.
      :type display_boundary: bool, optional
      :param style: The style to apply to the network. Defaults to DEFAULT_STYLE ("perturbationx-default").
      :type style: str, optional
      :param cytoscape_url: The URL of the Cytoscape instance to display the network in. Defaults to
          DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
      :type cytoscape_url: str, optional
      :return: The display object.
      :rtype: perturbationx.NPAResultDisplay


   .. py:method:: to_networkx()

      Retrieve the NetworkX graph for this result.

      :return: The NetworkX graph.
      :rtype: networkx.DiGraph


   .. py:method:: to_dict()

      Convert this result to a dictionary.

      :return: The result as a dictionary. Top-level keys are "metadata" and dataset names. For each dataset, the
          top-level keys are "global_info", "node_info", and "distributions".
      :rtype: dict


   .. py:method:: to_json(filepath: str, indent=4)

      Save this result to a JSON file. The format is the same as the output of to_dict().

      :param filepath: The path to save the result to.
      :type filepath: str
      :param indent: The indentation to use. Defaults to 4.
      :type indent: int, optional



