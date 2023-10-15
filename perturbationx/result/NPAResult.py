import platform
import copy
import warnings
import logging
from datetime import datetime
import json
from importlib.metadata import version, PackageNotFoundError

import matplotlib.pyplot as plt
import pandas
import seaborn as sns
import networkx as nx
import py4cytoscape as p4c
from py4cytoscape.py4cytoscape_utils import DEFAULT_BASE_URL

from perturbationx.resources import DEFAULT_STYLE
from perturbationx.vis import NPAResultDisplay
from perturbationx.vis.cytoscape import init_cytoscape, load_network_data
from perturbationx.util import get_shortest_path_components, get_neighborhood_components


class NPAResult:
    """Class for storing and accessing the results of a Network Perturbation Analysis (NPA). It is recommended
    to build an NPAResult object using NPAResultBuilder to ensure correct formatting. Metadata is prefixed with
    "network_" to avoid conflicts, unless the metadata key already starts with "network" or "dataset". By default,
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
    """
    def __init__(self, graph: nx.DiGraph, datasets: list, global_info: pandas.DataFrame, node_info: pandas.DataFrame,
                 distributions: dict, metadata=None):
        """Construct a new NPAResult.
        """

        self._graph = graph
        self._datasets = datasets
        self._global_info = global_info
        self._node_info = node_info
        self._distributions = distributions
        self._metadata = metadata if metadata is not None else dict()

        # System metadata
        self._metadata["datetime_utc"] = datetime.utcnow().isoformat()
        self._metadata["python_implementation"] = platform.python_implementation()
        self._metadata["python_version"] = platform.python_version()
        self._metadata["system_name"] = platform.system()
        self._metadata["system_release"] = platform.release()
        self._metadata["system_version"] = platform.version()

        # Network metadata
        for k, v in graph.graph.items():
            if k.startswith("dataset") or k.startswith("network"):
                self._metadata[k] = v
            self._metadata["network_" + k] = v

        if "network_title" not in self._metadata:
            self._metadata["network_title"] = "Untitled network"
        if "network_collection" not in self._metadata:
            self._metadata["network_collection"] = "Untitled collection"

        # Library metadata
        try:
            self._metadata["perturbationx_version"] = version("perturbationx")
        except PackageNotFoundError:
            self._metadata["perturbationx_version"] = "dev"

        self._metadata["numpy_version"] = version("numpy")
        self._metadata["networkx_version"] = version("networkx")
        self._metadata["pandas_version"] = version("pandas")
        self._metadata["scipy_version"] = version("scipy")
        self._metadata["matplotlib_version"] = version("matplotlib")
        self._metadata["seaborn_version"] = version("seaborn")
        self._metadata["py4cytoscape_version"] = version("py4cytoscape")

    def metadata(self):
        """Get the metadata for this result.

        :return: The metadata for this result.
        :rtype: dict
        """
        return copy.deepcopy(self._metadata)

    def datasets(self):
        """Get the datasets used for this result.

        :return: The datasets used for this result.
        :rtype: list
        """
        return copy.deepcopy(self._datasets)

    def node_attributes(self):
        """Get the node attributes for this result.

        :return: The node attributes for this result.
        :rtype: list
        """
        return self._node_info.index.unique('attr').tolist()

    def distributions(self):
        """Get the distributions for this result.

        :return: The distributions for this result.
        :rtype: list
        """
        return list(self._distributions.keys())

    def global_info(self):
        """Get the global information for this result.

        :return: The global information for this result.
        :rtype: pandas.DataFrame
        """
        return self._global_info.copy()

    def node_info(self, accessor: str):
        """Get the node information for this result.

        :param accessor: The dataset or node attribute to get the information for.
        :type accessor: str
        :return: The node information for this result.
        :rtype: pandas.DataFrame
        """
        level = "data" if accessor in self._datasets else "attr"
        return self._node_info.xs(accessor, level=level).transpose(copy=True)

    def get_distribution(self, distribution: str, dataset: str, include_reference=False):
        """Get the distribution for a permutation.

        :param distribution: The permutation to get the distribution for.
        :type distribution: str
        :param dataset: The dataset to get the distribution for.
        :type dataset: str
        :param include_reference: If True, the reference value will be included in the distribution. Defaults to False.
        :type include_reference: bool, optional
        :return: The distribution for the permutation. If include_reference is True, a tuple of the distribution and
            the reference value will be returned.
        :rtype: list | (list, float)
        """
        value_distribution = copy.deepcopy(self._distributions[distribution][dataset][0])
        if include_reference:
            return value_distribution, self._distributions[distribution][dataset][1]
        else:
            return value_distribution

    def plot_distribution(self, distribution: str, datasets=None, show=True):
        """Plot the distribution for a permutation.

        :param distribution: The permutation to plot the distribution for.
        :type distribution: str
        :param datasets: The datasets to plot the distribution for. If None, all datasets will be plotted.
        :type datasets: list, optional
        :param show: If True, the plot will be shown. Defaults to True.
        :type show: bool, optional
        :return: The axes of the plot.
        :rtype: matplotlib.axes.Axes
        """
        if datasets is None:
            datasets = self._datasets
        else:
            datasets = [d for d in datasets if d in self._datasets]

        if len(datasets) == 0:
            warnings.warn('Nothing to plot for distribution %s and datasets %s.' % (distribution, str(datasets)))
            return

        plt.clf()
        fig, ax = plt.subplots(nrows=len(datasets), figsize=(8, 4*len(datasets)))
        if len(datasets) == 1:
            ax = [ax]
        plt.suptitle("Permutation '%s'" % distribution)

        for idx, d in enumerate(datasets):
            distr = self._distributions[distribution][d]
            ax[idx].set_ylabel(d)

            sns.histplot(distr[0], ax=ax[idx], bins=25, stat="density")
            sns.kdeplot(distr[0], ax=ax[idx], color="tab:blue")
            if distr[1] is not None:
                ax[idx].axvline(x=distr[1], color="tab:red")

        plt.tight_layout(rect=[0.01, 0.01, 1, 0.97])
        if show:
            plt.show()
        return ax

    def get_leading_nodes(self, dataset: str, cutoff=0.8, attr="contribution", abs_value=True):
        """Get the leading nodes for a dataset. The leading nodes are the nodes that contribute the most
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
        """
        contributions = self.node_info(attr)[dataset]
        if abs_value:
            contributions = contributions.abs()
        contributions = contributions.sort_values(ascending=False)

        cumulative_contributions = contributions.cumsum() / contributions.sum()
        max_idx = 0
        for contr in cumulative_contributions:
            max_idx += 1
            if contr > cutoff:
                break
        return set(cumulative_contributions.index[:max_idx].tolist())

    def get_node_subgraph(self, nodes, include_shortest_paths="none", path_length_tolerance=0,
                          include_neighbors=0, neighborhood_type="union"):
        """Get the subgraph for a set of nodes. The subgraph can include the shortest paths between the nodes,
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
        :return: The nodes and edges in the subgraph. They are returned as a pair of lists.
        :rtype: (list, list)
        """
        # Remove boundary nodes
        graph = self._graph.copy()
        boundary_nodes = [n for n in graph.nodes if graph.nodes[n]["type"] == "boundary"]
        graph.remove_nodes_from(boundary_nodes)

        # Find paths
        match include_shortest_paths:
            case "directed":
                path_nodes, path_edges = get_shortest_path_components(
                    graph, nodes, directed=True, length_tolerance=path_length_tolerance
                )
            case "undirected":
                path_nodes, path_edges = get_shortest_path_components(
                    graph, nodes, directed=False, length_tolerance=path_length_tolerance
                )
            case "none" | None:
                path_nodes, path_edges = set(), set()
            case _:
                raise ValueError("Argument include_shortest_paths must be "
                                 "'directed', 'undirected', or 'none'.")

        # Find neighborhood
        neigh_nodes, neigh_edges = get_neighborhood_components(graph, nodes, include_neighbors, neighborhood_type)

        nodes = list(nodes.union(path_nodes).union(neigh_nodes))
        edges = list(path_edges.union(neigh_edges))
        return nodes, edges

    def display_network(self, display_boundary=False, style=DEFAULT_STYLE, cytoscape_url=DEFAULT_BASE_URL):
        """Display the network in Cytoscape.

        :param display_boundary: If True, boundary nodes will be displayed. Defaults to False.
        :type display_boundary: bool, optional
        :param style: The style to apply to the network. Defaults to DEFAULT_STYLE ("perturbationx-default").
        :type style: str, optional
        :param cytoscape_url: The URL of the Cytoscape instance to display the network in. Defaults to
            DEFAULT_BASE_URL in py4cytoscape (http://127.0.0.1:1234/v1).
        :type cytoscape_url: str, optional
        :return: The display object.
        :rtype: NPAResultDisplay
        """
        logging.getLogger().handlers.clear()  # Block logging to stdout

        # Initialize Cytoscape
        network_suid = init_cytoscape(
            self._graph, self._metadata['network_title'], self._metadata['network_collection'],
            display_boundary, network_suid=None, cytoscape_url=cytoscape_url
        )

        # Apply style
        p4c.styles.set_visual_style(style, network=network_suid, base_url=cytoscape_url)

        # Load node data
        load_network_data(self._node_info.transpose(), "node", network_suid, cytoscape_url)

        return NPAResultDisplay(self._graph, self, style, network_suid, cytoscape_url)

    def to_networkx(self):
        """Retrieve the NetworkX graph for this result.

        :return: The NetworkX graph.
        :rtype: networkx.DiGraph
        """
        graph_copy = self._graph.copy()
        graph_copy.graph.update(self._metadata)
        for n in graph_copy.nodes:
            graph_copy.nodes[n].update(self._node_info.loc[(n, "data")].to_dict())

        node_dataframe = self._node_info.transpose(copy=True)
        # Rename MultiIndex columns
        node_dataframe.columns = [col if isinstance(col, str) else ' '.join(col)
                                  for col in node_dataframe.columns]
        node_attr = node_dataframe.to_dict('index')
        nx.set_node_attributes(graph_copy, node_attr)

        return graph_copy

    def to_dict(self):
        """Convert this result to a dictionary.

        :return: The result as a dictionary. Top-level keys are "metadata" and dataset names. For each dataset, the
            top-level keys are "global_info", "node_info", and "distributions".
        :rtype: dict
        """
        data = dict()
        data["metadata"] = self._metadata.copy()

        for d in self.datasets():
            data[d] = dict()

            data[d]["global_info"] = dict()
            for column in self._global_info.columns:
                data[d]["global_info"][column] = self._global_info.loc[d, column]

            data[d]["node_info"] = dict()
            for attr in self.node_attributes():
                data[d]["node_info"][attr] = {n: self._node_info.loc[(d, attr), n] for n in self._node_info.columns}

            data[d]["distributions"] = dict()
            for distr in self.distributions():
                data[d]["distributions"][distr] = self._distributions[distr][d][0]

        return data

    def to_json(self, filepath: str, indent=4):
        """Save this result to a JSON file. The format is the same as the output of to_dict().

        :param filepath: The path to save the result to.
        :type filepath: str
        :param indent: The indentation to use. Defaults to 4.
        :type indent: int, optional
        """
        data = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
