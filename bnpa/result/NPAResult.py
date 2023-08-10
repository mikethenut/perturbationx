import platform
import copy
import warnings
import logging
from datetime import datetime
import json

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import py4cytoscape as p4c
from py4cytoscape.py4cytoscape_utils import DEFAULT_BASE_URL

from bnpa.resources.resources import DEFAULT_STYLE
from bnpa.vis.cytoscape import init_cytoscape, load_network_data
from bnpa.vis.shortest_paths import get_shortest_path_components
from bnpa.vis.neighbors import get_neighborhood
from bnpa.vis.NPAResultDisplay import NPAResultDisplay


class NPAResult:
    def __init__(self, graph, datasets, global_info, node_info, distributions):
        self._graph = graph
        self._datasets = datasets
        self._global_info = global_info
        self._node_info = node_info
        self._distributions = distributions
        self._metadata = dict()

        # System metadata
        self._metadata["datetime_utc"] = datetime.utcnow().isoformat()
        self._metadata["python_implementation"] = platform.python_implementation()
        self._metadata["python_version"] = platform.python_version()
        self._metadata["system_name"] = platform.system()
        self._metadata["system_release"] = platform.release()
        self._metadata["system_version"] = platform.version()

        # Network metadata
        for k, v in graph.graph.items():
            self._metadata["network_" + k] = v

        # TODO: Add metadata for package and dependency versions

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def datasets(self):
        return copy.deepcopy(self._datasets)

    def node_attributes(self):
        return self._node_info.index.unique('attr').tolist()

    def distributions(self):
        return list(self._distributions.keys())

    def global_info(self):
        return self._global_info.copy()

    def node_info(self, accessor):
        level = "data" if accessor in self._datasets else "attr"
        return self._node_info.xs(accessor, level=level).transpose(copy=True)

    def get_distribution(self, distribution, dataset):
        return copy.deepcopy(self._distributions[distribution][dataset][0])

    def plot_distribution(self, distribution, datasets=None, show=True):
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

    def get_leading_nodes(self, dataset, cutoff=0.8):
        contributions = self.node_info("contribution")[dataset].sort_values(ascending=False)
        cumulative_contributions = contributions.cumsum() / contributions.sum()
        max_idx = 0
        for contr in cumulative_contributions:
            max_idx += 1
            if contr > cutoff:
                break
        return set(cumulative_contributions.index[:max_idx].tolist())

    def get_node_subgraph(self, nodes, include_shortest_paths="none", path_length_tolerance=0,
                          include_neighbors=0, neighborhood_type="union"):
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
            case "undirected" | "all":
                path_nodes, path_edges = get_shortest_path_components(
                    graph, nodes, directed=False, length_tolerance=path_length_tolerance
                )
            case "none" | None:
                path_nodes, path_edges = set(), set()
            case _:
                raise ValueError("Argument neighborhood_type must be 'union' or 'intersection'.")

        # Find neighborhood
        neigh_nodes, neigh_edges = get_neighborhood(graph, nodes, include_neighbors, neighborhood_type)

        nodes = list(nodes.union(path_nodes).union(neigh_nodes))
        edges = list(path_edges.union(neigh_edges))
        return nodes, edges

    def display_network(self, display_boundary=False, style=DEFAULT_STYLE, cytoscape_url=DEFAULT_BASE_URL):
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

    def to_json(self, filepath, indent=4):
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

        with open(filepath, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
