import platform
import copy
import warnings
import logging
import json
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import py4cytoscape as p4c
from py4cytoscape.py4cytoscape_utils import DEFAULT_BASE_URL

from bnpa.resources.resources import DEFAULT_STYLE, DEFAULT_NODE_COLOR, DEFAULT_GRADIENT
from bnpa.vis.cytoscape import init_cytoscape, load_network_data, set_boundary_display


class NPAResult:
    def __init__(self, graph, datasets, global_info, node_info, distributions):
        self._graph = graph
        self._datasets = datasets
        self._global_info = global_info
        self._node_info = node_info
        self._distributions = distributions
        self._metadata = dict()
        self._cytoscape_suid = dict()

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

    def get_distribution(self, distribution):
        return copy.deepcopy(self._distributions[distribution])

    def plot_distribution(self, distribution, datasets=None):
        if datasets is None:
            datasets = self._datasets
        else:
            datasets = [d for d in datasets if d in self._datasets]

        if len(datasets) == 0:
            warnings.warn('Nothing to plot for distribution %s and datasets %s.' % (distribution, str(datasets)))
            return

        plt.clf()
        fig, ax = plt.subplots(nrows=len(datasets), figsize=(6, 3*len(datasets)))
        if len(datasets) == 1:
            ax = [ax]
        plt.suptitle(distribution)

        for idx, d in enumerate(datasets):
            distr = self._distributions[distribution][d]
            ax[idx].set_ylabel(d)

            sns.histplot(distr[0], ax=ax[idx], color='lightblue', stat='density', bins=25)
            sns.kdeplot(distr[0], ax=ax[idx], color='navy')
            if distr[1] is not None:
                ax[idx].axvline(x=distr[1], color='red')

        plt.show()
        return ax

    def display_network(self, display_boundary: Optional[bool] = False,
                        style=DEFAULT_STYLE, cytoscape_url=DEFAULT_BASE_URL):
        logging.getLogger().handlers.clear()  # Block logging to stdout

        # Initialize Cytoscape
        network_suid = self._cytoscape_suid[cytoscape_url] \
            if cytoscape_url in self._cytoscape_suid else None
        network_suid = init_cytoscape(
            self._graph, self._metadata['network_title'], self._metadata['network_collection'],
            display_boundary, network_suid, cytoscape_url
        )

        # Apply style
        if p4c.styles.get_current_style(network=network_suid, base_url=cytoscape_url) != style:
            p4c.styles.set_visual_style(style, network=network_suid, base_url=cytoscape_url)

        # If network is new, load node data
        if cytoscape_url not in self._cytoscape_suid \
                or network_suid != self._cytoscape_suid[cytoscape_url]:
            load_network_data(self._node_info.transpose(), "node", network_suid, cytoscape_url)
        # Otherwise adjust boundary display as requested
        elif display_boundary is not None:
            set_boundary_display(self._graph, display_boundary, network_suid, cytoscape_url)

        self._cytoscape_suid[cytoscape_url] = network_suid
        return network_suid

    def color_nodes(self, attribute, dataset, gradient=DEFAULT_GRADIENT, default_color=DEFAULT_NODE_COLOR,
                    style=DEFAULT_STYLE, cytoscape_url=DEFAULT_BASE_URL):
        self.display_network(display_boundary=None, style=style, cytoscape_url=cytoscape_url)

        data_column = dataset + ' ' + attribute
        data_range = [self._node_info.min(axis=1)[dataset][attribute],
                      self._node_info.max(axis=1)[dataset][attribute]]

        p4c.style_mappings.set_node_color_mapping(
            data_column, data_range, colors=gradient, default_color=default_color,
            style_name=style, network=self._cytoscape_suid[cytoscape_url], base_url=cytoscape_url
        )

    def highlight_leading_nodes(self, include_paths, include_neighbours, neighbourhood_type="union"):
        pass

    def extract_leading_nodes(self, include_paths, include_neighbours, neighbourhood_type="union"):
        pass

    def to_json(self, filepath, indent):
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
