import sys
import platform
import copy
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import py4cytoscape as p4c


class NPAResult:
    def __init__(self, graph, datasets, global_info, node_info, distributions):
        self._graph = graph
        self._datasets = datasets
        self._global_info = global_info
        self._node_info = node_info
        self._distributions = distributions

        # System metadata
        self._metadata = dict()
        self._metadata["python_implementation"] = platform.python_implementation()
        self._metadata["python_version"] = platform.python_version()
        self._metadata["python_executable"] = sys.executable
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

    def export(self):
        # TODO
        # allow export of results to a file, preferably in json (for now)
        # export metadata as well: Python version, Software version, file names (network and dataset),
        # datetime (https://en.wikipedia.org/wiki/FAIR_data)
        pass

    def display(self):
        # TODO
        p4c.networks.create_network_from_networkx(self._graph)
