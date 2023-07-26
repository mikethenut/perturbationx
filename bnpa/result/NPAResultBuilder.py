import warnings

import numpy as np
import pandas as pd

from bnpa.result.NPAResult import NPAResult


class NPAResultBuilder:
    def __init__(self, graph, datasets):
        self._graph = graph
        self._datasets = datasets
        self._nodes = sorted([n for n in graph.nodes if graph.nodes[n]["type"] == "core"],
                             key=lambda n: graph.nodes[n]["idx"])

        self._global_attributes = {}  # One score per dataset
        self._node_attributes = {}  # N scores per dataset, where N is number of nodes
        self._distributions = {}  # K scores per dataset, where K is any positive integer

    @classmethod
    def new_builder(cls, datasets, nodes):
        return cls(datasets, nodes)

    def set_global_attributes(self, dataset_id, attributes, values):
        if dataset_id not in self._datasets:
            warnings.warn("Attributes for unknown dataset %s will be ignored." % dataset_id)
            return

        for attr, val in zip(attributes, values):
            if attr not in self._global_attributes:
                self._global_attributes[attr] = {d: np.nan for d in self._datasets}
            self._global_attributes[attr][dataset_id] = val

    def set_node_attributes(self, dataset_id, attributes, values):
        if dataset_id not in self._datasets:
            warnings.warn("Node attributes %s for unknown dataset %s will be ignored."
                          % (str(attributes), dataset_id))
            return

        for attr, val in zip(attributes, values):
            if len(val) != len(self._nodes):
                warnings.warn("Node attribute %s for dataset %s is of incorrect length "
                              "and will be ignored" % (attr, dataset_id))
                continue

            if attr not in self._node_attributes:
                self._node_attributes[attr] = {d: np.full(len(self._nodes), np.nan) for d in self._datasets}
            self._node_attributes[attr][dataset_id] = val

    def set_distribution(self, dataset_id, distribution, values, reference=None):
        if dataset_id not in self._datasets:
            warnings.warn("Distribution %s for unknown dataset %s will be ignored."
                          % (distribution, dataset_id))
            return

        if distribution not in self._distributions:
            self._distributions[distribution] = {d: ([], None) for d in self._datasets}
        self._distributions[distribution][dataset_id] = (values, reference)

    def build(self):
        # Create global info dataframe
        global_info = pd.DataFrame({
            attr: [self._global_attributes[attr][d] for d in self._datasets]
            for attr in self._global_attributes
        })
        if len(self._global_attributes.keys()) > 0:
            global_info = global_info.set_index([self._datasets])

        # Create node info dataframe
        node_indices = pd.MultiIndex.from_product(
            [self._datasets, self._node_attributes.keys()], names=["data", "attr"]
        )
        node_info = pd.DataFrame([
            self._node_attributes[attr][d]
            for d, attr in node_indices
        ], index=node_indices, columns=self._nodes)

        return NPAResult(self._graph, self._datasets, global_info, node_info, self._distributions)
