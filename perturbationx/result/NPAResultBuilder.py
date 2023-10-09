import warnings

import networkx as nx
import numpy as np
import pandas as pd

from .NPAResult import NPAResult


class NPAResultBuilder:
    """Class for building NPAResult objects. Node attributes can be only passed for nodes in the graph with type "core".
    Each core node should have an attribute "idx" that is a unique integer index for that node. Node attributes should
    be passed as a list of values that is ordered by the node indices.

    :param graph: The graph used to generate the result.
    :type graph: networkx.DiGraph
    :param datasets: The datasets used to generate the result.
    :type datasets: list
    """
    def __init__(self, graph: nx.DiGraph, datasets: list):
        """Construct a new NPAResultBuilder object.
        """
        self._graph = graph
        self._datasets = datasets
        self._nodes = sorted([n for n in graph.nodes if graph.nodes[n]["type"] == "core"],
                             key=lambda n: graph.nodes[n]["idx"])

        self._global_attributes = {}  # One score per dataset
        self._node_attributes = {}  # N scores per dataset, where N is number of nodes
        self._distributions = {}  # K scores per dataset, where K is any positive integer

    @classmethod
    def new_builder(cls, graph: nx.DiGraph, datasets: list):
        """Construct a new NPAResultBuilder object.
        """
        return cls(graph, datasets)

    def set_global_attributes(self, dataset_id: str, attributes: list, values: list):
        """Set global attributes for a dataset. Attributes and values should be ordered in the same way.

        :param dataset_id: Dataset to set attributes for.
        :type dataset_id: str
        :param attributes: List of attribute names.
        :type attributes: list
        :param values: List of attribute values.
        :type values: list
        """
        if dataset_id not in self._datasets:
            warnings.warn("Attributes for unknown dataset %s will be ignored." % dataset_id)
            return

        for attr, val in zip(attributes, values):
            if attr not in self._global_attributes:
                self._global_attributes[attr] = {d: np.nan for d in self._datasets}
            self._global_attributes[attr][dataset_id] = val

    def set_node_attributes(self, dataset_id: str, attributes: list, values: list):
        """Set node attributes for a dataset. Attributes and values should be ordered in the same way.
        Values should be passed as a nested list that is ordered by the node indices.

        :param dataset_id: Dataset to set attributes for.
        :type dataset_id: str
        :param attributes: List of attribute names.
        :type attributes: list
        :param values: List of attribute values.
        :type values: list
        """
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

    def set_distribution(self, dataset_id: str, distribution: str, values: list, reference=None):
        """Set a distribution for a dataset.

        :param dataset_id: Dataset to set distribution for.
        :type dataset_id: str
        :param distribution: Name of distribution.
        :type distribution: str
        :param values: List of values.
        :type values: list
        :param reference: Reference value for distribution. Defaults to None.
        :type reference: float, optional
        """
        if dataset_id not in self._datasets:
            warnings.warn("Distribution %s for unknown dataset %s will be ignored."
                          % (distribution, dataset_id))
            return

        if distribution not in self._distributions:
            self._distributions[distribution] = {d: ([], None) for d in self._datasets}
        self._distributions[distribution][dataset_id] = (values, reference)

    def build(self, metadata=None):
        """Construct an NPAResult object.

        :param metadata: Metadata to include in the result. Defaults to None.
        :type metadata: dict, optional
        :return: NPAResult object.
        :rtype: NPAResult
        """
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

        return NPAResult(self._graph, self._datasets, global_info,
                         node_info, self._distributions, metadata)
