:py:mod:`perturbationx.result.NPAResultBuilder`
===============================================

.. py:module:: perturbationx.result.NPAResultBuilder


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   perturbationx.result.NPAResultBuilder.NPAResultBuilder




.. py:class:: NPAResultBuilder(graph: networkx.DiGraph, datasets: list)


   Class for building NPAResult objects. Node attributes can be only passed for nodes in the graph with type "core".
   Each core node should have an attribute "idx" that is a unique integer index for that node. Node attributes should
   be passed as a list of values that is ordered by the node indices.

   :param graph: The graph used to generate the result.
   :type graph: networkx.DiGraph
   :param datasets: The datasets used to generate the result.
   :type datasets: list

   .. py:method:: new_builder(graph: networkx.DiGraph, datasets: list)
      :classmethod:

      Construct a new NPAResultBuilder object.
              


   .. py:method:: set_global_attributes(dataset_id: str, attributes: list, values: list)

      Set global attributes for a dataset. Attributes and values should be ordered in the same way.

      :param dataset_id: Dataset to set attributes for.
      :type dataset_id: str
      :param attributes: List of attribute names.
      :type attributes: list
      :param values: List of attribute values.
      :type values: list


   .. py:method:: set_node_attributes(dataset_id: str, attributes: list, values: list)

      Set node attributes for a dataset. Attributes and values should be ordered in the same way.
      Values should be passed as a nested list that is ordered by the node indices.

      :param dataset_id: Dataset to set attributes for.
      :type dataset_id: str
      :param attributes: List of attribute names.
      :type attributes: list
      :param values: List of attribute values.
      :type values: list


   .. py:method:: set_distribution(dataset_id: str, distribution: str, values: list, reference=None)

      Set a distribution for a dataset.

      :param dataset_id: Dataset to set distribution for.
      :type dataset_id: str
      :param distribution: Name of distribution.
      :type distribution: str
      :param values: List of values.
      :type values: list
      :param reference: Reference value for distribution. Defaults to None.
      :type reference: float, optional


   .. py:method:: build(metadata=None)

      Construct an NPAResult object.

      :param metadata: Metadata to include in the result. Defaults to None.
      :type metadata: dict, optional
      :return: NPAResult object.
      :rtype: perturbationx.NPAResult



