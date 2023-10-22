:py:mod:`perturbationx.io`
==========================

.. py:module:: perturbationx.io


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   RelationTranslator/index.rst
   network_io/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   perturbationx.io.RelationTranslator



Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.io.read_dsv
   perturbationx.io.parse_pandas
   perturbationx.io.validate_nx_graph
   perturbationx.io.to_edge_list
   perturbationx.io.write_dsv



.. py:function:: read_dsv(filepath: str, default_edge_type='infer', delimiter='\t', header_cols=DEFAULT_DATA_COLS)

   Read a DSV file and return a list of edges.

   :param filepath: Path to the DSV file.
   :type filepath: str
   :param default_edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
   :type default_edge_type: str, optional
   :param delimiter: The delimiter used in the DSV file. Defaults to " ".
   :type delimiter: str, optional
   :param header_cols: The column names of the columns in the DSV file. Defaults to ["subject", "object", "relation",
       "type"].
   :type header_cols: list, optional
   :return: A list of edges.
   :rtype: list


.. py:function:: parse_pandas(df: pandas.DataFrame, default_edge_type='infer', header_cols=DEFAULT_DATA_COLS)

   Parse a pandas DataFrame and return a list of edges.

   :param df: The DataFrame to parse.
   :type df: pandas.DataFrame
   :param default_edge_type: The default edge type to use if none is specified in the DataFrame. Defaults to "infer".
   :type default_edge_type: str, optional
   :param header_cols: The column names of the columns in the DataFrame. Defaults to ["subject", "object", "relation",
       "type"].
   :type header_cols: list, optional
   :return: A list of edges.
   :rtype: list


.. py:function:: validate_nx_graph(graph: networkx.DiGraph, allowed_edge_types: list)

   Validate a networkx.DiGraph object for use with toponpa.

   :param graph: The graph to validate.
   :type graph: nx.DiGraph
   :param allowed_edge_types: List of allowed edge types. Edges with types not in this list will be replaced with
       "infer".
   :type allowed_edge_types: list
   :raises TypeError: If the graph is not a networkx.DiGraph.


.. py:function:: to_edge_list(graph: networkx.DiGraph, edge_type='all', data_cols=DEFAULT_DATA_COLS)

   Convert a networkx.DiGraph to a list of edges.

   :param graph: The graph to convert.
   :type graph: nx.DiGraph
   :param edge_type: List of edge types to include. If "all", all edges will be included. Defaults to "all".
   :type edge_type: list | str, optional
   :param data_cols: List of data columns to include. Defaults to ["subject", "object", "relation", "type"].
   :type data_cols: list, optional
   :return: A list of edges.
   :rtype: list


.. py:function:: write_dsv(graph: networkx.DiGraph, filepath: str, edge_type='all', delimiter='\t', data_cols=DEFAULT_DATA_COLS, header=None)

   Write a networkx.DiGraph to a delimited file.

   :param graph: The graph to write.
   :type graph: nx.DiGraph
   :param filepath: The path to write the file to.
   :type filepath: str
   :param edge_type: List of edge types to include. If "all", all edges will be included. Defaults to "all".
   :type edge_type: list | str, optional
   :param delimiter: The delimiter to use in the DSV file. Defaults to "       ".
   :type delimiter: str, optional
   :param data_cols: List of data columns to include. Columns not in ["subject", "object", "relation", "type"] will
       be ignored. Defaults to ["subject", "object", "relation", "type"].
   :type data_cols: list, optional
   :param header: List of header values to use. Must be given in the same order as data_cols. Defaults to None.
   :type header: list, optional
   :raises ValueError: If the length of the header list does not match the length of the data_cols list.


.. py:class:: RelationTranslator(mappings=None, allow_numeric=True)
   :no-index:


   Class for translating relations to numeric values. By default, relations "1", "increases", "-\>",
   "directlyIncreases", and "=\>" are mapped to 1.0, while relations "-1", "decreases", "-\|", "directlyDecreases",
   and "=\|" are mapped to -1.0. Relations that cannot be mapped to a numeric value will be parsed as 0.0.

   :param mappings: Dictionary of additional relation to numeric value mappings. It extends and overrides the default
       mappings.
   :type mappings: dict, optional
   :param allow_numeric: If True, relations will be parsed as numeric values if they cannot be found in the mappings
       dictionary. Defaults to True.
   :type allow_numeric: bool, optional

   .. py:method:: add_mapping(relation, maps_to)

      Add a new mapping from a relation to a numeric value.

      :param relation: The relation to map.
      :type relation: str
      :param maps_to: The numeric value to map to.
      :type maps_to: float


   .. py:method:: remove_mapping(relation)

      Remove a mapping from a relation to a numeric value.

      :param relation: The relation to remove the mapping for.
      :type relation: str


   .. py:method:: copy()

      Create a copy of this RelationTranslator.

      :return: A copy of this RelationTranslator.
      :rtype: RelationTranslator


   .. py:method:: translate(relation)

      Translate a relation to a numeric value.

      :param relation: The relation to translate.
      :type relation: str
      :return: The numeric value that the relation maps to.
      :rtype: float



