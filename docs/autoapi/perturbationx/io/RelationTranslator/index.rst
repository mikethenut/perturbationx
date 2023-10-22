:py:mod:`perturbationx.io.RelationTranslator`
=============================================

.. py:module:: perturbationx.io.RelationTranslator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   perturbationx.io.RelationTranslator.RelationTranslator

   

.. py:class:: RelationTranslator(mappings=None, allow_numeric=True)


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



