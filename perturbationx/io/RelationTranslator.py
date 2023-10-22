
defaultMap = {
    "1": 1.,
    "increases": 1.,
    "->": 1.,
    "directlyIncreases": 1.,
    "=>": 1.,
    "-1": -1.,
    "decreases": -1.,
    "-|": -1.,
    "directlyDecreases": -1.,
    "=|": -1.
}


class RelationTranslator:
    """Class for translating relations to numeric values. By default, relations "1", "increases", "-\>",
    "directlyIncreases", and "=\>" are mapped to 1.0, while relations "-1", "decreases", "-\|", "directlyDecreases",
    and "=\|" are mapped to -1.0. Relations that cannot be mapped to a numeric value will be parsed as 0.0.

    :param mappings: Dictionary of additional relation to numeric value mappings. It extends and overrides the default
        mappings.
    :type mappings: dict, optional
    :param allow_numeric: If True, relations will be parsed as numeric values if they cannot be found in the mappings
        dictionary. Defaults to True.
    :type allow_numeric: bool, optional
    """

    def __init__(self, mappings=None, allow_numeric=True):
        """Construct a new RelationTranslator.
        """
        self._map = defaultMap.copy()
        if mappings is not None:
            for k, v in mappings.items():
                self._map[k] = float(v)
        self._allow_numeric = allow_numeric

    def add_mapping(self, relation, maps_to):
        """Add a new mapping from a relation to a numeric value.

        :param relation: The relation to map.
        :type relation: str
        :param maps_to: The numeric value to map to.
        :type maps_to: float
        """
        self._map[relation] = float(maps_to)

    def remove_mapping(self, relation):
        """Remove a mapping from a relation to a numeric value.

        :param relation: The relation to remove the mapping for.
        :type relation: str
        """
        del self._map[relation]

    def copy(self):
        """Create a copy of this RelationTranslator.

        :return: A copy of this RelationTranslator.
        :rtype: RelationTranslator
        """
        return RelationTranslator(self._map.copy(), self._allow_numeric)

    def translate(self, relation):
        """Translate a relation to a numeric value.

        :param relation: The relation to translate.
        :type relation: str
        :return: The numeric value that the relation maps to.
        :rtype: float
        """

        if relation in self._map:
            res = self._map[relation]
        elif self._allow_numeric:
            try:
                res = float(relation)
            except ValueError:
                res = 0.
        else:
            res = 0.

        return res
