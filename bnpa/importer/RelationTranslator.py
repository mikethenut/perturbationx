
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
    def __init__(self, mappings=None, allow_numeric=True):
        self._map = defaultMap.copy()
        if mappings is not None:
            for k, v in mappings.items():
                self._map[k] = float(v)
        self._allow_numeric = allow_numeric

    def add_mapping(self, relation, maps_to):
        self._map[relation] = float(maps_to)

    def remove_mapping(self, relation):
        del self._map[relation]

    def translate(self, relation):
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
