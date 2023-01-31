from .io.dataset import import_dataset_npa


class ContrastDataset:
    def __init__(self, input_format='NPA', input_path=None):
        match input_format:
            case 'NPA':
                if input_path is None or not isinstance(input_path, str):
                    raise TypeError("Argument file_path is not a file path.")

                self._node_count, self._node_name, self._t_statistic, self._fold_change = \
                    import_dataset_npa(input_path)

            case '_':
                raise ValueError("Input format %s is not supported." % input_format)

    def get_node_count(self):
        return self._node_count

    def get_node_names(self):
        return self._node_name

    def get_t_statistics(self):
        return self._t_statistic

    def get_fold_changes(self):
        return self._fold_change
