from .importer.dataset import import_dataset_npa


class Dataset:
    def __init__(self, input_format='NPA', input_path=None):
        match input_format:
            case 'NPA':
                if input_path is None or not isinstance(input_path, str):
                    raise TypeError("Argument file_path is not a file path.")

                self.node_count, self.node_name, self.t_statistic, self.fold_change = \
                    import_dataset_npa(input_path)

            case '_':
                raise ValueError("Input format %s is not supported." % input_format)
