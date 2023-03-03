import csv


def parse_dsv(input_path, delimiter='\t'):
    edges = []

    with open(input_path) as input_file:
        dsv_file = csv.reader(input_file, delimiter=delimiter)

        header = [h.lower() for h in next(dsv_file)]
        if {'subject', 'relation', 'object'}.issubset(header):
            src_idx, rel_idx, trg_idx = header.index('subject'), header.index('relation'), header.index('object')
        else:
            src_idx, rel_idx, trg_idx = 0, 1, 2
            input_file.seek(0)

        for line in dsv_file:
            src, rel, trg = line[src_idx], line[rel_idx], line[trg_idx]
            edges.append((src, rel, trg))

    return edges
