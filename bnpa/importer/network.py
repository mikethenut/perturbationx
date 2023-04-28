import csv


def parse_dsv(input_path, delimiter='\t', default_edge_type='infer'):
    edge_list = []

    with open(input_path) as input_file:
        dsv_file = csv.reader(input_file, delimiter=delimiter)

        header = [h.lower() for h in next(dsv_file)]
        if {'subject', 'relation', 'object'}.issubset(header):
            src_idx, rel_idx, trg_idx = header.index('subject'), header.index('relation'), header.index('object')
            typ_idx = header.index('type') if 'type' in header else None
        else:
            src_idx, rel_idx, trg_idx, typ_idx = 0, 1, 2, None
            input_file.seek(0)

        for line in dsv_file:
            src, rel, trg = line[src_idx], line[rel_idx], line[trg_idx]
            typ = line[typ_idx] if typ_idx is not None else default_edge_type
            edge_list.append((src, trg, {'relation': rel, 'type': typ}))

    return edge_list
