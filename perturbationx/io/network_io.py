import csv
import warnings

import networkx as nx
import pandas

from perturbationx.resources.resources import DEFAULT_DATA_COLS


__all__ = ["read_dsv", "parse_pandas", "validate_nx_graph", "to_edge_list", "write_dsv"]


def read_dsv(filepath: str, default_edge_type="infer", delimiter='\t', header_cols=DEFAULT_DATA_COLS):
    """Read a DSV file and return a list of edges.

    :param filepath: Path to the DSV file.
    :type filepath: str
    :param default_edge_type: The default edge type to use if none is specified in the file. Defaults to "infer".
    :type default_edge_type: str, optional
    :param delimiter: The delimiter used in the DSV file. Defaults to "\t".
    :type delimiter: str, optional
    :param header_cols: The column names of the columns in the DSV file. Defaults to ["subject", "object", "relation",
        "type"].
    :type header_cols: list, optional
    :return: A list of edges.
    :rtype: list
    """
    if default_edge_type is None:
        default_edge_type = "infer"
    edge_dict = dict()

    with open(filepath, newline='') as input_file:
        dsv_file = csv.reader(input_file, delimiter=delimiter)
        header = [h.lower() for h in next(dsv_file)]

        # check if columns for subject, object and relation are present
        if header_cols is not None and set(header_cols[0:2]).issubset(header):
            # determine location of columns based on header
            src_idx, trg_idx, rel_idx, typ_idx = header.index(header_cols[0]), \
                header.index(header_cols[1]), header.index(header_cols[2]), None
            if len(header_cols) > 3 and header_cols[3] in header:
                typ_idx = header.index(header_cols[3])

        else:  # set to default values
            src_idx, trg_idx, rel_idx, typ_idx = 0, 1, 2, None
            if len(header) > 3:
                typ_idx = 3
            input_file.seek(0)

        # parse edges
        for line in dsv_file:
            if len(line) == 0:
                continue

            src, rel, trg = line[src_idx], line[rel_idx], line[trg_idx]
            typ = line[typ_idx] if typ_idx is not None else default_edge_type

            if (src, trg) in edge_dict:
                warnings.warn("Duplicate edge (%s, %s) found in file %s. "
                              "Only the first occurrence will be used." % (src, trg, filepath))
                continue

            edge_dict[(src, trg)] = (rel, typ)

    edge_list = [(src, trg, {"relation": rel, "type": typ})
                 for (src, trg), (rel, typ) in edge_dict.items()]
    return edge_list


def parse_pandas(df: pandas.DataFrame, default_edge_type="infer", header_cols=DEFAULT_DATA_COLS):
    """Parse a pandas DataFrame and return a list of edges.

    :param df: The DataFrame to parse.
    :type df: pandas.DataFrame
    :param default_edge_type: The default edge type to use if none is specified in the DataFrame. Defaults to "infer".
    :type default_edge_type: str, optional
    :param header_cols: The column names of the columns in the DataFrame. Defaults to ["subject", "object", "relation",
        "type"].
    :type header_cols: list, optional
    :return: A list of edges.
    :rtype: list
    """
    if header_cols is not None and set(header_cols[0:2]).issubset(df.columns):
        src_col, trg_col, rel_col, typ_col = header_cols[0], header_cols[1], header_cols[2], None
        if len(header_cols) > 3 and header_cols[3] in df.columns:
            typ_col = header_cols[3]
    else:
        src_col, trg_col, rel_col, typ_col = df.columns[0], df.columns[1], df.columns[2], None
        if len(df.columns) > 3:
            typ_col = df.columns[3]

    df_copy = df.copy()
    if typ_col is None:
        df_copy["type"] = default_edge_type
        typ_col = "type"
    else:
        df_copy[typ_col][df_copy[typ_col].isna()] = default_edge_type

    edge_list = df_copy[[src_col, trg_col, rel_col, typ_col]].to_records(index=False)
    edge_list = [(src, trg, {"relation": rel, "type": typ}) for src, trg, rel, typ in edge_list]
    return edge_list


def validate_nx_graph(graph: nx.DiGraph, allowed_edge_types: list):
    """Validate a networkx.DiGraph object for use with toponpa.

    :param graph: The graph to validate.
    :type graph: nx.DiGraph
    :param allowed_edge_types: List of allowed edge types. Edges with types not in this list will be replaced with
        "infer".
    :type allowed_edge_types: list
    :raises TypeError: If the graph is not a networkx.DiGraph.
    """
    if not isinstance(graph, nx.DiGraph):
        raise TypeError("Argument graph is not a networkx.Digraph.")

    to_remove = []
    for src, trg, data in graph.edges.data():
        if "relation" not in data:
            warnings.warn("Edge between %s and %s does not have a "
                          "relation specified and will be ignored." % (src, trg))
            to_remove.append((src, trg))
            continue

        if "type" not in data:
            graph[src][trg]["type"] = "infer"
        elif data["type"] not in allowed_edge_types:
            warnings.warn("Unknown type %s of edge %s will be replaced "
                          "with \"infer\"." % (data["type"], str((src, trg))))
            graph[src][trg]["type"] = "infer"

    for src, trg in to_remove:
        graph.remove_edge(src, trg)
        if graph.degree[src] == 0:
            graph.remove_node(src)
        if graph.degree[trg] == 0:
            graph.remove_node(trg)


def to_edge_list(graph: nx.DiGraph, edge_type="all", data_cols=DEFAULT_DATA_COLS):
    """Convert a networkx.DiGraph to a list of edges.

    :param graph: The graph to convert.
    :type graph: nx.DiGraph
    :param edge_type: List of edge types to include. If "all", all edges will be included. Defaults to "all".
    :type edge_type: list | str, optional
    :param data_cols: List of data columns to include. Defaults to ["subject", "object", "relation", "type"].
    :type data_cols: list, optional
    :return: A list of edges.
    :rtype: list
    """
    edge_list = nx.to_edgelist(graph)

    # ensure that data_cols contains valid values,
    d_cols = [c for c in data_cols if c in DEFAULT_DATA_COLS]
    if len(d_cols) < len(data_cols):
        warnings.warn("Data columns %s are invalid and will be ignored. "
                      "Please use 'subject', 'object', 'relation' and 'type'."
                      % str(tuple(c for c in data_cols if c not in d_cols)))

    # compute data index to ease line construction
    d_idx = [DEFAULT_DATA_COLS.index(c) for c in d_cols]

    # filter edge list based on edge type
    if edge_type != "all":
        edge_list = [e for e in edge_list if e[2]["type"] == edge_type]

    # format edge list
    edge_list = [(e[0], e[1], e[2]["relation"], e[2]["type"]) for e in edge_list]
    edge_list = [tuple([e[i] for i in d_idx]) for e in edge_list]

    return edge_list


def write_dsv(graph: nx.DiGraph, filepath: str, edge_type="all", delimiter='\t',
              data_cols=DEFAULT_DATA_COLS, header=None):
    """Write a networkx.DiGraph to a delimited file.

    :param graph: The graph to write.
    :type graph: nx.DiGraph
    :param filepath: The path to write the file to.
    :type filepath: str
    :param edge_type: List of edge types to include. If "all", all edges will be included. Defaults to "all".
    :type edge_type: list | str, optional
    :param delimiter: The delimiter to use in the DSV file. Defaults to "\t".
    :type delimiter: str, optional
    :param data_cols: List of data columns to include. Columns not in ["subject", "object", "relation", "type"] will
        be ignored. Defaults to ["subject", "object", "relation", "type"].
    :type data_cols: list, optional
    :param header: List of header values to use. Must be given in the same order as data_cols. Defaults to None.
    :type header: list, optional
    :raises ValueError: If the length of the header list does not match the length of the data_cols list.
    """
    if header is not None and len(header) != len(data_cols):
        raise ValueError("Please pass header values for all columns.")

    # construct edge list
    edge_list = to_edge_list(graph, edge_type=edge_type, data_cols=data_cols)

    # shorten header by skipping ignored columns
    d_cols = [c for c in data_cols if c in DEFAULT_DATA_COLS]
    if header is not None:
        header = [h for h, c in zip(header, data_cols) if c in d_cols]

    with open(filepath, 'w', newline='') as output_file:
        dsv_file = csv.writer(output_file, delimiter=delimiter)
        if header is not None:
            dsv_file.writerow(header)

        dsv_file.writerows(edge_list)
