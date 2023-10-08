import os
import logging
import time

import numpy as np
import pandas as pd

from perturbationx.resources import DEFAULT_LOGGING_KWARGS
from perturbationx import CausalNetwork


def example_run(causalbionet, datasets, sparse=True,
                permutations=("o", "k1", "k2"), full_core_permutation=True):
    results = causalbionet.toponpa(
        datasets, permutations=permutations,
        opposing_value_pruning_mode=None,
        exact_boundary_outdegree=False,
        full_core_permutation=full_core_permutation,
        verbose=True, sparse=sparse
    )

    results.plot_distribution("k2")
    results.plot_distribution("k1")
    results.plot_distribution("o")
    results.to_json("results.json")
    # print(results.metadata())

    # print(results.node_info("CS (2m) + Sham (3m)")["coefficient"].sort_values(ascending=False))
    # result_display = results.display_network()
    # result_display.highlight_leading_nodes("CS (2m) + Sham (3m)", include_shortest_paths="directed",
    #                                       path_length_tolerance=0.1)
    # results.reset_display()
    # result_display.highlight_leading_nodes(dataset="CS (2m) + Sham (3m)", include_paths="all",
    #                                directed_paths=False)
    # result_display.extract_leading_nodes(dataset="CS (2m) + Sham (3m)", include_paths="all",
    #                              directed_paths=True, inplace=True)
    # result_display.color_nodes("coefficient", "CS (2m) + Sham (3m)")

    output_file = "test_results.txt"
    with open(output_file, "w") as f:
        f.write(str(results.metadata()) + "\n")
        f.write(results.global_info().to_string() + "\n")
        for attr in results.node_attributes():
            f.write(results.node_info(attr).to_string() + "\n")
        #for distr in results.distributions():
        #    results.plot_distribution(distr)


def test_rewiring(causalbionet, datasets):
    nodes = ["p(MGI:Bcl2)", "p(MGI:Bcl2a1b)", "act(p(MGI:Akt1))", "p(SFAM:\"AKT Family\")", "act(p(MGI:Map3k5))",
         "act(p(MGI:Nkx3-1))", "bp(MESHPP:Apoptosis)", "act(p(MGI:Birc5))"]

    results = causalbionet.wire_edges(
        nodes, 5, datasets, 3, ("1.", "-1.")
    )

    results2 = causalbionet.rewire_edges(
        nodes, 5, datasets, p_rate=0.4
    )

    edges = causalbionet.edges(typ="core", data=False)
    rng = np.random.default_rng(0)
    for edge in edges:
        if edge[0] in nodes and edge[1] in nodes:
            causalbionet.modify_edge(edge[0], edge[1], confidence=rng.uniform(0.85, 1.))

    results3 = causalbionet.rewire_edges(
        nodes, 5, datasets, p_rate="confidence"
    )

    for dataset in datasets:
        best_idx = -1
        best_score = 0.
        for idx, r in enumerate(results):
            if r[1][dataset] > best_score:
                best_score = r[1][dataset]
                best_idx = idx

        best_idx_2 = -1
        best_score = 0.
        for idx, r in enumerate(results2):
            if r[1][dataset] > best_score:
                best_score = r[1][dataset]
                best_idx_2 = idx

        best_idx_3 = -1
        best_score = 0.
        for idx, r in enumerate(results3):
            if r[1][dataset] > best_score:
                best_score = r[1][dataset]
                best_idx_3 = idx

        print(dataset, best_idx, results[best_idx][1][dataset],
              best_idx_2, results2[best_idx_2][1][dataset],
              best_idx_3, results3[best_idx_3][1][dataset])

        new_cbn = causalbionet.copy()
        new_cbn.modify_network(results[best_idx][0])

        new_cbn2 = causalbionet.copy()
        new_cbn2.modify_network(results2[best_idx_2][0])

        new_cbn3 = causalbionet.copy()
        new_cbn3.modify_network(results3[best_idx_3][0])


def test_import():
    import_start = time.time()
    edge_df = pd.read_table("data/Arabidopsis/ckn-directed.tsv", header=None)
    edge_df.columns = ["source", "target"]
    edge_df["relation"] = "1."
    my_cbn = CausalNetwork.from_pandas(edge_df)
    import_end = time.time()
    print("Import time: ", import_end - import_start)
    print(my_cbn.number_of_nodes())
    my_cbn.infer_graph_attributes(inplace=True)
    print(my_cbn.number_of_nodes(),
          my_cbn.number_of_nodes(typ="core"),
          my_cbn.number_of_nodes(typ="boundary"))
    print(my_cbn.number_of_edges(),
          my_cbn.number_of_edges(typ="core"),
          my_cbn.number_of_edges(typ="boundary"))

    my_cbn.to_dsv("test.tsv", delimiter=";", data_cols=["subject", "relation"], header=("src", "reg"))

    import_start = time.time()
    my_cbn = CausalNetwork.from_tsv("data/BAGen03Large/10000_3062_247405_core.tsv", edge_type="core")
    my_cbn.add_edges_from_tsv("data/BAGen03Large/10000_3062_247405_boundary.tsv", edge_type="boundary")
    import_end = time.time()
    print("Import time: ", import_end - import_start)

    dataset_folder = "data/ExpressionExamplesGen02"
    large_data = dict()
    for file_name in os.listdir(dataset_folder):
        if file_name.startswith("10000_3062_247405"):
            large_data[file_name] = pd.read_table(os.path.join(dataset_folder, file_name))

    example_run(my_cbn, large_data, permutations=["o", "k1", "k2"])


def test_opposing_value_pruning():
    rng = np.random.default_rng(42)
    my_cbn = CausalNetwork()
    core_nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for x, y in zip(core_nodes, core_nodes[1:]):
        rel = rng.choice([-1., 1.])
        my_cbn.add_edge(x, y, rel=rel, typ="core")

    for _ in range(11):
        x, y = rng.choice(core_nodes, size=2, replace=False)
        rel = rng.choice([-1., 1.])
        my_cbn.add_edge(x, y, rel=rel, typ="core")

    boundary_nodes = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    for x, y in zip(core_nodes, boundary_nodes):
        my_cbn.add_edge(x, y, rel=1., typ="boundary")

    data = pd.DataFrame({"nodeID": boundary_nodes,
                         "logFC": [1., 0.5, 0.25, 0., -0.2, -0.4, -0.6, -0.8, -1., -2.]})

    for sparse in [True, False]:
        for ovmin in [3., 2., 1., 0.8, 0.6, 0.4, 0.2, 0.]:
            logging.info("ovmin: %f" % ovmin)
            my_cbn.toponpa({"test": data}, opposing_value_pruning_mode="nullify",
                           opposing_value_minimum_amplitude=ovmin, boundary_edge_minimum=0,
                           compute_statistics=False, permutations=None, sparse=sparse)


if __name__ == "__main__":
    dataset_files = ["CS (2m) + Sham (3m)", "CS (2m) + Sham (5m)", "CS (4m) + Sham (1m)",
                     "CS (4m) + Sham (3m)", "CS (5m)", "CS (7m)"]
    copd1_data = dict()
    for file in dataset_files:
        copd1_data[file] = pd.read_table("./data/COPD1/" + file + ".tsv")
        copd1_data[file] = copd1_data[file].rename(columns={"nodeLabel": "nodeID", "foldChange": "logFC"})

    dataset_folder = "data/ExpressionExamplesGen05"
    large_data = dict()
    for file_name in os.listdir(dataset_folder):
        if file_name.startswith("500_175_15457"):
            large_data[file_name] = pd.read_table(os.path.join(dataset_folder, file_name), delimiter=',')

    logging.basicConfig(**DEFAULT_LOGGING_KWARGS)
    mm_apoptosis = CausalNetwork.from_tsv("data/NPANetworks/Mm_CFA_Apoptosis_backbone.tsv", edge_type="core")
    mm_apoptosis.add_edges_from_tsv("data/NPANetworks/Mm_CFA_Apoptosis_downstream.tsv", edge_type="boundary")
    example_run(mm_apoptosis, copd1_data, permutations=["o", "k1", "k2"], sparse=False)
    exit()
    # example_run(mm_apoptosis, copd1_data, permutations=["o", "k1", "k2"], sparse=True)
    # mm_apoptosis.toponpa(large_data)
    logging.info("Starting rewiring")
    test_rewiring(mm_apoptosis, copd1_data)
    logging.info("Finished rewiring")

    # test_opposing_value_pruning()
