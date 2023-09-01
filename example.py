import os
import logging
import time

import pandas as pd

from perturbationx.resources import DEFAULT_LOGGING_KWARGS
from perturbationx import CausalNetwork


def example_run(causalbionet, datasets, permutations=("o", "k1", "k2")):
    results = causalbionet.toponpa(
        datasets, permutations=permutations, verbose=True
    )

    # results.plot_distribution("k2")
    # results.plot_distribution("k1")
    # results.plot_distribution("o")
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
    results = causalbionet.wire_edges(3,
        ["p(MGI:Bcl2)", "p(MGI:Bcl2a1b)", "act(p(MGI:Akt1))", "p(SFAM:\"AKT Family\")", "act(p(MGI:Map3k5))",
         "act(p(MGI:Nkx3-1))", "bp(MESHPP:Apoptosis)", "act(p(MGI:Birc5))"], ("1.", "-1."),
        3, datasets
    )

    for dataset in datasets:
        best_idx = -1
        best_score = 0.
        for idx, r in enumerate(results):
            if r[1][dataset] > best_score:
                best_score = r[1][dataset]
                best_idx = idx
        print(dataset, best_idx, results[best_idx][1][dataset])
        new_cbn = my_cbn.copy()
        new_cbn.modify_network(results[best_idx][0])


if __name__ == "__main__":
    dataset_files = ["CS (2m) + Sham (3m)", "CS (2m) + Sham (5m)", "CS (4m) + Sham (1m)",
                     "CS (4m) + Sham (3m)", "CS (5m)", "CS (7m)"]
    copd1_data = dict()
    for file in dataset_files:
        copd1_data[file] = pd.read_table("./data/COPD1/" + file + ".tsv")
        copd1_data[file] = copd1_data[file].rename(columns={"nodeLabel": "nodeID", "foldChange": "logFC"})

    logging.basicConfig(**DEFAULT_LOGGING_KWARGS)
    my_cbn = CausalNetwork.from_tsv("data/NPANetworks/Mm_CFA_Apoptosis_backbone.tsv", edge_type="core")
    my_cbn.add_edges_from_tsv("data/NPANetworks/Mm_CFA_Apoptosis_downstream.tsv", edge_type="boundary")
    example_run(my_cbn, copd1_data, permutations=["o", "k1", "k2"])
    logging.info("Finished run")
    test_rewiring(my_cbn, copd1_data)
    logging.info("Finished rewiring")

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

    # my_cbn.add_edge("p(MGI:Bcl2)", "p(MGI:Bcl2a1b)", "0.", "core")
    # my_cbn.to_dsv("test.tsv", delimiter=";", data_cols=["subject", "relation"], header=("src", "reg"))

    # example_run(my_cbn, copd1_data, permutations=[])
