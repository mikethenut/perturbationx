import pandas as pd

from bnpa.CausalNetwork import CausalNetwork


def example_run(causalbionet, datasets, permutations=("o", "k1", "k2")):
    results = causalbionet.compute_npa(
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

    my_cbn = CausalNetwork.from_tsv("data/NPANetworks/Mm_CFA_Apoptosis_backbone.tsv", edge_type="core")
    my_cbn.add_edges_from_tsv("data/NPANetworks/Mm_CFA_Apoptosis_downstream.tsv", edge_type="boundary")
    # my_cbn.add_edge("p(MGI:Bcl2)", "p(MGI:Bcl2a1b)", "0.", "core")
    # my_cbn.to_dsv("test.tsv", delimiter=";", data_cols=["subject", "relation"], header=("src", "reg"))

    example_run(my_cbn, copd1_data, permutations=[])
