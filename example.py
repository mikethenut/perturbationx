import pandas as pd

from bnpa.CausalNetwork import CausalNetwork


if __name__ == "__main__":
    my_cbn = CausalNetwork.from_tsv("data/NPANetworks/Hs_CFA_Apoptosis_backbone.tsv", edge_type="core")
    my_cbn.add_edges_from_tsv("data/NPANetworks/Hs_CFA_Apoptosis_downstream.tsv", edge_type="boundary")
    my_cbn.add_edge("p(MGI:Bcl2)", "p(MGI:Bcl2a1b)", "0.", "core")
    # my_cbn.to_dsv("test.tsv", delimiter=";", data_cols=["subject", "relation"], header=("src", "reg"))
    # my_cbn.display()
    dataset_files = ["CS (2m) + Sham (3m)", "CS (2m) + Sham (5m)", "CS (4m) + Sham (1m)",
                     "CS (4m) + Sham (3m)", "CS (5m)", "CS (7m)"]
    datasets = dict()
    for dataset in dataset_files:
        datasets[dataset] = pd.read_table("./data/COPD1/" + dataset + ".tsv")
        datasets[dataset] = datasets[dataset].rename(columns={"nodeLabel": "nodeID", "foldChange": "logFC"})
        break

    results = my_cbn.compute_npa(datasets)
    # results.display_network()
    # results.color_nodes("coefficient", "CS (2m) + Sham (3m)")

    output_file = "test_results.txt"
    with open(output_file, "w") as f:
        f.write(str(results.metadata()) + "\n")
        f.write(results.global_info().to_string() + "\n")
        for attr in results.node_attributes():
            f.write(results.node_info(attr).to_string() + "\n")
        #for distr in results.distributions():
        #    results.plot_distribution(distr)
