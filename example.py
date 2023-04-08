import pandas as pd

from bnpa.CausalNetwork import CausalNetwork


if __name__ == "__main__":
    my_cbn = CausalNetwork.from_tsv("data/NPANetworks/Hs_CFA_Apoptosis_backbone.tsv",
                                    "data/NPANetworks/Hs_CFA_Apoptosis_downstream.tsv")

    dataset_files = ["CS (2m) + Sham (3m)", "CS (2m) + Sham (5m)", "CS (4m) + Sham (1m)",
                     "CS (4m) + Sham (3m)", "CS (5m)", "CS (7m)"]
    datasets = dict()
    for dataset in dataset_files:
        datasets[dataset] = pd.read_table("./data/COPD1/" + dataset + ".tsv")
        datasets[dataset] = datasets[dataset].rename(columns={"nodeLabel": "nodeID", "foldChange": "logFC"})

    results = my_cbn.compute_npa(datasets)

    output_file = "test_results.txt"
    with open(output_file, "w") as f:
        f.write(results.network_info.to_string() + "\n")
        f.write(results.permutation_info.to_string() + "\n")
