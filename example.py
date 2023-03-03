from bnpa.CausalNetwork import CausalNetwork
from bnpa.Dataset import Dataset


def run_example(causal_network, data_path):
    data = Dataset(input_format='NPA', input_path=data_path)
    results = causal_network.compute_npa(data)
    print(data_path, results.npa, results.npa_variance, results.npa_confidence_interval)
    # print(results.node_contributions, results.node_coefficients, results.node_variance,
    #      results.node_confidence_interval, results.node_p_value)
    # print(results.o_value, results.o_distribution)
    # print(results.k_value, results.k_distribution)


if __name__ == "__main__":
    my_cbn = CausalNetwork.from_tsv("data/NPANetworks/Hs_CFA_Apoptosis_backbone.tsv",
                                    "data/NPANetworks/Hs_CPR_Cell_Cycle_downstream.tsv")

    for dataset in ['CS (2m) + Sham (3m)', 'CS (2m) + Sham (5m)', 'CS (4m) + Sham (1m)',
                    'CS (4m) + Sham (3m)', 'CS (5m)', 'CS (7m)']:
        run_example(my_cbn, "./data/COPD1/" + dataset + '.tsv')
