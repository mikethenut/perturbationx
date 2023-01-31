from bnpa.CBN import CBN
from bnpa.Dataset import ContrastDataset


if __name__ == "__main__":
    myCBN = CBN(input_format='NPA',
                backbone_network="./data/NPANetworks/Hs_CFA_Apoptosis_backbone.tsv",
                downstream_network="./data/NPANetworks/Hs_CFA_Apoptosis_downstream.tsv")

    for dataset in ['CS (2m) + Sham (3m)', 'CS (2m) + Sham (5m)', 'CS (4m) + Sham (1m)',
                    'CS (4m) + Sham (3m)', 'CS (5m)', 'CS (7m)']:
        data = ContrastDataset(input_format='NPA', input_path="./data/COPD1/" + dataset + '.tsv')
        backbone_values = myCBN.diffuse_values(data, diffusion_method='NPA')
        npa_score = myCBN.compute_perturbation(data)
        print(dataset + ": " + str(npa_score))
