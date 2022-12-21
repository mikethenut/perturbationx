from bnpa.CBN import CBN

if __name__ == "__main__":
    myCBN = CBN(input_format='NPA',
                backbone_network="./data/networks/Hs_CFA_Apoptosis_backbone.tsv",
                downstream_network="./data/networks/Hs_CFA_Apoptosis_downstream.tsv")
    myCBN.write_edge_list(name_nodes=True)
    myCBN.write_laplacians(lb_file="./output/laplacian_backbone.txt",
                           ld_file="./output/laplacian_downstream.txt",
                           lbs_file="./output/laplacian_backbone_signless.txt")
