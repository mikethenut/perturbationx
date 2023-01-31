import unittest
import pickle
import os

import numpy as np

from bnpa.npa import core

fixtures_path = os.path.join(os.path.dirname(__file__), '../fixtures')


class TestNPAPreprocess(unittest.TestCase):
    def testNPAValueDiffusion(self):
        with open(os.path.join(fixtures_path, "diffusion_matrix.pkl"), 'rb') as dmf, \
                open(os.path.join(fixtures_path, "node_name.pkl"), 'rb') as nnf, \
                open(os.path.join(fixtures_path, "dataset.pkl"), 'rb') as df, \
                open(os.path.join(fixtures_path, "diffused_values.pkl"), 'rb') as dvf:

            diffusion_matrix = pickle.load(dmf)
            network_node_name = pickle.load(nnf)
            dataset_node_name, fold_change, _ = pickle.load(df)

            diffused_values_true = pickle.load(dvf)
            diffused_values = core.npa_value_diffusion(diffusion_matrix, network_node_name,
                                                       fold_change, dataset_node_name)
            self.assertTrue(np.allclose(diffused_values, diffused_values_true))

    def testNetworkPerturbationAmplitude(self):
        with open(os.path.join(fixtures_path, "laplacians.pkl"), 'rb') as lf, \
                open(os.path.join(fixtures_path, "diffused_values.pkl"), 'rb') as dvf:

            _, laplacian_backbone_signless, _ = pickle.load(lf)
            diffused_values = pickle.load(dvf)

            perturbation_score = core.network_perturbation_amplitude(230, laplacian_backbone_signless, diffused_values)
            self.assertAlmostEqual(perturbation_score, 0.102559356508792)


if __name__ == '__main__':
    unittest.main()
