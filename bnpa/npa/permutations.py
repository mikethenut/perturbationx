import numpy as np
import numpy.random
import numpy.linalg as la

from bnpa.npa.core import value_inference, perturbation_amplitude


def permutation_test_o(lap: dict, fold_change: np.ndarray,
                       backbone_edge_count: int, permutations=500, seed=None):
    if seed is None:
        generator = np.random.default_rng()
    else:
        generator = np.random.default_rng(seed)

    true_perturbation = perturbation_amplitude(lap['q'], value_inference(lap, fold_change), backbone_edge_count)
    distribution = []

    for p in range(permutations):
        permuted_downstream = generator.permutation(fold_change)
        backbone_values = value_inference(lap, permuted_downstream)
        sample_perturbation = perturbation_amplitude(lap['q'], backbone_values, backbone_edge_count)
        distribution.append(sample_perturbation)

    p_value = sum(sample > true_perturbation for sample in distribution) / permutations
    return p_value, distribution


def permutation_test_k(l3_permutations, l2: np.ndarray, q: np.ndarray, fold_change: np.ndarray,
                       backbone_edge_count: int, true_perturbation: float):

    distribution = []
    temp = - l2.dot(fold_change)
    for l3 in l3_permutations:
        backbone_values = np.matmul(la.inv(l3), temp)
        sample_perturbation = np.matmul(q.dot(backbone_values), backbone_values) / backbone_edge_count
        distribution.append(sample_perturbation)

    p_value = sum(sample > true_perturbation for sample in distribution) / len(distribution)
    return p_value, distribution
