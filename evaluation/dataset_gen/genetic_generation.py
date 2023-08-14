import os
import sys
import logging
import uuid

import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt

from bnpa.CausalNetwork import CausalNetwork
from bnpa.toponpa.core import perturbation_amplitude


def generate_mask(array, value):
    array_indices = np.flatnonzero(array == value)
    array_mask = np.zeros(len(array), dtype=bool)
    array_mask[array_indices] = True
    return array_mask


def generate_dataset(network_name, causalbionet, population_size,
                     max_iterations, target_fitness, mutation_rate, directionality=None,
                     verbose=True, log_frequency=0.01, seed=None):
    # Create generator, instance id and set up logging
    rng = np.random.default_rng(seed=seed)
    instance_id = str(uuid.uuid4())
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(levelname)s -- %(message)s")

    # Get laplacian matrices
    core_edge_count = causalbionet.number_of_edges("core")
    lap_b, lap_c, lap_q, node_ordering = causalbionet.get_laplacians(verbose=False)
    inference_matrix = - np.matmul(la.inv(lap_c), lap_b)

    # Determine boundary node labels
    boundary_nodes = causalbionet.get_nodes(typ="boundary")
    boundary_node_count = len(boundary_nodes)
    boundary_node_labels = node_ordering[-boundary_node_count:]

    # Determine number of mutations
    boundary_node_count = causalbionet.number_of_nodes("boundary")
    mutation_node_count = np.rint(boundary_node_count * mutation_rate).astype(int)

    # Determine sign of boundary edges (laplacian has opposite sign of adjacency matrix)
    positive_boundary_edge_counts = np.sum(lap_b < 0, axis=0)
    negative_boundary_edge_counts = np.sum(lap_b > 0, axis=0)
    boundary_direction = np.array([1 if p > n else -1 if n > p else 0 for p, n in
                                   zip(positive_boundary_edge_counts, negative_boundary_edge_counts)])
    positive_boundary_mask = generate_mask(boundary_direction, 1)
    negative_boundary_mask = generate_mask(boundary_direction, -1)
    neutral_boundary_mask = generate_mask(boundary_direction, 0)

    # Load data samples
    positive_data_samples = pd.read_table("../../data/ExpressionExamples/positive_samples.csv", sep=",")
    negative_data_samples = pd.read_table("../../data/ExpressionExamples/negative_samples.csv", sep=",")
    combined_data_samples = pd.concat([positive_data_samples, negative_data_samples],
                                      axis=0, ignore_index=True)

    # Generate initial population
    if directionality is not None and directionality in ["consistent", "1", "opposing", "-1"]:
        if directionality in ["opposing", "-1"]:
            positive_data_samples, negative_data_samples = negative_data_samples, positive_data_samples

        population = np.empty((population_size, boundary_node_count))
        population[:, positive_boundary_mask] = rng.choice(
            positive_data_samples['log_fc'],
            size=(population_size, np.sum(positive_boundary_mask)),
            replace=True
        )
        population[:, negative_boundary_mask] = rng.choice(
            negative_data_samples['log_fc'],
            size=(population_size, np.sum(negative_boundary_mask)),
            replace=True
        )
        population[:, neutral_boundary_mask] = rng.choice(
            combined_data_samples['log_fc'],
            size=(population_size, np.sum(neutral_boundary_mask)),
            replace=True
        )
    else:
        population = rng.choice(
            combined_data_samples['log_fc'],
            size=(population_size, boundary_node_count),
            replace=True
        )

    average_fitness_per_iter = []
    best_fitness_per_iter = []
    worst_fitness_per_iter = []
    best_fitness_overall = 0
    best_dataset_overall = None
    next_log = 0.
    for ga_iter in range(max_iterations + 1):
        # Evaluate population with npa
        core_coefficients = np.matmul(inference_matrix, population.T)
        scores = perturbation_amplitude(lap_q, core_coefficients, core_edge_count)

        # Save best network
        average_fitness_per_iter.append(np.mean(scores))
        worst_fitness_per_iter.append(np.min(scores))
        best_score = np.max(scores)
        best_fitness_per_iter.append(best_score)

        if best_score > best_fitness_overall:
            best_fitness_overall = best_score
            best_dataset_overall = population[np.argmax(scores)]

        if verbose and ga_iter >= next_log * max_iterations:
            completion = ga_iter / max_iterations if max_iterations > 0 else 1
            logging.info("Iteration %d best score: %f (%.2f%% done)",
                         ga_iter, best_score, completion * 100)
            next_log += log_frequency

        if ga_iter == max_iterations or \
                (target_fitness is not None
                 and best_score >= target_fitness):
            break

        probabilities = scores / np.sum(scores)

        parents = np.array([
            population[rng.choice(
                population_size, size=2,
                replace=False, p=probabilities
            )] for _ in range(population_size)
        ])

        # Uniform crossover
        new_population = np.where(
            rng.choice([True, False], size=(population_size, boundary_node_count)),
            parents[:, 0, :], parents[:, 1, :]
        )

        # Mutate
        mutation_indices = np.array([
            rng.choice(
                boundary_node_count,
                size=mutation_node_count,
                replace=False
            ) for _ in range(population_size)
        ])
        mutation_mask = np.zeros((population_size, boundary_node_count), dtype=bool)
        mutation_mask[np.arange(population_size)[:, None], mutation_indices] = True

        if directionality in ["consistent", "1", "opposing", "-1"]:
            positive_mutation_mask = np.logical_and(mutation_mask, positive_boundary_mask)
            negative_mutation_mask = np.logical_and(mutation_mask, negative_boundary_mask)
            neutral_mutation_mask = np.logical_and(mutation_mask, neutral_boundary_mask)

            new_population[positive_mutation_mask] = rng.choice(
                positive_data_samples['log_fc'],
                size=np.sum(positive_mutation_mask),
                replace=True
            )
            new_population[negative_mutation_mask] = rng.choice(
                negative_data_samples['log_fc'],
                size=np.sum(negative_mutation_mask),
                replace=True
            )
            new_population[neutral_mutation_mask] = rng.choice(
                combined_data_samples['log_fc'],
                size=np.sum(neutral_mutation_mask),
                replace=True
            )

        else:
            new_population[mutation_mask] = rng.choice(
                combined_data_samples['log_fc'],
                size=population_size * mutation_node_count,
                replace=True
            )

        population = np.array(new_population)

    # Plot fitness
    plt.figure(figsize=(8, 4))
    plt.title(network_name + ", population size " + str(population_size))
    plt.plot(average_fitness_per_iter, label="Average fitness")
    plt.plot(best_fitness_per_iter, label="Best fitness")
    plt.plot(worst_fitness_per_iter, label="Worst fitness")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()
    plt.savefig("dataset_generation_plots/" + network_name +
                "_(" + str(directionality) + ")_dataset_" +
                instance_id + "_fitness.png")
    plt.show()
    plt.close()

    # Save best dataset
    best_dataset = pd.DataFrame(best_dataset_overall, columns=["logFC"])
    best_dataset["nodeID"] = boundary_node_labels
    best_dataset.to_csv("../../data/ExpressionExamplesGen04/" + network_name +
                        "_(" + str(directionality) + ")_dataset_" +
                        instance_id + ".csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(levelname)s -- %(message)s")

    # Network
    network_folder = "../../data/BAGen03Large/"
    core_suffix = "_core.tsv"
    boundary_suffix = "_boundary.tsv"
    directions = ["0"]
    datasets_per_direction = 1
    pop_size = 500
    iters = 2000

    for file_name in os.listdir(network_folder):
        if file_name.endswith(core_suffix) and not file_name.startswith("Hs_CST_Xenobiotic"):
            network_title = file_name[:-len(core_suffix)]
            logging.info("Generating datasets for %s", network_title)

            core_file = network_folder + network_title + core_suffix
            boundary_file = network_folder + network_title + boundary_suffix

            my_cbn = CausalNetwork.from_tsv(core_file, edge_type="core")
            my_cbn.add_edges_from_tsv(boundary_file, edge_type="boundary")
            my_cbn.infer_graph_attributes(verbose=False, inplace=True)

            for didx, direction in enumerate(directions):
                for i in range(datasets_per_direction):
                    logging.info("Generating dataset %d/%d for %s",
                                 didx*datasets_per_direction + i + 1,
                                 len(directions)*datasets_per_direction,
                                 network_title)

                    generate_dataset(
                        network_title, my_cbn,
                        population_size=pop_size, max_iterations=iters, target_fitness=None,
                        directionality=direction, mutation_rate=0.002,
                        verbose=True, log_frequency=0.1, seed=(i+1)*72
                    )
