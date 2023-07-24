import sys
import uuid
import json
import csv
import logging

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


GLOBAL_RNG = None
FEATURE_ORDERING = [
    "core_node_count",
    "core_edge_count",
    "core_negative_edge_count",
    "inner_boundary_node_count",
    "boundary_node_count",
    "boundary_edge_count",
    "boundary_negative_edge_count"
]


def rrmse(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("The shape of y_true and y_pred must be the same.")

    true_mean = np.mean(y_true, axis=0)
    mean_se = np.sum((y_true - true_mean) ** 2, axis=0)
    pred_se = np.sum((y_true - y_pred) ** 2, axis=0)

    return np.sqrt(pred_se / mean_se)


def get_samples(network_statistics, network_names=None):
    if network_names is None:
        network_names = network_statistics.keys()

    features = [[network_statistics[n][f] for n in network_names] for f in FEATURE_ORDERING]
    return np.array(features).T


def generate_model(samples, plot_residuals=False, filename=None):
    sample_x = samples[:, 0].reshape(-1, 1)
    sample_y = samples[:, 1:]

    model = LinearRegression()
    model.fit(sample_x, sample_y)
    predicted_y = model.predict(sample_x)
    error = rrmse(sample_y, predicted_y)
    logging.info(f"Base Error: {error}")
    logging.info(f"Base Scores: {model.score(sample_x, sample_y)}")

    residual_y = sample_y - predicted_y

    if plot_residuals:
        residuals = np.concatenate((sample_x, residual_y), axis=1)
        residual_df = pd.DataFrame(residuals)
        residual_df.rename(columns={i: FEATURE_ORDERING[i] for i in range(len(FEATURE_ORDERING))}, inplace=True)

        plt.figure(figsize=(6, 6))
        sns.pairplot(residual_df)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename + "_pairplot.png")
        plt.show()
        plt.close()

        plt.figure(figsize=(8, 8))
        sns.heatmap(residual_df.corr(), annot=True, cmap="vlag", vmin=-1, center=0, vmax=1)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename + "_heatmap.png")
        plt.show()
        plt.close()

    return model, residual_y


def is_sample_valid(sample):
    if sample is None:
        return False

    # core edge count, core negative edge count
    if sample["core_edge_count"] < sample["core_node_count"] - 1:
        return False
    if sample["core_edge_count"] > sample["core_node_count"] * (sample["core_node_count"] - 1) / 2:
        return False
    if sample["core_negative_edge_count"] < 0:
        return False
    if sample["core_negative_edge_count"] > sample["core_edge_count"]:
        return False

    # inner boundary node count, boundary node count
    if sample["inner_boundary_node_count"] < 1:
        return False
    if sample["inner_boundary_node_count"] > sample["core_node_count"]:
        return False
    if sample["boundary_node_count"] < 1:
        return False

    # boundary edge count, boundary negative edge count
    if sample["boundary_edge_count"] < sample["inner_boundary_node_count"]:
        return False
    if sample["boundary_edge_count"] < sample["boundary_node_count"]:
        return False
    if sample["boundary_edge_count"] > sample["inner_boundary_node_count"] * sample["boundary_node_count"]:
        return False
    if sample["boundary_negative_edge_count"] < 0:
        return False
    if sample["boundary_negative_edge_count"] > sample["boundary_edge_count"]:
        return False

    return True


def sample_parameters(model, residuals, core_node_count):
    sample = model.predict([[core_node_count]])
    parameters = None

    while not is_sample_valid(parameters):
        residual_sample = GLOBAL_RNG.multivariate_normal(
            residuals.mean(axis=0),
            np.cov(residuals.T)
        )
        combined_sample = sample + residual_sample

        parameters = {"core_node_count": core_node_count}
        for i, f in enumerate(FEATURE_ORDERING[1:]):
            parameters[f] = np.rint(combined_sample[0][i]).astype(int)

    return parameters


def generate_network_core(parameters):
    logging.info("Generating network core")
    network_name = "_".join([
        str(parameters[f]) for f in
        ["core_node_count", "inner_boundary_node_count", "boundary_node_count"]
    ])

    node_multiplicity = []
    with open("../../data/BAGen03/" + network_name + "_core.tsv",
              'w', newline='') as output_file:
        dsv_file = csv.writer(output_file, delimiter="\t")
        dsv_file.writerow(["subject", "object", "relation", "type"])

        # Start with a star graph (same as nx)
        m = np.ceil(parameters["core_edge_count"] / parameters["core_node_count"]).astype(int)
        nodes_added = m + 1
        edges_remaining = parameters["core_edge_count"]
        negative_edges_remaining = parameters["core_negative_edge_count"]

        center_node_id = uuid.uuid4()
        node_multiplicity.extend([center_node_id] * m)
        for _ in range(m):
            node_id = uuid.uuid4()
            relation = "-1" if GLOBAL_RNG.uniform() < negative_edges_remaining / edges_remaining else "1"

            if GLOBAL_RNG.uniform() < 0.5:
                dsv_file.writerow([node_id, center_node_id, relation, "core"])
            else:
                dsv_file.writerow([center_node_id, node_id, relation, "core"])

            node_multiplicity.append(node_id)
            edges_remaining -= 1
            if relation == "-1":
                negative_edges_remaining -= 1

        while nodes_added < parameters["core_node_count"]:
            node_id = uuid.uuid4()
            m = edges_remaining / (parameters["core_node_count"] - nodes_added)
            m = np.ceil(m).astype(int) if GLOBAL_RNG.uniform() < m % 1 else np.floor(m).astype(int)

            targets = set()
            while len(targets) < m:
                targets.update(GLOBAL_RNG.choice(node_multiplicity, m - len(targets), replace=False))

            for t in targets:
                relation = "-1" if GLOBAL_RNG.uniform() < negative_edges_remaining / edges_remaining else "1"

                if GLOBAL_RNG.uniform() < 0.5:
                    dsv_file.writerow([node_id, t, relation, "core"])
                else:
                    dsv_file.writerow([t, node_id, relation, "core"])

                edges_remaining -= 1
                if relation == "-1":
                    negative_edges_remaining -= 1

            node_multiplicity.extend(targets)
            node_multiplicity.extend([node_id] * m)

            nodes_added += 1

    logging.info("Core network generated")

    return list(set(node_multiplicity))


def generate_network_boundary(parameters, core_nodes, boundary_out_degree_distribution):
    logging.info("Generating network boundary")
    network_name = "_".join([
        str(parameters[f]) for f in
        ["core_node_count", "inner_boundary_node_count", "boundary_node_count"]
    ])

    # Assign a random number of edges to each inner boundary node
    # Note: This is not guaranteed to match the exact number of edges, but it should be close
    inner_boundary_nodes = GLOBAL_RNG.choice(
        core_nodes, parameters["inner_boundary_node_count"], replace=False
    )
    boundary_edge_distribution = {n: GLOBAL_RNG.choice(boundary_out_degree_distribution, replace=True)
                                  for n in inner_boundary_nodes}
    multiplier = parameters["boundary_edge_count"] / sum(boundary_edge_distribution.values())
    boundary_edge_distribution = {n: np.rint(boundary_edge_distribution[n] * multiplier).astype(int)
                                  for n in boundary_edge_distribution}
    for n in boundary_edge_distribution:
        if boundary_edge_distribution[n] > parameters["boundary_node_count"]:
            boundary_edge_distribution[n] = parameters["boundary_node_count"]

    logging.info(f"Assigned {sum(boundary_edge_distribution.values())} boundary edges.")

    # Add an edge for every boundary node
    boundary_nodes = [uuid.uuid4() for _ in range(parameters["boundary_node_count"])]
    boundary_edges = {n: set() for n in inner_boundary_nodes}
    selected_source_nodes = sorted(GLOBAL_RNG.choice(
        [n for n in inner_boundary_nodes for _ in range(boundary_edge_distribution[n])],
        len(boundary_nodes), replace=False
    ))
    for n, m in zip(selected_source_nodes, boundary_nodes):
        boundary_edges[n].add(m)

    logging.info("Added initial boundary edges.")

    # Preferentially distribute the remaining edges
    boundary_node_multiplicity = boundary_nodes.copy()

    for n in inner_boundary_nodes:
        if len(boundary_edges[n]) >= boundary_edge_distribution[n]:
            boundary_edge_distribution[n] = len(boundary_edges[n])
            continue

        for m in boundary_edges[n]:
            boundary_node_multiplicity.remove(m)
        remaining_node_multiplicity = boundary_node_multiplicity

        while len(boundary_edges[n]) < boundary_edge_distribution[n]:
            remaining_node_multiplicity = [
                bn for bn in remaining_node_multiplicity
                if bn not in boundary_edges[n]
            ]
            boundary_edges[n].update(GLOBAL_RNG.choice(
                remaining_node_multiplicity,
                boundary_edge_distribution[n] - len(boundary_edges[n]),
                replace=False
            ))

        boundary_node_multiplicity.extend(boundary_edges[n])

    logging.info("Added remaining boundary edges.")

    # Randomize the sign of the edges
    relations = ["-1"] * parameters["boundary_negative_edge_count"] + \
                ["1"] * (sum(boundary_edge_distribution.values()) - parameters["boundary_negative_edge_count"])
    GLOBAL_RNG.shuffle(relations)

    logging.info("Randomized edge signs.")

    # Write the boundary network
    with open("../../data/BAGen03/" + network_name + "_boundary.tsv",
              'w', newline='') as output_file:
        dsv_file = csv.writer(output_file, delimiter="\t")
        dsv_file.writerow(["subject", "object", "relation", "type"])

        rows = [[n, m, relations.pop(), "boundary"] for n in boundary_edges for m in boundary_edges[n]]
        dsv_file.writerows(rows)

    logging.info("Boundary network generated.")


if __name__ == "__main__":
    GLOBAL_RNG = np.random.default_rng(seed=72)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(levelname)s -- %(message)s")
    node_counts = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    repetitions = 5

    group_a = ["Hs_CST_Oxidative_Stress", "Mm_CST_Oxidative_Stress", "Hs_CFA_Apoptosis", "Mm_CFA_Apoptosis",
               "Hs_IPN_Epithelial_Innate_Immune_Activation", "Mm_IPN_Epithelial_Innate_Immune_Activation",
               "Hs_IPN_Neutrophil_Signaling", "Mm_IPN_Neutrophil_Signaling"]
    group_b = ["Hs_CPR_Cell_Cycle", "Mm_CPR_Cell_Cycle", "Hs_TRA_ECM_Degradation", "Mm_TRA_ECM_Degradation"]
    group_c = ["Hs_CST_Xenobiotic_Metabolism", "Mm_CST_Xenobiotic_Metabolism", "Hs_CPR_Jak_Stat", "Mm_CPR_Jak_Stat"]
    group_d = ["Dr_ORG_Heart"]

    with open("../../output/network_stats/network_stats.json", "r") as in_file:
        net_stats = json.load(in_file)

    networks = net_stats.keys()
    samples = get_samples(net_stats, networks)

    boundary_out_degrees = list(d for n in networks for d in net_stats[n]["boundary_out_degrees"] if d != 0)
    linear_model, model_residuals = generate_model(
        samples, plot_residuals=True, filename="network_gen_plots/linear_model_residuals"
    )

    node_counts = []
    for node_count in node_counts:
        for _ in range(repetitions):
            net_params = sample_parameters(linear_model, model_residuals, node_count)
            logging.info("Generating network with parameters: {}".format(net_params))

            core_node_names = generate_network_core(net_params)
            generate_network_boundary(net_params, core_node_names, boundary_out_degrees)
