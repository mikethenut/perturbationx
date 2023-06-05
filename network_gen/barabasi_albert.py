import csv
import uuid
from collections import Counter

import numpy as np


CORE_EDGE_INTERCEPT = -6.1214
CORE_EDGE_SLOPE = 1.6455
CORE_EDGE_STD = 12.278

CORE_NEG_EDGE_INTERCEPT = -2.3347
CORE_NEG_EDGE_SLOPE = 0.16530
CORE_NEG_EDGE_STD = 10.041

INNER_BOUNDARY_INTERCEPT = 2.5953
INNER_BOUNDARY_SLOPE = 0.58824
INNER_BOUNDARY_STD = 7.8776

BOUNDARY_EDGE_INTERCEPT = 1403.4
BOUNDARY_EDGE_SLOPE = 169.00
BOUNDARY_EDGE_STD = 2401.5

OUTER_BOUNDARY_INTERCEPT = 1718.8
OUTER_BOUNDARY_SLOPE = 0.35327
OUTER_BOUNDARY_STD = 439.93

BOUNDARY_NEG_EDGE_INTERCEPT = -84.853
BOUNDARY_NEG_EDGE_SLOPE = 0.37767
BOUNDARY_NEG_EDGE_STD = 405.26


def truncated_normal_draw(mean, std, min_val, max_val):
    draw = None
    while draw is None or draw < min_val or draw > max_val:
        draw = rng.normal(mean, std)
    return draw


if __name__ == "__main__":
    node_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
    node_sizes = node_sizes * 5

    rng = np.random.default_rng()
    for node_count in node_sizes:
        # Compute the edge count
        edge_forecast_mean = CORE_EDGE_INTERCEPT + node_count * CORE_EDGE_SLOPE
        edge_count = np.rint(truncated_normal_draw(
                edge_forecast_mean, CORE_EDGE_STD, node_count, 2 * edge_forecast_mean - node_count
        )).astype(int)

        # Compute the negative edge count
        neg_edge_forecast_mean = CORE_NEG_EDGE_INTERCEPT + edge_count * CORE_NEG_EDGE_SLOPE
        negative_edge_count = np.rint(truncated_normal_draw(
            neg_edge_forecast_mean, CORE_NEG_EDGE_STD, 0, 2 * neg_edge_forecast_mean
        )).astype(int)

        print("Node count: {}, Edge count: {}, Positive edge count: {}, Negative edge count: {}".format(
            node_count, edge_count, edge_count - negative_edge_count, negative_edge_count))

        # Generate the core network
        node_multiplicity = []
        with open("../data/BAGen01/" + str(node_count) + '_' + str(edge_count) +
                  '_' + str(negative_edge_count) + "_core.tsv",
                  'w', newline='') as output_file:
            dsv_file = csv.writer(output_file, delimiter="\t")
            dsv_file.writerow(["subject", "object", "relation", "type"])

            # Start with a star graph (same as nx)
            m = np.ceil(edge_count / node_count).astype(int)
            nodes_added = m + 1
            edges_remaining = edge_count
            negative_edges_remaining = negative_edge_count

            center_node_id = uuid.uuid4()
            node_multiplicity.extend([center_node_id] * m)
            for _ in range(m):
                node_id = uuid.uuid4()
                if rng.uniform() < negative_edges_remaining / edges_remaining:
                    relation = "-1"
                    negative_edges_remaining -= 1
                else:
                    relation = "1"

                if rng.uniform() < 0.5:
                    dsv_file.writerow([node_id, center_node_id, relation, "core"])
                else:
                    dsv_file.writerow([center_node_id, node_id, relation, "core"])
                edges_remaining -= 1
                node_multiplicity.append(node_id)

            while nodes_added < node_count:
                node_id = uuid.uuid4()
                m = edges_remaining / (node_count - nodes_added)
                m = np.ceil(m).astype(int) if rng.uniform() < m % 1 else np.floor(m).astype(int)

                targets = set()
                while len(targets) < m:
                    targets.update(rng.choice(node_multiplicity, m - len(targets)))

                for t in targets:
                    if rng.uniform() < negative_edges_remaining / edges_remaining:
                        relation = "-1"
                        negative_edges_remaining -= 1
                    else:
                        relation = "1"

                    if rng.uniform() < 0.5:
                        dsv_file.writerow([node_id, t, relation, "core"])
                    else:
                        dsv_file.writerow([t, node_id, relation, "core"])
                    edges_remaining -= 1

                node_multiplicity.extend(targets)
                node_multiplicity.extend([node_id] * m)

                nodes_added += 1

        # Compute the inner boundary node count
        inner_boundary_forecast_mean = INNER_BOUNDARY_INTERCEPT + node_count * INNER_BOUNDARY_SLOPE
        inner_boundary_count = np.rint(truncated_normal_draw(
            inner_boundary_forecast_mean, INNER_BOUNDARY_STD, 1, 2 * inner_boundary_forecast_mean - 1
        )).astype(int)

        # Compute boundary edges
        boundary_edge_forecast_mean = BOUNDARY_EDGE_INTERCEPT + inner_boundary_count * BOUNDARY_EDGE_SLOPE
        boundary_edge_count = np.rint(truncated_normal_draw(
            boundary_edge_forecast_mean, BOUNDARY_EDGE_STD, 1, 2 * boundary_edge_forecast_mean - 1
        )).astype(int)

        # Compute the boundary node count
        outer_boundary_forecast_mean = OUTER_BOUNDARY_INTERCEPT + boundary_edge_count * OUTER_BOUNDARY_SLOPE
        outer_boundary_count = np.rint(truncated_normal_draw(
            outer_boundary_forecast_mean, OUTER_BOUNDARY_STD, 1, 2 * outer_boundary_forecast_mean - 1
        )).astype(int)

        # Compute the negative boundary edge count
        neg_boundary_edge_forecast_mean = BOUNDARY_NEG_EDGE_INTERCEPT + boundary_edge_count * BOUNDARY_NEG_EDGE_SLOPE
        negative_boundary_edge_count = np.rint(truncated_normal_draw(
            neg_boundary_edge_forecast_mean, BOUNDARY_NEG_EDGE_STD, 0, 2 * neg_boundary_edge_forecast_mean
        )).astype(int)

        print(
            "Inner boundary count: {}, Outer boundary count: {}, Boundary edge count: {}, "
            "Positive boundary edge count: {}, Negative boundary edge count: {}".format(
             inner_boundary_count, outer_boundary_count, boundary_edge_count,
             boundary_edge_count - negative_boundary_edge_count, negative_boundary_edge_count)
        )

        # Generate the boundary network
        core_nodes = list(set(node_multiplicity))
        inner_core_nodes = rng.choice(core_nodes, inner_boundary_count, replace=False)

        # TODO: This is significantly too slow
        # Try to assign a number of edges to each inner boundary node
        # While ensuring that each boundary node has at least one edge

        boundary_nodes = []
        with open("../data/BAGen01/" + str(node_count) + '_' + str(edge_count) +
                  '_' + str(negative_edge_count) + "_boundary.tsv",
                  'w', newline='') as output_file:
            dsv_file = csv.writer(output_file, delimiter="\t")
            dsv_file.writerow(["subject", "object", "relation", "type"])

            # Add one edge for every boundary node
            boundary_edges_remaining = boundary_edge_count
            negative_edges_remaining = negative_boundary_edge_count
            for _ in range(outer_boundary_count):
                node_id = uuid.uuid4()
                boundary_nodes.append(node_id)
                core_node = rng.choice(core_nodes)

                if rng.uniform() < negative_edges_remaining / boundary_edges_remaining:
                    relation = "-1"
                    negative_edges_remaining -= 1
                else:
                    relation = "1"

                dsv_file.writerow([core_node, node_id, relation, "boundary"])
                boundary_edges_remaining -= 1

            # Add remaining boundary edges: core node should be random, boundary node should be preferential
            while boundary_edges_remaining > 0:
                core_node = rng.choice(core_nodes)
                boundary_node = rng.choice(boundary_nodes)
                boundary_nodes.append(boundary_node)

                relation = "1"
                dsv_file.writerow([core_node, boundary_node, relation, "boundary"])
