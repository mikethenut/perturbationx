import csv
import uuid
import logging
import sys

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

BOUNDARY_NODE_INTERCEPT = 2682.4
BOUNDARY_NODE_SLOPE = 31.513
BOUNDARY_NODE_STD = 1371.1

BOUNDARY_EDGE_INTERCEPT = -4267.4
BOUNDARY_EDGE_SLOPE = 2.7203
BOUNDARY_EDGE_STD = 1220.8

BOUNDARY_NEG_EDGE_INTERCEPT = -84.853
BOUNDARY_NEG_EDGE_SLOPE = 0.37767
BOUNDARY_NEG_EDGE_STD = 405.26


def truncated_normal_draw(mean, std, min_val, max_val):
    draw = None
    while draw is None or draw < min_val or draw > max_val:
        draw = rng.normal(mean, std)
    return draw


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(levelname)s -- %(message)s")
    node_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
    node_sizes = node_sizes * 5

    rng = np.random.default_rng()
    for node_count in node_sizes:
        logging.info("Generating network with {} nodes".format(node_count))
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

        logging.info("Node count: {}, Edge count: {}, Positive edge count: {}, Negative edge count: {}".format(
            node_count, edge_count, edge_count - negative_edge_count, negative_edge_count)
        )

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
                    targets.update(rng.choice(node_multiplicity, m - len(targets), replace=False))

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

        logging.info("Core network generated")

        # Compute the inner boundary node count
        inner_boundary_forecast_mean = INNER_BOUNDARY_INTERCEPT + node_count * INNER_BOUNDARY_SLOPE
        inner_boundary_count = np.rint(truncated_normal_draw(
            inner_boundary_forecast_mean, INNER_BOUNDARY_STD,
            2 * inner_boundary_forecast_mean - node_count, node_count
        )).astype(int)

        # Compute the (outer) boundary node count
        boundary_node_forecast_mean = BOUNDARY_NODE_INTERCEPT + node_count * BOUNDARY_NODE_SLOPE
        # Hard lower limit on the number of boundary nodes: 2500
        # This is to ensure that the number of edges is not lower than the number of nodes
        boundary_node_count = np.rint(truncated_normal_draw(
            boundary_node_forecast_mean, BOUNDARY_NODE_STD,
            2500, 2 * boundary_node_forecast_mean - 2500
        )).astype(int)

        # Compute boundary edges
        boundary_edge_forecast_mean = BOUNDARY_EDGE_INTERCEPT + boundary_node_count * BOUNDARY_EDGE_SLOPE
        boundary_edge_count = np.rint(truncated_normal_draw(
            boundary_edge_forecast_mean, BOUNDARY_EDGE_STD,
            boundary_node_count, 2 * boundary_edge_forecast_mean - boundary_node_count
        )).astype(int)

        # Compute the negative boundary edge count
        neg_boundary_edge_forecast_mean = BOUNDARY_NEG_EDGE_INTERCEPT + boundary_edge_count * BOUNDARY_NEG_EDGE_SLOPE
        negative_boundary_edge_count = np.rint(truncated_normal_draw(
            neg_boundary_edge_forecast_mean, BOUNDARY_NEG_EDGE_STD, 0, 2 * neg_boundary_edge_forecast_mean
        )).astype(int)

        # Generate the boundary network
        core_nodes = list(set(node_multiplicity))
        inner_boundary_nodes = rng.choice(core_nodes, inner_boundary_count, replace=False)

        # Assign a random number of edges to each inner boundary node
        # Note: This is not guaranteed to match the exact number of edges, but it should be close
        boundary_edge_distribution = {n: rng.uniform(0, 1) for n in inner_boundary_nodes}
        multiplier = boundary_edge_count / sum(boundary_edge_distribution.values())
        boundary_edge_distribution = {n: np.rint(boundary_edge_distribution[n] * multiplier).astype(int)
                                      for n in boundary_edge_distribution}

        logging.info(
            "Inner boundary count: {}, Outer boundary count: {}, Boundary edge count: {}, "
            "Positive boundary edge count: {}, Negative boundary edge count: {}".format(
             inner_boundary_count, boundary_node_count, sum(boundary_edge_distribution.values()),
             sum(boundary_edge_distribution.values()) - negative_boundary_edge_count, negative_boundary_edge_count)
        )

        # Add an edge for every boundary node
        boundary_nodes = [uuid.uuid4() for _ in range(boundary_node_count)]
        boundary_edges = {n: set() for n in inner_boundary_nodes}
        selected_source_nodes = sorted(rng.choice(
            [n for n in inner_boundary_nodes for _ in range(boundary_edge_distribution[n])],
            len(boundary_nodes), replace=False
        ))
        for n, m in zip(selected_source_nodes, boundary_nodes):
            boundary_edges[n].add(m)

        logging.info("Added initial boundary edges.")

        # Preferentially distribute the remaining edges
        boundary_node_multiplicity = boundary_nodes.copy()
        for n in inner_boundary_nodes:
            for m in boundary_edges[n]:
                boundary_node_multiplicity.remove(m)

            while len(boundary_edges[n]) < boundary_edge_distribution[n]:
                boundary_edges[n].update(rng.choice(
                    boundary_node_multiplicity, boundary_edge_distribution[n] - len(boundary_edges[n]), replace=False
                ))
            boundary_node_multiplicity.extend(boundary_edges[n])

        logging.info("Added remaining boundary edges.")

        # Randomize the sign of the edges
        relations = ["-1"] * negative_boundary_edge_count + \
                    ["1"] * (sum(boundary_edge_distribution.values()) - negative_boundary_edge_count)
        rng.shuffle(relations)

        logging.info("Randomized edge signs.")

        # Write the boundary network
        with open("../data/BAGen01/" + str(node_count) + '_' + str(edge_count) +
                  '_' + str(negative_edge_count) + "_boundary.tsv",
                  'w', newline='') as output_file:
            dsv_file = csv.writer(output_file, delimiter="\t")
            dsv_file.writerow(["subject", "object", "relation", "type"])

            rows = [[n, m, relations.pop(), "boundary"] for n in boundary_edges for m in boundary_edges[n]]
            dsv_file.writerows(rows)

        logging.info("Boundary network generated.")
