import json

import numpy as np
import matplotlib.pyplot as plt


def linear_regression(x, y, x_label, y_label, title, show_plot=True, output_file=None):
    x_avg = np.mean(x)
    x_var = np.std(x)**2
    print("Predictor mean, variance: {}, {}".format(x_avg, x_var))

    slope, intercept = np.polyfit(x, y, 1)
    print("Target range: {} - {} ({})".format(str(min(y)), str(max(y)), str(max(y) - min(y))))
    print("Slope: {}, Intercept: {}".format(str(slope), str(intercept)))

    errors = []
    for idx in range(len(x)):
        x_val = x[idx]
        y_val = y[idx]
        y_estimate = slope * x_val + intercept
        error = y_val - y_estimate
        errors.append(error)

    s = np.sqrt(np.sum(np.square(errors)) / (len(x) - 2))
    print("Errors: {}".format(str(errors)))
    print("Standard error: {}".format(str(s)))
    r_squared = 1.0 - (np.var(errors) / np.var(y))
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (len(y) - 1.0) / (len(y) - 2.0)
    print("R^2 (adjusted): {}, {}".format(str(r_squared), str(adjusted_r_squared)))

    # add some noise to the data to show overlapping points
    rng = np.random.default_rng()
    x_range = max(x) - min(x)
    y_range = max(y) - min(y)
    x = [d + rng.uniform(-x_range / 100., x_range / 100.) for d in x]
    y = [d + rng.uniform(-y_range / 100., y_range / 100.) for d in y]

    plt.clf()
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.5)
    plt.plot([min(x), max(x)], [intercept + slope * min(x), intercept + slope * max(x)],
             color='red', linestyle='-', linewidth=2)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if output_file is not None:
        plt.savefig(output_file)
    if show_plot:
        plt.show()


if __name__ == "__main__":
    with open("../output/network_stats/network_stats.json", "r") as in_file:
        net_stats = json.load(in_file)
    output_dir = "../output/network_stats"

    # CORE

    # Node count vs. edge count
    print("\nCore node count vs. edge count")
    x = [net_stats[network]["core_node_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    y = [net_stats[network]["core_edge_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    linear_regression(x, y, "Core node count", "Core edge count", "Core nodes vs. edges",
                      True, output_file=output_dir+"/core/core_node_edge_linear.png")

    # Core weight count
    print("\nCore weight distribution")
    x = [sum(1 for w in net_stats[network]["core_weights"])
         for network in net_stats if network != "Dr_ORG_Heart"]
    y = [sum(1 for w in net_stats[network]["core_weights"] if w < 0)
         for network in net_stats if network != "Dr_ORG_Heart"]
    linear_regression(x, y, "Core edges", "Core negative edges",
                      "Proportion of negative core weights", True,
                      output_file=output_dir+"/core/core_weight_linear.png")

    # BOUNDARY

    # Core nodes vs. inner boundary nodes
    print("\nCore node count vs. inner boundary node count")
    x = [net_stats[network]["core_node_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    y = [net_stats[network]["inner_boundary_node_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    linear_regression(x, y, "Core node count", "Inner boundary node count",
                      "Core vs. inner boundary nodes", True,
                      output_file=output_dir+"/boundary/core_node_inner_boundary_node_linear.png")

    # Core nodes vs. boundary nodes
    print("\nCore node count vs. boundary node count")
    x = [net_stats[network]["core_node_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    y = [net_stats[network]["boundary_node_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    linear_regression(x, y, "Core node count", "Boundary node count",
                      "Core vs. boundary nodes", True,
                      output_file=output_dir+"/boundary/core_node_boundary_node_linear.png")

    # Inner boundary nodes vs. boundary nodes
    print("\nInner boundary node count vs. boundary node count")
    x = [net_stats[network]["inner_boundary_node_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    y = [net_stats[network]["boundary_node_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    linear_regression(x, y, "Inner boundary node count", "Outer boundary node count",
                      "Inner vs. outer boundary nodes", True,
                      output_file=output_dir+"/boundary/inner_boundary_node_boundary_node_linear.png")

    # Inner boundary nodes vs. boundary edges
    print("\nInner boundary node count vs. boundary edge count")
    x = [net_stats[network]["inner_boundary_node_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    y = [net_stats[network]["boundary_edge_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    linear_regression(x, y, "Inner boundary node count", "Boundary edge count",
                      "Inner boundary nodes vs. boundary edges", True,
                      output_file=output_dir+"/boundary/inner_boundary_node_boundary_edge_linear.png")

    # Core nodes vs. boundary edges
    print("\nCore node count vs. boundary edge count")
    x = [net_stats[network]["core_node_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    y = [net_stats[network]["boundary_edge_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    linear_regression(x, y, "Core node count", "Boundary edge count",
                      "Core nodes vs. boundary edges", True,
                      output_file=output_dir+"/boundary/core_node_boundary_edge_linear.png")

    # Boundary nodes vs. boundary edges
    print("\nBoundary node count vs. boundary edge count")
    x = [net_stats[network]["boundary_node_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    y = [net_stats[network]["boundary_edge_count"]
         for network in net_stats if network != "Dr_ORG_Heart"]
    linear_regression(x, y, "Boundary node count", "Boundary edge count",
                      "Boundary nodes vs. edges", True,
                      output_file=output_dir+"/boundary/boundary_node_boundary_edge_linear.png")

    # Boundary edges vs. boundary nodes
    print("\nBoundary edge count vs. boundary node count")
    x, y = y, x
    linear_regression(x, y, "Boundary edge count", "Boundary node count",
                      "Boundary edges vs. nodes", True,
                      output_file=output_dir+"/boundary/boundary_edge_boundary_node_linear.png")

    # Boundary weights
    print("\nBoundary weight distribution")
    x = [sum(1 for w in net_stats[network]["boundary_weights"])
         for network in net_stats if network != "Dr_ORG_Heart"]
    y = [sum(1 for w in net_stats[network]["boundary_weights"] if w < 0)
         for network in net_stats if network != "Dr_ORG_Heart"]
    linear_regression(x, y, "Boundary edges", "Boundary negative edges",
                      "Proportion of negative boundary weights", True,
                      output_file=output_dir+"/boundary/boundary_weight_linear.png")
