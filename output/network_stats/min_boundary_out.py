import json

with open("network_stats.json") as f:
    net_stats = json.load(f)
    for net in net_stats:
        boundary_out_degrees = net_stats[net]["boundary_out_degrees"]
        boundary_out_degrees = [int(x) for x in boundary_out_degrees if int(x) != 0]
        min_boundary_out = min(boundary_out_degrees)
        print(net, min_boundary_out)
