

class NPAResult:
    def __init__(self, npa, npa_variance, npa_confidence_interval, node_contributions,
                 node_coefficients, node_variance, node_confidence_interval, node_p_value,
                 o_value, o_distribution, k_value, k_distribution):
        self.npa = npa
        self.npa_variance = npa_variance
        self.npa_confidence_interval = npa_confidence_interval
        self.o_value = o_value
        self.k_value = k_value

        self.node_contributions = node_contributions
        self.node_coefficients = node_coefficients
        self.node_variance = node_variance
        self.node_confidence_interval = node_confidence_interval
        self.node_p_value = node_p_value
        self.o_distribution = o_distribution
        self.k_distribution = k_distribution
