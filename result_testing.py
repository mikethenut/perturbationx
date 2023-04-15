import numpy as np
from bnpa.output.NPAResultBuilder import NPAResultBuilder


if __name__ == "__main__":
    my_builder = NPAResultBuilder.new_builder(['data2', 'data1'], ['n1', 'n2', 'n4', 'n5', 'n3'])
    my_builder.set_node_attributes('data1', ['attr1', 'attr2', 'attr3'], np.random.rand(3, 5))
    my_builder.set_node_attributes('data2', ['attr1', 'attr3'], np.random.rand(2, 5))
    my_builder.set_distribution('data1', 'random', np.random.rand(1000), np.random.rand())
    my_builder.set_distribution('data2', 'random', np.random.rand(1000), np.random.rand())

    res = my_builder.build()
    print(res.node_info("data1"))
    print(res.node_info("attr1"))
    print(res.distributions())
    res.plot_distribution('random')
