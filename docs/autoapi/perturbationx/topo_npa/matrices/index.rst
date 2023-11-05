:py:mod:`perturbationx.topo_npa.matrices`
=========================================

.. py:module:: perturbationx.topo_npa.matrices


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.toponpa.matrices.generate_adjacencies
   perturbationx.toponpa.matrices.generate_boundary_laplacian
   perturbationx.toponpa.matrices.generate_core_laplacians



.. py:function:: generate_adjacencies(graph: networkx.DiGraph, directed=False, sparse=True)

   Generate the boundary and core adjacency matrices from a graph.

   :param graph: The graph.
   :type graph: nx.DiGraph
   :param directed: Whether to generate directed adjacency matrices. Defaults to False.
   :type directed: bool, optional
   :param sparse: Whether to generate sparse adjacency matrices. Defaults to True.
   :type sparse: bool, optional
   :return: The boundary and core adjacency matrices.
   :rtype: (np.ndarray, np.ndarray) | (sp.sparray, sp.sparray)


.. py:function:: generate_boundary_laplacian(adj_b: numpy.ndarray | scipy.sparse.sparray, boundary_edge_minimum=6)

   Generate the boundary Lb Laplacian from a boundary adjacency matrix.

   :param adj_b: The boundary adjacency matrix.
   :type adj_b: np.ndarray | sp.sparray
   :param boundary_edge_minimum: The minimum number of boundary edges a core node must have to be
                                   included in the Lb Laplacian. Nodes with fewer boundary edges
                                   are removed from the Lb Laplacian. Defaults to 6.
   :type boundary_edge_minimum: int, optional
   :raises ValueError: If the adjacency matrix is misshapen or if the boundary edge minimum is negative.
   :return: The boundary Lb Laplacian.
   :rtype: np.ndarray | sp.sparray


.. py:function:: generate_core_laplacians(lap_b: numpy.ndarray | scipy.sparse.sparray, adj_c: numpy.ndarray | scipy.sparse.sparray, exact_boundary_outdegree=True)

   Generate the core Laplacians from a boundary Laplacian and core adjacency matrix.

   :param lap_b: The boundary Laplacian.
   :type lap_b: np.ndarray | sp.sparray
   :param adj_c: The core adjacency matrix.
   :type adj_c: np.ndarray | sp.sparray
   :param exact_boundary_outdegree: Whether to use the exact boundary outdegree. If False, the boundary outdegree
                                       is set to 1 for all core nodes with boundary edges. Defaults to True.
   :type exact_boundary_outdegree: bool, optional
   :return: The core Laplacians.
   :rtype: (np.ndarray, np.ndarray) | (sp.sparray, sp.sparray)


