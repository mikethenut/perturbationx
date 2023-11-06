:py:mod:`perturbationx.toponpa.core`
====================================

.. py:module:: perturbationx.toponpa.core


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.toponpa.core.coefficient_inference
   perturbationx.toponpa.core.perturbation_amplitude
   perturbationx.toponpa.core.perturbation_amplitude_contributions



.. py:function:: coefficient_inference(lap_b: numpy.ndarray | scipy.sparse.sparray, lap_c: numpy.ndarray | scipy.sparse.sparray, boundary_coefficients: numpy.ndarray)

   Infer core coefficients from boundary coefficients and Laplacian matrices.

   :param lap_b: The Lb boundary Laplacian.
   :type lap_b: np.ndarray | sp.sparray
   :param lap_c: The Lc core Laplacian.
   :type lap_c: np.ndarray | sp.sparray
   :param boundary_coefficients: The boundary coefficients.
   :type boundary_coefficients: np.ndarray
   :raises ValueError: If the Laplacian matrices are misshapen or if the matrix dimensions do not match.
   :return: The inferred core coefficients.
   :rtype: np.ndarray


.. py:function:: perturbation_amplitude(lap_q: numpy.ndarray | scipy.sparse.sparray, core_coefficients: numpy.ndarray, core_edge_count: int)

   Compute the perturbation amplitude from the core Laplacian and core coefficients.

   :param lap_q: The Q core Laplacian.
   :type lap_q: np.ndarray | sp.sparray
   :param core_coefficients: The core coefficients.
   :type core_coefficients: np.ndarray
   :param core_edge_count: The number of edges in the core network.
   :type core_edge_count: int
   :raises ValueError: If the Laplacian matrix is misshapen or if the matrix dimensions do not match.
   :return: The perturbation amplitude.
   :rtype: np.ndarray


.. py:function:: perturbation_amplitude_contributions(lap_q: numpy.ndarray | scipy.sparse.sparray, core_coefficients: numpy.ndarray, core_edge_count: int)

   Compute the perturbation amplitude and relative contributions from the core Laplacian and core coefficients.

   :param lap_q: The Q core Laplacian.
   :type lap_q: np.ndarray | sp.sparray
   :param core_coefficients: The core coefficients.
   :type core_coefficients: np.ndarray
   :param core_edge_count: The number of edges in the core network.
   :type core_edge_count: int
   :raises ValueError: If the Laplacian matrix is misshapen or if the matrix dimensions do not match.
   :return: The perturbation amplitude and relative contributions.
   :rtype: (np.ndarray, np.ndarray)


