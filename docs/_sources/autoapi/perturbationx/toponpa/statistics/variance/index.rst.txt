:py:mod:`perturbationx.toponpa.statistics.variance`
===================================================

.. py:module:: perturbationx.toponpa.statistics.variance


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.toponpa.statistics.variance.core_covariance_matrix
   perturbationx.toponpa.statistics.variance.perturbation_variance



.. py:function:: core_covariance_matrix(lap_b: numpy.ndarray | scipy.sparse.sparray, lap_c: numpy.ndarray | scipy.sparse.sparray, stderr: numpy.ndarray)

   Compute the covariance matrix of the core node coefficients.

   :param lap_b: The Lb boundary Laplacian.
   :type lap_b: np.ndarray | sp.sparray
   :param lap_c: The Lc core Laplacian.
   :type lap_c: np.ndarray | sp.sparray
   :param stderr: The standard error of the boundary coefficients.
   :type stderr: np.ndarray
   :return: The covariance matrix of the core node coefficients.
   :rtype: np.ndarray


.. py:function:: perturbation_variance(lap_q: numpy.ndarray | scipy.sparse.sparray, core_coefficients: numpy.ndarray, core_covariance: numpy.ndarray, core_edge_count: int)

   Compute the variance of the perturbation score.

   :param lap_q: The Q core Laplacian.
   :type lap_q: np.ndarray | sp.sparray
   :param core_coefficients: The core node coefficients.
   :type core_coefficients: np.ndarray
   :param core_covariance: The covariance matrix of the core node coefficients.
   :type core_covariance: np.ndarray
   :param core_edge_count: The number of edges in the core network.
   :type core_edge_count: int
   :return: The variance of the perturbation score.
   :rtype: float


