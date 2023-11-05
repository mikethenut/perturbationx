:py:mod:`perturbationx.resources`
=================================

.. py:module:: perturbationx.resources


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   resources/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   perturbationx.resources.get_style_xml_path
   perturbationx.resources.get_osmotic_stress_core_path
   perturbationx.resources.get_osmotic_stress_boundary_path
   perturbationx.resources.get_osmotic_stress_dataset_paths

   

.. py:function:: get_style_xml_path()

   Get the path to the default style xml file.
   :return: The path to the default style xml file.
   :rtype: str


.. py:function:: get_osmotic_stress_core_path()

   Get the path to the osmotic stress core network file. The network is a subset of the human osmotic stress network
   from the Causal Biological Networks database (https://www.causalbionet.com/, network ID osmotic_stress_2.0_hs).
   All non-causal edges were removed and the largest connected component was then selected.

   :return: The path to the osmotic stress core network file.
   :rtype: str


.. py:function:: get_osmotic_stress_boundary_path()

   Get the path to the osmotic stress boundary network file. The network was generated using the Barabási-Albert
   model and is not biological in nature. It contains 100 outer boundary nodes and 500 boundary edges.

   :return: The path to the osmotic stress boundary network file.
   :rtype: str


.. py:function:: get_osmotic_stress_dataset_paths()

   Get the paths to the osmotic stress dataset files. The datasets were generated using a genetic algorithm and
   are not biological in nature.

   :return: The paths to the osmotic stress dataset files.
   :rtype: list


