# PerturbationX
PerturbationX is a package for analyzing causal networks in combination with gene expression data. It is based on the [TopoNPA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4227138/) algorithm. It was developed as part of a Master's thesis at the University of Ljubljana, Faculty of Computer and Information Science and in collaboration with the National Institute of Biology.

## Installation
The package can be installed from PyPI or directly from GitHub. It requires a Python version of 3.10 or newer. It is based on NetworkX and pandas and requires Cytoscape for visualization. The latter can be downloaded from [here](https://cytoscape.org/download.html).

```bash
python -m pip install perturbationx # PyPI
python -m pip install git+https://github.com/mikethenut/perturbationx # GitHub
```

## Usage
An example Jupyter notebook [example.ipynb](https://github.com/mikethenut/perturbationx/blob/main/example.ipynb) is available for step-by-step instructions on how to use the package. For advanced usage, refer to the [documentation](https://mikethenut.github.io/perturbationx/index.html) and [Master's thesis](https://github.com/mikethenut/perturbationx/blob/main/Integration%20of%20gene%20expression%20data%20with%20causal%20networks.pdf).

## Citation
No paper describing the package has been published yet. If you use this package in your research, please cite the following Zenodo DOI:

[![DOI](https://zenodo.org/badge/580879301.svg)](https://zenodo.org/doi/10.5281/zenodo.10073529)

```bibtex
@software{rajh2023perturbationx,
  author       = {Rajh, Mihael and Bleker, Carissa Robyn and Curk, Tomaž},
  title        = {PerturbationX},
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.10073529},
  url          = {https://doi.org/10.5281/zenodo.10073529}
}
```
