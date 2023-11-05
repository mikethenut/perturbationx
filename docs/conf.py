# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PerturbationX'
copyright = 'CC BY-NC 4.0'
author = 'Mihael Rajh'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autoapi.extension',
    'sphinx.ext.coverage'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Configuration of extensions
coverage_show_missing_items = True
autoapi_dirs = ['../perturbationx']
autoapi_template_dir = '_templates/_autoapi'
autoapi_generate_api_docs = False

# Build instructions
# 1.1 Comment out `autoapi_generate_api_docs = False`
# 1.2 Run `sphinx-build -b html . _build`
# 1.3 Fix any warnings unrelated to duplicates and rerun
# 2.1 `autoapi_keep_files = True` doesn't work (but try it anyway)
# 2.2 Go to venv/Lib/autoapi/extension.py and comment out `shutil.rmtree(normalized_root)` in build_finished
# 2.3 Run `sphinx-build -b html . _build` once and revert change in autoapi source
# 3.1 Uncomment `autoapi_generate_api_docs = False`
# 3.2 Run `sphinx-build -b html -E . _build`
# 3.3 Fix any warnings related to duplicates and rerun
# 4.1 For some reason this re-adds data and attribute members to the documentation
#     Search for ::py:attribute::, ::py:data::, Attribute (case-sensitive) to find and remove them
# 4.2 Make sure to also check for correct references in function tables
# 4.3 Run `sphinx-build -b html -E . _build` once more
