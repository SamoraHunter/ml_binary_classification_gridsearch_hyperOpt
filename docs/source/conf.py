# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ml-grid'
copyright = '2024, SamoraHunter'
author = 'SamoraHunter'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'autoapi.extension',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
]



templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- AutoAPI Configuration ---------------------------------------------------
autoapi_dirs = ['../../ml_grid']
autoapi_type = 'python'
autoapi_template_dir = '_templates/autoapi'
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]
autoapi_python_class_content = 'init'


# -- Intersphinx Configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
}
# This is a workaround for SSL certificate verification issues that can occur
# in corporate environments or behind a proxy. It's not recommended for
# production environments outside of a trusted network.
intersphinx_tls_verify = False

# General TLS verification setting for Sphinx
# This helps bypass SSL issues in proxied environments for any web request Sphinx makes.
tls_verify = False


# -- Mock Imports ------------------------------------------------------------
# This is a list of modules to be mocked by autodoc. This is useful when
# some external dependencies are not met at build time and break the
# documentation building process.
autodoc_mock_imports = [
    "aeon",
    "tensorflow",
    "keras",
    "scikeras",
    "tsfresh",
    "tensorflow_probability",
    "keras_self_attention",
    "pmdarima",
    "prophet",
    "gluonts",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']