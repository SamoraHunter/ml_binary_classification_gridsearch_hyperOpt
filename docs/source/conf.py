# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ml-grid"
copyright = "2024, SamoraHunter"
author = "SamoraHunter"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- AutoAPI Configuration ---------------------------------------------------
autoapi_dirs = ["../../ml_grid"]
autoapi_type = "python"
autoapi_template_dir = "_templates/autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "init"


# -- Intersphinx Configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
}
# This is a workaround for SSL certificate verification issues that can occur
# in corporate environments or behind a proxy. It's not recommended for
# production environments outside of a trusted network.
intersphinx_tls_verify = False

# General TLS verification setting for Sphinx
# This helps bypass SSL issues in proxied environments for any web request Sphinx makes.
# Pass certificate: export REQUESTS_CA_BUNDLE=/etc/ssl/certs/
tls_verify = True


# -- Mock Imports ------------------------------------------------------------
# This is a list of modules to be mocked by autodoc. This is useful when
# some external dependencies are not met at build time and break the
# documentation building process.
autodoc_mock_imports = [
    "aeon",  # For time series classifiers
    "catboost",  # ML library
    "gluonts",  # Time series library
    "h2o",  # ML library
    "keras",  # Deep learning framework
    "keras_self_attention",  # Keras extension
    "lightgbm",  # ML library
    "numba",  # A key dependency for aeon
    "pmdarima",  # For ARIMA models
    "prophet",  # Time series library
    "scikeras",  # Scikit-learn wrapper for Keras
    "sktime",  # In case of legacy time series imports
    "tensorflow",  # Deep learning framework
    "tensorflow_probability",  # TF extension
    "torch",  # Deep learning framework
    "tsfresh",  # Time series feature extraction
    "xgboost",  # ML library
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
