# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path("../../python/swiflow").resolve()

sys.path.insert(0, str(ROOT_DIR))

project = "swiflow"
copyright = "2024, TeamGraphix"  # noqa: A001
author = "S.S."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
# exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# html_static_path = []

intersphinx_mapping = {
    "networkx": ("https://networkx.org/documentation/stable", None),
    "python": ("https://docs.python.org/3", None),
}

default_role = "any"

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
bibtex_bibfiles = ["ref.bib"]
