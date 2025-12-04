# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Configuration file for the Sphinx documentation builder.

import os
import sys
from pathlib import Path

# Add EMBO package to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


project = 'EMBO'
#copyright = '2025, Subrata Mukherjee'
author = "Subrata Mukherjee, Kris Villez"
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ 
    "myst_parser",               # Markdown support
    "sphinx.ext.autodoc",        # Auto API documentation
    "sphinx.ext.napoleon",       # Google-style docstrings
    "sphinx.ext.autosummary",    # Generate API summary pages
]

autosummary_generate = True      # Generate modules.rst automatically
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
