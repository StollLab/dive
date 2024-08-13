# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from numpy.typing import ArrayLike
from matplotlib.typing import ColorType
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dive'
copyright = '2024, Sarah Sweger, Julian Cheung, Lukas Zha, Stephan Pribitzer, Stefan Stoll'
author = 'Sarah Sweger, Julian Cheung, Lukas Zha, Stephan Pribitzer, Stefan Stoll'
release = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = []
# autodoc_mock_imports = ['pymc','deerlab','scipy','matplotlib','pandas','numpy']
autodoc_type_aliases = {ArrayLike: 'ArrayLike', ColorType: 'ColorType'}

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_type_aliases = {ArrayLike: 'ArrayLike', ColorType: 'ColorType'}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
