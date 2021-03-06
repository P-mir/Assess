# https://www.sphinx-doc.org/en/master/usage/configuration.html
# To convert markdown to rst: https://github.com/CrossNox/m2r2

######## How to using Anaconda

#in anaconda prompt:
# sphinx-apidoc -f -o docs/source src/utils  #repeat as much time as there is folder to generate all the rst files
# docs: make html

########

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:


# -- Path setup --------------------------------------------------------------
# https://www.tjelvarolsson.com/blog/how-to-generate-beautiful-technical-documentation/
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../../src"))
target_dir = os.path.abspath(os.path.join(current_dir, "../../src/utils"))

sys.path.insert(0, target_dir)
print(target_dir)

# -- Project information -----------------------------------------------------

project = 'ASSESS'
copyright = '2021, Patrick Guerin'
author = 'Patrick Guerin'

# The full version, including alpha/beta/rc tags
release = 'v1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


