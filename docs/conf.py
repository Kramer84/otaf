import os
import sys

# Add your project's `src` directory to the path
sys.path.insert(0, os.path.abspath('../src'))

project = 'otaf'
copyright = '2024, Kramer84'
author = 'Kramer84'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Extensions
extensions = [
    'sphinx.ext.autodoc',             # Automatically documents docstrings
    'sphinx.ext.napoleon',            # Supports Google and NumPy-style docstrings
    'sphinx_autodoc_typehints',       # Adds type hints to documentation
    'sphinx.ext.viewcode',            # Adds links to source code
    'myst_parser',
    'nbsphinx',
]

# HTML Theme
html_theme = 'sphinx_rtd_theme'  # Replace 'sphinx_rtd_theme' with 'furo'
html_theme_options = {
    #"style_external_links": True,  # Ensure external links match the theme
}
# Use custom CSS if needed
html_static_path = ['_static']
#html_css_files = ['custom.css']

autodoc_default_options = {
    'members': True,         # Include class and module members
    'undoc-members': False,  # Exclude undocumented members
    'show-inheritance': True, # Show class inheritance
    "imported-members": True,  # Include re-exported members
    'special-members': '__all__',  # Include constants explicitly
}
autodoc_typehints = "description"  # Show type hints as part of parameter descriptions

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False  # Exclude private methods
napoleon_include_special_with_doc = True  # Exclude `__init__`, etc.


myst_enable_extensions = [
    "deflist",  # Definition lists
    "linkify",  # Auto-detect and convert URLs into links
    "colon_fence",  # Enable ::: for block elements like admonitions
    "replacements",
]
myst_heading_anchors = 3  # Automatically generate anchors for headings up to level 3

nbsphinx_allow_errors = True  # Allow notebooks to load without execution errors
nbsphinx_execute = 'never'   # Prevent execution of notebooks
# nbsphinx_input_prompt = ''   # Remove input prompts
# nbsphinx_output_prompt = ''  # Remove output prompts

nbsphinx_prolog = """
.. raw:: html

   <style>
      .toctree-wrapper .tocsection { display: none; }
   </style>
"""

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
