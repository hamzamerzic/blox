import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

project = 'blox'
copyright = '2025, Hamza Merzić'
author = 'Hamza Merzić'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

# Don't execute notebooks during build (some require Colab/GPU).
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'  # Default theme, easy to swap
