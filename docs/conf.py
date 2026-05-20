import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

project = 'blox'
copyright = '2026, Hamza Merzić'
author = 'Hamza Merzić'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

# Don't execute notebooks during build (some require Colab/GPU).
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

html_theme = 'furo'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://docs.jax.dev/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
