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
    'sphinx_copybutton',
    'nbsphinx',
]

# Don't execute notebooks during build (some require Colab/GPU).
nbsphinx_execute = 'never'

# Use the IPython lexer so cell magics (e.g. `!pip install`) highlight cleanly.
nbsphinx_codecell_lexer = 'ipython3'

# Pygments still warns once per file when its python lexer sees `!pip` magic
# inside a notebook; nbsphinx falls back to relaxed mode and renders fine.
suppress_warnings = ['misc.highlighting_failure']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

html_theme = 'furo'
html_static_path = ['_static']
html_logo = '_static/logo.png'
html_favicon = '_static/logo.png'
html_title = 'blox'

html_theme_options = {
    'sidebar_hide_name': True,
    'navigation_with_keys': True,
    'source_repository': 'https://github.com/hamzamerzic/blox/',
    'source_branch': 'main',
    'source_directory': 'docs/',
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/hamzamerzic/blox',
            'html': '',
            'class': 'fa-brands fa-github',
        },
    ],
}

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

# sphinx-copybutton: skip prompts and outputs when copying code.
copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: '
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False
