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

# Inject "Open in Colab" + "View on GitHub" badges at the top of every notebook
# page. nbsphinx renders this Jinja template before each notebook with
# `env.docname` set to the doc name (without extension), letting us build a
# stable Colab/GitHub URL per notebook.
nbsphinx_prolog = r"""
{% set docpath = env.docname %}
{% set gh_url = "https://github.com/hamzamerzic/blox/blob/main/docs/" + docpath + ".ipynb" %}
{% set colab_url = "https://colab.research.google.com/github/hamzamerzic/blox/blob/main/docs/" + docpath + ".ipynb" %}

.. raw:: html

   <p class="colab-badges">
     <a href="{{ colab_url }}" target="_blank" rel="noopener"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
     <a href="{{ gh_url }}" target="_blank" rel="noopener"><img src="https://img.shields.io/badge/View%20source-GitHub-181717?logo=github" alt="View on GitHub"></a>
   </p>
"""

# Pygments still warns once per file when its python lexer sees `!pip` magic
# inside a notebook; nbsphinx falls back to relaxed mode and renders fine.
suppress_warnings = ['misc.highlighting_failure']

templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    # readme_examples is a validation harness for the snippets in README.md,
    # not user-facing docs. Keep it under docs/ for proximity but skip the
    # Sphinx build.
    'readme_examples.ipynb',
    # The installation page has been folded into index.rst.
    'installation.rst',
]

html_theme = 'furo'
html_static_path = ['_static']
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.png'
html_title = 'blox'

# Strip the Sphinx-generated copyright + "Made with Sphinx" line from the
# footer. We hide Furo's "@pradyunsg's Furo" credit via CSS too (see
# custom.css), since the user requested a clean footer.
html_show_copyright = False
html_show_sphinx = False

html_css_files = ['custom.css']

_FONT_STACK = (
    '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, '
    'Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji"'
)
_MONO_STACK = (
    '"JetBrains Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", '
    'monospace'
)

html_theme_options = {
    'sidebar_hide_name': True,
    'navigation_with_keys': True,
    # Drop the view/edit icons — only the theme toggle remains in the
    # content-icon-container, repositioned via CSS so it doesn't push the
    # centered landing H1.
    'top_of_page_buttons': [],
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
    # Minimal palette overrides: keep every Furo default, only swap the
    # brand accent to Tailwind's violet-600 (light) / violet-300 (dark) and
    # the font stack to Inter + JetBrains Mono. Tailwind's palette is a
    # widely-used, well-tested color system; Inter / JetBrains Mono is the
    # pairing used by Vercel, Prisma, Supabase, PostHog, etc.
    'light_css_variables': {
        'font-stack': _FONT_STACK,
        'font-stack--monospace': _MONO_STACK,
        'color-brand-primary': '#7c3aed',   # tailwind violet-600
        'color-brand-content': '#7c3aed',
        'color-brand-visited': '#7c3aed',
    },
    'dark_css_variables': {
        'font-stack': _FONT_STACK,
        'font-stack--monospace': _MONO_STACK,
        'color-brand-primary': '#c4b5fd',   # tailwind violet-300
        'color-brand-content': '#c4b5fd',
        'color-brand-visited': '#c4b5fd',
    },
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
