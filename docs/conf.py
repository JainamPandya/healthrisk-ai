import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'HealthRisk AI'
copyright = '2026, Zetheta Intern'
author = 'Zetheta Intern'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']
