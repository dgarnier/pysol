"""Sphinx configuration."""
project = "Inductance"
author = "Darren Garnier"
copyright = "2023, Darren Garnier"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
