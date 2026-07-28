"""Sphinx configuration for the generated Thouless Python reference."""

from __future__ import annotations

from importlib.metadata import version as package_version

project = "Thouless Python API"
author = "Matrix Lab"
release = package_version("thouless")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "furo"
html_title = f"Thouless Python API {release}"
html_static_path: list[str] = []
