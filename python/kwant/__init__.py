"""Kwant 1.5 compatibility layer backed by the Thouless Rust core."""

import sys

from . import builder, digest, graph, kpm, lattice, linalg, operator, physics, rmt, solvers
from .builder import Builder, HoppingKind, Site, SiteFamily, UserCodeError
from .lattice import TranslationalSymmetry
from .solvers import (
    GreensFunction,
    SMatrix,
    greens_function,
    ldos,
    smatrix,
    wave_function,
)

__version__ = "1.5.0+thouless"

# Kwant exposes the lead routines as ``kwant.physics.leads``. The compatibility
# module is intentionally a single thin boundary, so the package-style name
# resolves to the same object.
physics.leads = physics
sys.modules[f"{__name__}.physics.leads"] = physics

__all__ = [
    "Builder",
    "HoppingKind",
    "GreensFunction",
    "SMatrix",
    "Site",
    "SiteFamily",
    "TranslationalSymmetry",
    "UserCodeError",
    "builder",
    "digest",
    "graph",
    "kpm",
    "lattice",
    "linalg",
    "operator",
    "physics",
    "rmt",
    "greens_function",
    "ldos",
    "solvers",
    "smatrix",
    "wave_function",
]
