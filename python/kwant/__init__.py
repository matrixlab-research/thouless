"""Kwant 1.5 compatibility layer backed by the Thouless Rust core."""

import importlib
import sys

from . import (
    _plotter,
    builder,
    digest,
    gauge,
    graph,
    kpm,
    lattice,
    linalg,
    operator,
    physics,
    plotter,
    rmt,
    solvers,
    wraparound,
)
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


def __getattr__(name):
    if name == "continuum":
        module = importlib.import_module(".continuum", __name__)
        globals()[name] = module
        return module
    raise AttributeError(name)

# Kwant exposes the lead routines as ``kwant.physics.leads``. The compatibility
# module is intentionally a single thin boundary, so the package-style name
# resolves to the same object.
physics.leads = physics
sys.modules[f"{__name__}.physics.leads"] = physics
physics.gauge = gauge
physics.magnetic_gauge = gauge.magnetic_gauge
sys.modules[f"{__name__}.physics.gauge"] = gauge

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
    "continuum",
    "digest",
    "gauge",
    "graph",
    "kpm",
    "lattice",
    "linalg",
    "operator",
    "physics",
    "plotter",
    "rmt",
    "greens_function",
    "ldos",
    "solvers",
    "smatrix",
    "wave_function",
    "wraparound",
]
