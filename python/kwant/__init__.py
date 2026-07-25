"""Kwant 1.5 compatibility layer backed by the Thouless Rust core."""

import importlib
import sys

from . import (
    _plotter,
    system,
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
from ._common import KwantDeprecationWarning
from .builder import Builder, HoppingKind, Site, SiteFamily, UserCodeError
from .lattice import TranslationalSymmetry
from .plotter import plot
from .solvers import (
    GreensFunction,
    SMatrix,
    greens_function,
    ldos,
    smatrix,
    wave_function,
)

__version__ = "1.5.0+thouless"


def test(verbose=True):
    """Run tests installed alongside the Kwant compatibility package."""
    from pathlib import Path

    import pytest

    arguments = [str(Path(__file__).resolve().parent), "-s"]
    if verbose:
        arguments.append("-v")
    return pytest.main(arguments)


test.__test__ = False


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
physics.dispersion = physics
physics.noise = physics
physics.symmetry = physics
sys.modules[f"{__name__}.physics.dispersion"] = physics
sys.modules[f"{__name__}.physics.noise"] = physics
sys.modules[f"{__name__}.physics.symmetry"] = physics
physics.gauge = gauge
physics.magnetic_gauge = gauge.magnetic_gauge
if "magnetic_gauge" not in physics.__all__:
    physics.__all__.append("magnetic_gauge")
sys.modules[f"{__name__}.physics.gauge"] = gauge

__all__ = [
    "Builder",
    "HoppingKind",
    "KwantDeprecationWarning",
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
    "plot",
    "plotter",
    "rmt",
    "greens_function",
    "ldos",
    "solvers",
    "system",
    "test",
    "smatrix",
    "wave_function",
    "wraparound",
]
