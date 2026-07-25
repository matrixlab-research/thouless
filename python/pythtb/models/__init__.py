"""Canonical tight-binding model constructors for compatibility workflows."""

from __future__ import annotations

import numpy as np

from ..lattice import Lattice
from ..tbmodel import TBModel


def checkerboard(delta, t):
    """Construct the two-sublattice square-lattice checkerboard model."""
    lattice = Lattice(
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0], [0.5, 0.5]],
        periodic_dirs=[0, 1],
    )
    model = TBModel(lattice)
    model.set_onsite([-delta, delta])
    for offset in ([0, 0], [1, 0], [0, 1], [1, 1]):
        model.set_hop(t, 1, 0, offset)
    return model


def graphene(delta, t):
    """Construct the nearest-neighbor two-band graphene model."""
    lattice = Lattice(
        [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]],
        [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]],
        periodic_dirs=[0, 1],
    )
    model = TBModel(lattice)
    model.set_onsite([-delta, delta])
    model.set_hop(t, 0, 1, [0, 0])
    model.set_hop(t, 1, 0, [1, 0])
    model.set_hop(t, 1, 0, [0, 1])
    return model


def ssh(v, w):
    """Construct the dimerized Su-Schrieffer-Heeger chain."""
    lattice = Lattice(
        [[1.0]],
        [[0.0], [0.5]],
        periodic_dirs=[0],
    )
    model = TBModel(lattice)
    model.set_hop(v, 0, 1, [0])
    model.set_hop(w, 1, 0, [1])
    return model


def fu_kane_mele(t, soc, dt=(0.0, 0.0, 0.0, 0.0)):
    """Construct the spinful Fu-Kane-Mele model on a diamond lattice."""
    if len(dt) != 4:
        raise ValueError("dt must contain four nearest-neighbor offsets")
    lattice = Lattice(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        periodic_dirs=[0, 1, 2],
    )
    model = TBModel(lattice, spinful=True)
    for offset, correction in zip(
        ([0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]),
        dt,
        strict=True,
    ):
        model.set_hop(t + correction, 0, 1, offset)
    offsets = (
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 1, 0],
        [0, -1, 1],
        [1, 0, -1],
    )
    directions = (
        [0, 1, -1],
        [-1, 0, 1],
        [1, -1, 0],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    )
    for offset, direction in zip(offsets, directions, strict=True):
        spin_hopping = 1j * soc * np.array([0.0, *direction])
        model.set_hop(spin_hopping, 0, 0, offset)
        model.set_hop(-spin_hopping, 1, 1, offset)
    return model


def haldane(delta, t1, t2, phi=np.pi / 2):
    """Construct the two-band Haldane model on a honeycomb lattice."""
    lattice = Lattice(
        [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]],
        [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]],
        periodic_dirs=[0, 1],
    )
    model = TBModel(lattice)
    model.set_onsite([-delta, delta])
    for offset in ([0, 0], [-1, 0], [0, -1]):
        model.set_hop(t1, 0, 1, offset)
    complex_hopping = t2 * np.exp(1j * phi)
    for offset in ([1, 0], [-1, 1], [0, -1]):
        model.set_hop(complex_hopping, 0, 0, offset)
    for offset in ([-1, 0], [1, -1], [0, 1]):
        model.set_hop(complex_hopping, 1, 1, offset)
    return model


def kane_mele(delta, t, soc, rashba):
    """Construct the four-band Kane-Mele model."""
    lattice = Lattice(
        [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]],
        [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]],
        periodic_dirs=[0, 1],
    )
    model = TBModel(lattice, spinful=True)
    model.set_onsite([delta, -delta])
    sigma_x = np.array([0.0, 1.0, 0.0, 0.0])
    sigma_y = np.array([0.0, 0.0, 1.0, 0.0])
    sigma_z = np.array([0.0, 0.0, 0.0, 1.0])
    bonds = ([0, 0], [0, -1], [-1, 0])
    for offset in bonds:
        model.set_hop(t, 0, 1, offset)
    intrinsic = 1j * soc * sigma_z
    for sign, orbital in ((-1, 0), (1, 1)):
        model.set_hop(sign * intrinsic, orbital, orbital, [0, 1])
        model.set_hop(-sign * intrinsic, orbital, orbital, [1, 0])
        model.set_hop(sign * intrinsic, orbital, orbital, [1, -1])
    rashba_terms = (
        1j * rashba * (0.5 * sigma_x - np.sqrt(3.0) * sigma_y / 2.0),
        -1j * rashba * sigma_x,
        1j * rashba * (0.5 * sigma_x + np.sqrt(3.0) * sigma_y / 2.0),
    )
    for offset, term in zip(bonds, rashba_terms, strict=True):
        model.set_hop(term, 0, 1, offset, mode="add")
    return model


__all__ = [
    "checkerboard",
    "fu_kane_mele",
    "graphene",
    "haldane",
    "kane_mele",
    "ssh",
]

# Import the public constructor modules once during package initialization.
# This preserves both ``pythtb.models.ssh(...)`` and
# ``from pythtb.models.ssh import ssh`` without letting Python replace the
# package-level constructor with the submodule object on first use.
from .checkerboard import checkerboard as checkerboard
from .fu_kane_mele import fu_kane_mele as fu_kane_mele
from .graphene import graphene as graphene
from .haldane import haldane as haldane
from .kane_mele import kane_mele as kane_mele
from .ssh import ssh as ssh
