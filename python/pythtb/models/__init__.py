"""Canonical tight-binding model constructors for compatibility workflows."""

from __future__ import annotations

import numpy as np

from ..lattice import Lattice
from ..tbmodel import TBModel


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


__all__ = ["haldane", "kane_mele"]
