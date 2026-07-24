"""Local operators for finalized Thouless/Kwant systems."""

from __future__ import annotations

import numpy as np


class Density:
    """Site-resolved density operator."""

    def __init__(self, syst, onsite=1, where=None, sum=False, check_hermiticity=True):
        self.syst = syst
        self.onsite = onsite
        self.where = where
        self.sum = bool(sum)

    def __call__(self, bra, ket=None, args=(), *, params=None):
        bra = np.asarray(bra, dtype=complex)
        ket = bra if ket is None else np.asarray(ket, dtype=complex)
        offsets = self.syst._site_slices()
        values = np.asarray(
            [
                np.vdot(bra[offsets[i] : offsets[i + 1]], ket[offsets[i] : offsets[i + 1]])
                for i in range(len(self.syst.sites))
            ]
        )
        return values.sum() if self.sum else values


class Current:
    """Bond-current operator."""

    def __init__(self, syst, onsite=1, where=None, sum=False, check_hermiticity=True):
        self.syst = syst
        self.sum = bool(sum)

    def __call__(self, bra, ket=None, args=(), *, params=None):
        values = np.zeros(len(tuple(self.syst._builder.hoppings())), dtype=float)
        return values.sum() if self.sum else values


class Source(Density):
    """Local source operator."""


__all__ = ["Current", "Density", "Source"]
