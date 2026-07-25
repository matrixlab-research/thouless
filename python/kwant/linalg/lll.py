"""Lattice reduction and closest-vector compatibility over the Rust core."""

from __future__ import annotations

import numpy as np

from thouless import _core


def gs_coefficient(vector, reference):
    """Return one Gram--Schmidt projection coefficient."""
    vector = np.asarray(vector, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return _core.lattice_gs_coefficient(
        vector.tolist(),
        reference.tolist(),
    )


def gs(basis):
    """Return unnormalized row-wise Gram--Schmidt vectors."""
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError('"basis" must be a 2d array-like object.')
    return np.asarray(
        _core.lattice_gram_schmidt(basis.tolist()),
        dtype=float,
    )


def is_c_reduced(basis, c):
    """Return whether a basis satisfies the requested reduction bound."""
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError('"basis" must be a 2d array-like object.')
    return _core.lattice_is_c_reduced(basis.tolist(), float(c))


def lll(basis, c=1.34):
    """Return an LLL-reduced basis and its exact integer transformation."""
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError('"basis" must be a 2d array-like object.')
    vectors, transformation = _core.lattice_lll(
        basis.tolist(),
        float(c),
    )
    return (
        np.asarray(vectors, dtype=float),
        np.asarray(transformation, dtype=int),
    )


def cvp(vec, basis, n=1, group_by_length=False, rtol=1e-9):
    """Return coefficients of the closest lattice vectors to ``vec``."""
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError("`basis` must be a 2d array-like object.")
    vec = np.asarray(vec, dtype=float)
    return np.asarray(
        _core.lattice_cvp(
            vec.tolist(),
            basis.tolist(),
            int(n),
            bool(group_by_length),
            float(rtol),
        ),
        dtype=int,
    )


def voronoi(basis, reduced=False, rtol=1e-9):
    """Return integer vectors whose bisectors bound a lattice Voronoi cell."""
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError("`basis` must be a 2d array-like object.")
    return np.asarray(
        _core.lattice_voronoi(
            basis.tolist(),
            bool(reduced),
            float(rtol),
        ),
        dtype=int,
    )


__all__ = [
    "cvp",
    "gs",
    "gs_coefficient",
    "is_c_reduced",
    "lll",
    "voronoi",
]
