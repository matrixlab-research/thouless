"""Kwant random-matrix compatibility over the Thouless Rust core."""

from __future__ import annotations

import numpy as np

from thouless import _core

from ._common import ensure_rng


sym_list = ("A", "AI", "AII", "AIII", "BDI", "CII", "D", "DIII", "C", "CI")

h_t_matrix = {
    "AI": [[1]],
    "CI": [[0, 1], [1, 0]],
    "BDI": [[1, 0], [0, -1]],
    "AII": [[0, 1j], [-1j, 0]],
    "CII": [
        [0, 0, 1j, 0],
        [0, 0, 0, 1j],
        [-1j, 0, 0, 0],
        [0, -1j, 0, 0],
    ],
    "DIII": [[0, 1j], [-1j, 0]],
}

h_p_matrix = {
    "C": [[0, 1j], [-1j, 0]],
    "CI": [[0, 1j], [-1j, 0]],
    "CII": [
        [0, 0, 1j, 0],
        [0, 0, 0, -1j],
        [-1j, 0, 0, 0],
        [0, 1j, 0, 0],
    ],
    "D": [[1]],
    "DIII": [[0, 1], [1, 0]],
    "BDI": [[1]],
}


def _check_symmetry(symmetry):
    if symmetry not in sym_list:
        raise ValueError("Non-existent symmetry class.")


def t(symmetry):
    """Return the square of the time-reversal operation."""
    _check_symmetry(symmetry)
    if symmetry in ("CI", "AI", "BDI"):
        return 1
    if symmetry in ("CII", "AII", "DIII"):
        return -1
    return 0


def p(symmetry):
    """Return the square of the particle-hole operation."""
    _check_symmetry(symmetry)
    if symmetry in ("D", "DIII", "BDI"):
        return 1
    if symmetry in ("C", "CI", "CII"):
        return -1
    return 0


def c(symmetry):
    """Return whether the class has chiral symmetry."""
    _check_symmetry(symmetry)
    return int(symmetry == "AIII" or (t(symmetry) != 0 and p(symmetry) != 0))


def _normal(rng, shape):
    if hasattr(rng, "standard_normal"):
        return rng.standard_normal(shape)
    return rng.randn(*shape)


def _binary(rng, size):
    if hasattr(rng, "integers"):
        return rng.integers(2, size=size)
    return rng.randint(2, size=size)


def _components(n, symmetry, rng, *, circular):
    real = _normal(rng, (n, n))
    needs_imaginary = p(symmetry) != 1 if circular else symmetry not in (
        "AI",
        "D",
        "BDI",
    )
    imaginary = (
        _normal(rng, (n, n))
        if needs_imaginary
        else np.zeros((n, n), dtype=float)
    )
    return real.ravel().tolist(), imaginary.ravel().tolist()


def gaussian(n, sym="A", v=1.0, rng=None):
    """Draw an ``n`` by ``n`` Gaussian Hamiltonian in symmetry class ``sym``."""
    _check_symmetry(sym)
    n = int(n)
    if n < 0:
        raise ValueError("Matrix dimension must be non-negative.")
    if (c(sym) or t(sym) == -1 or p(sym) == -1) and n % 2:
        raise ValueError("Matrix dimension should be even in chosen symmetry class.")
    if sym == "CII" and n % 4:
        raise ValueError(
            "Matrix dimension should be a multiple of 4 in symmetry class CII."
        )
    rng = ensure_rng(rng)
    real, imaginary = _components(n, sym, rng, circular=False)
    return np.asarray(
        _core.rmt_gaussian(n, sym, float(v), real, imaginary),
        dtype=complex,
    )


def circular(n, sym="A", charge=None, rng=None):
    """Draw an ``n`` by ``n`` matrix from a circular symmetry ensemble."""
    _check_symmetry(sym)
    n = int(n)
    if n < 0:
        raise ValueError("Matrix dimension must be non-negative.")
    if (t(sym) == -1 or p(sym) == -1) and n % 2:
        raise ValueError("n must be even in chosen symmetry class.")
    rng = ensure_rng(rng)
    real, imaginary = _components(n, sym, rng, circular=True)

    sector = None
    if charge is not None:
        try:
            sector = int(charge)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Impossible value of topological invariant.") from error
        if sector != charge:
            raise ValueError("Impossible value of topological invariant.")

    if sym in ("AIII", "BDI", "CII") and sector is None:
        count = n // 2 if sym == "CII" else n
        random_bits = [bool(value) for value in _binary(rng, count)]
    else:
        random_bits = []
    try:
        result = _core.rmt_circular(
            n,
            sym,
            real,
            imaginary,
            random_bits,
            sector,
        )
    except ValueError as error:
        if "topological sector" in str(error):
            raise ValueError("Impossible value of topological invariant.") from error
        raise
    return np.asarray(result, dtype=complex)


__all__ = [
    "c",
    "circular",
    "gaussian",
    "h_p_matrix",
    "h_t_matrix",
    "p",
    "sym_list",
    "t",
]
