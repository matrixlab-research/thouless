"""Random-matrix helpers for symmetry-free Gaussian and circular ensembles."""

from __future__ import annotations

import numpy as np

from ._common import ensure_rng


sym_list = ("A",)
h_t_matrix = {}
h_p_matrix = {}


def t(symmetry):
    return 0


def p(symmetry):
    return 0


def c(symmetry):
    return 0


def gaussian(n, sym="A", v=1.0, rng=None):
    """Draw a Hermitian matrix from the unitary Gaussian ensemble."""
    if sym != "A":
        raise NotImplementedError(f"Gaussian symmetry class {sym!r} is not implemented")
    rng = ensure_rng(rng)
    matrix = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return float(v) * (matrix + matrix.conj().T) / np.sqrt(2 * n)


def circular(n, sym="A", charge=None, rng=None):
    """Draw a Haar-distributed unitary matrix."""
    if sym != "A":
        raise NotImplementedError(f"Circular symmetry class {sym!r} is not implemented")
    rng = ensure_rng(rng)
    matrix = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0, phases / np.abs(phases), 1)
    return q * phases.conj()


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
