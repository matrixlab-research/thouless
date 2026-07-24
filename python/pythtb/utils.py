"""Small PythTB utility compatibility functions backed by Rust."""

from __future__ import annotations

import numpy as np

from thouless import _core


def pauli_decompose(matrix):
    values = np.asarray(matrix, dtype=complex)
    if values.shape != (2, 2):
        raise ValueError("Matrix must be 2x2 for Pauli decomposition.")
    return _core.pauli_decompose(values.tolist())


__all__ = ["pauli_decompose"]
