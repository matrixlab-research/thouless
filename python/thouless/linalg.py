"""Dense decompositions and reusable sparse direct solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse

from . import _core
from ._binding import call, complex_matrix, real_matrix


@dataclass(frozen=True)
class SchurDecomposition:
    form: np.ndarray
    vectors: np.ndarray
    eigenvalues: np.ndarray


@dataclass(frozen=True)
class GeneralizedSchurDecomposition:
    left_form: np.ndarray
    right_form: np.ndarray
    left_vectors: np.ndarray
    right_vectors: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray


def schur(matrix: npt.ArrayLike) -> SchurDecomposition:
    form, vectors, eigenvalues = call(
        _core.dense_schur,
        complex_matrix(matrix, name="matrix").tolist(),
    )
    return SchurDecomposition(
        np.asarray(form, dtype=np.complex128),
        np.asarray(vectors, dtype=np.complex128),
        np.asarray(eigenvalues, dtype=np.complex128),
    )


def real_schur(matrix: npt.ArrayLike) -> SchurDecomposition:
    form, vectors, eigenvalues = call(
        _core.dense_real_schur,
        real_matrix(matrix, name="matrix").tolist(),
    )
    return SchurDecomposition(
        np.asarray(form, dtype=np.float64),
        np.asarray(vectors, dtype=np.float64),
        np.asarray(eigenvalues, dtype=np.complex128),
    )


def generalized_schur(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
) -> GeneralizedSchurDecomposition:
    result = call(
        _core.dense_generalized_schur,
        complex_matrix(left, name="left").tolist(),
        complex_matrix(right, name="right").tolist(),
    )
    return GeneralizedSchurDecomposition(
        np.asarray(result[0], dtype=np.complex128),
        np.asarray(result[1], dtype=np.complex128),
        np.asarray(result[2], dtype=np.complex128),
        np.asarray(result[3], dtype=np.complex128),
        np.asarray(result[4], dtype=np.complex128),
        np.asarray(result[5], dtype=np.complex128),
    )


class SparseLU:
    """Reusable symbolic analysis and numerical factorization."""

    def __init__(self, matrix: scipy.sparse.spmatrix) -> None:
        csr = scipy.sparse.csr_matrix(matrix, dtype=np.complex128)
        csr.sort_indices()
        self._shape = tuple(int(value) for value in csr.shape)
        self._row_offsets = csr.indptr.astype(np.uintp).tolist()
        self._column_indices = csr.indices.astype(np.uintp).tolist()
        self._analysis = call(
            _core.sparse_lu_analyze,
            self._shape[0],
            self._shape[1],
            self._row_offsets,
            self._column_indices,
            csr.data.tolist(),
        )
        self._factor = call(
            self._analysis.factor,
            self._shape[0],
            self._shape[1],
            self._row_offsets,
            self._column_indices,
            csr.data.tolist(),
        )

    @property
    def input_nonzeros(self) -> int:
        return int(self._factor.input_nonzeros)

    def solve(self, right_hand_side: npt.ArrayLike) -> np.ndarray:
        rhs = complex_matrix(right_hand_side, name="right_hand_side")
        return np.asarray(
            call(self._factor.solve, rhs.tolist()),
            dtype=np.complex128,
        )


def sparse_schur_complement(
    matrix: scipy.sparse.spmatrix,
    selected: npt.ArrayLike,
) -> np.ndarray:
    csr = scipy.sparse.csr_matrix(matrix, dtype=np.complex128)
    csr.sort_indices()
    return np.asarray(
        call(
            _core.sparse_schur_complement,
            int(csr.shape[0]),
            int(csr.shape[1]),
            csr.indptr.astype(np.uintp).tolist(),
            csr.indices.astype(np.uintp).tolist(),
            csr.data.tolist(),
            np.asarray(selected, dtype=np.uintp).tolist(),
        ),
        dtype=np.complex128,
    )


__all__ = [
    "GeneralizedSchurDecomposition",
    "SchurDecomposition",
    "SparseLU",
    "generalized_schur",
    "real_schur",
    "schur",
    "sparse_schur_complement",
]
