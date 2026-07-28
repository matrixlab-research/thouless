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
    """Ordinary Schur factorization ``A = Z T Zᴴ``.

    Attributes:
        form: Upper triangular or real quasi-triangular Schur form ``T``.
        vectors: Unitary or orthogonal Schur vectors ``Z``.
        eigenvalues: Complex eigenvalues in Schur order.
    """

    form: np.ndarray
    vectors: np.ndarray
    eigenvalues: np.ndarray


@dataclass(frozen=True)
class GeneralizedSchurDecomposition:
    """Generalized Schur factorization of a matrix pencil ``A - λB``.

    Attributes:
        left_form: Generalized Schur form corresponding to ``A``.
        right_form: Generalized Schur form corresponding to ``B``.
        left_vectors: Left transformation vectors.
        right_vectors: Right transformation vectors.
        alpha: Numerators of generalized eigenvalues.
        beta: Denominators of generalized eigenvalues; ``alpha / beta`` gives
            finite generalized eigenvalues.
    """

    left_form: np.ndarray
    right_form: np.ndarray
    left_vectors: np.ndarray
    right_vectors: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray


def schur(matrix: npt.ArrayLike) -> SchurDecomposition:
    """Compute the complex Schur decomposition of a square matrix.

    Returns ``T`` and unitary ``Z`` satisfying ``matrix = Z T Zᴴ``, together
    with the diagonal eigenvalues in Schur order.
    """
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
    """Compute the real Schur decomposition of a real square matrix.

    The returned form is real quasi-triangular, retaining conjugate pairs as
    adjacent ``2 × 2`` blocks, while ``eigenvalues`` is complex.
    """
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
    """Compute the generalized complex Schur decomposition of ``(left, right)``.

    Both matrices must be square and have identical shape. Generalized
    eigenvalues are represented as ``alpha / beta`` to retain infinite values.
    """
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
    """Reusable symbolic analysis and numerical sparse LU factorization.

    Args:
        matrix: Square SciPy sparse matrix. It is canonicalized to sorted
            complex CSR storage before native analysis and factorization.
    """

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
        """Number of stored nonzeros in the normalized CSR input."""
        return int(self._factor.input_nonzeros)

    def solve(self, right_hand_side: npt.ArrayLike) -> np.ndarray:
        """Solve against one or more dense right-hand-side columns.

        The symbolic analysis and numeric LU factorization created at
        construction time are reused for every call.
        """
        rhs = complex_matrix(right_hand_side, name="right_hand_side")
        return np.asarray(
            call(self._factor.solve, rhs.tolist()),
            dtype=np.complex128,
        )


def sparse_schur_complement(
    matrix: scipy.sparse.spmatrix,
    selected: npt.ArrayLike,
) -> np.ndarray:
    """Eliminate unselected variables from a sparse square matrix.

    Args:
        matrix: Sparse matrix to partition and factor.
        selected: Zero-based indices retained in the dense Schur complement.

    Returns:
        Dense Schur complement in the order given by ``selected``.
    """
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
