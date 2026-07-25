"""MUMPS-compatible sparse direct-solver interface.

The compatibility layer preserves Kwant's public sparse-solver protocol while
using SciPy's SuperLU backend when a native MUMPS installation is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import warnings

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg


orderings = {
    "amd": 0,
    "amf": 2,
    "scotch": 3,
    "pord": 4,
    "metis": 5,
    "qamd": 6,
    "auto": 7,
}


def possible_orderings():
    """Return ordering names accepted by this compatibility backend."""
    return ["auto", "amd"]


class MUMPSError(RuntimeError):
    """Sparse factorization failure exposed through Kwant's public type."""


@dataclass
class AnalysisStatistics:
    """Portable subset of Kwant's MUMPS analysis statistics."""

    est_mem_incore: int = 0
    est_mem_ooc: int = 0
    est_nonzeros: int = 0
    est_flops: float = 0.0
    ordering: str = "auto"
    time: float | None = None


@dataclass
class FactorizationStatistics:
    """Portable subset of Kwant's MUMPS factorization statistics."""

    offdiag_pivots: int = 0
    delayed_pivots: int = 0
    tiny_pivots: int = 0
    memory: int = 0
    nonzeros: int = 0
    flops: float = 0.0
    time: float | None = None
    ordering: str | None = None


def _matrix(value):
    if not sparse.isspmatrix(value):
        raise AttributeError("input matrix must provide a sparse matrix interface")
    matrix = value.tocsc().astype(np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square!")
    return matrix


def _ordering(value):
    if value not in orderings:
        raise ValueError(f"Unknown ordering '{value}'!")
    return "COLAMD" if value == "auto" else "MMD_AT_PLUS_A"


class MUMPSContext:
    """Reusable sparse LU factorization with the Kwant MUMPS protocol."""

    def __init__(self, verbose=False):
        self.verbose = bool(verbose)
        self.mumps_instance = None
        self.dtype = None
        self.factored = False
        self.analysis_stats = None
        self.factor_stats = None
        self._factorization = None
        self._shape_signature = None

    def analyze(self, a, ordering="auto", overwrite_a=False):
        """Validate and record the symbolic sparse structure."""
        del overwrite_a
        matrix = _matrix(a)
        _ordering(ordering)
        started = time.process_time()
        self.n = matrix.shape[0]
        self.dtype = matrix.dtype
        self._shape_signature = (
            matrix.shape,
            matrix.indptr.copy(),
            matrix.indices.copy(),
        )
        self.mumps_instance = self
        self.factored = False
        self.analysis_stats = AnalysisStatistics(
            est_nonzeros=int(matrix.nnz),
            ordering=ordering,
            time=time.process_time() - started,
        )

    def factor(
        self,
        a,
        ordering="auto",
        ooc=False,
        pivot_tol=0.01,
        reuse_analysis=False,
        overwrite_a=False,
    ):
        """Factor a sparse square matrix for repeated solves."""
        del ooc
        matrix = _matrix(a)
        permutation = _ordering(ordering)
        if not 0 <= pivot_tol <= 1:
            raise ValueError("pivot_tol must lie in the interval [0, 1]")
        if reuse_analysis and self.mumps_instance is None:
            warnings.warn(
                "Missing analysis although reuse_analysis=True. "
                "New analysis is performed.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.analyze(matrix, ordering=ordering, overwrite_a=overwrite_a)
        elif not reuse_analysis:
            self.analyze(matrix, ordering=ordering, overwrite_a=overwrite_a)
        elif matrix.shape != (self.n, self.n):
            raise ValueError("reused analysis has an incompatible matrix shape")

        started = time.process_time()
        try:
            self._factorization = sparse_linalg.splu(
                matrix,
                permc_spec=permutation,
                diag_pivot_thresh=float(pivot_tol),
            )
        except RuntimeError as error:
            raise MUMPSError(str(error)) from error
        self.factored = True
        self.factor_stats = FactorizationStatistics(
            nonzeros=int(
                self._factorization.L.nnz + self._factorization.U.nnz
            ),
            time=time.process_time() - started,
        )

    def solve(self, b, overwrite_b=False):
        """Solve against a dense or sparse vector or matrix right-hand side."""
        del overwrite_b
        if not self.factored or self._factorization is None:
            raise RuntimeError("Factorization must be done before solving!")
        if sparse.isspmatrix(b):
            right_hand_side = b.toarray()
        else:
            right_hand_side = np.asarray(b)
        if right_hand_side.ndim not in (1, 2):
            raise ValueError("Right hand side must be a vector or matrix")
        if right_hand_side.shape[0] != self.n:
            raise ValueError("Right hand side has wrong size")
        return self._factorization.solve(
            np.asarray(right_hand_side, dtype=np.complex128)
        )


def schur_complement(
    a,
    indices,
    ordering="auto",
    ooc=False,
    pivot_tol=0.01,
    calc_stats=False,
    overwrite_a=False,
):
    """Return the Schur complement on the selected row and column indices."""
    del ooc, overwrite_a
    matrix = _matrix(a)
    permutation = _ordering(ordering)
    selected = np.asarray(indices)
    if selected.ndim != 1:
        raise ValueError("Schur indices must be specified in a 1d array!")
    selected = selected.astype(int, copy=False)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Schur indices must be unique")
    if np.any(selected < 0) or np.any(selected >= matrix.shape[0]):
        raise IndexError("Schur index is outside the matrix")

    complement_mask = np.ones(matrix.shape[0], dtype=bool)
    complement_mask[selected] = False
    eliminated = np.flatnonzero(complement_mask)
    selected_block = matrix[selected][:, selected].toarray()
    started = time.process_time()
    if len(eliminated):
        eliminated_block = matrix[eliminated][:, eliminated].tocsc()
        coupling_from_selected = matrix[eliminated][:, selected].toarray()
        factorization = sparse_linalg.splu(
            eliminated_block,
            permc_spec=permutation,
            diag_pivot_thresh=float(pivot_tol),
        )
        result = (
            selected_block
            - matrix[selected][:, eliminated].toarray()
            @ factorization.solve(coupling_from_selected)
        )
        nonzeros = factorization.L.nnz + factorization.U.nnz
    else:
        result = selected_block
        nonzeros = 0
    result = np.asarray(result, dtype=np.complex128)
    if not calc_stats:
        return result
    statistics = FactorizationStatistics(
        nonzeros=int(nonzeros),
        time=time.process_time() - started,
        ordering=ordering,
    )
    return result, statistics


__all__ = [
    "AnalysisStatistics",
    "FactorizationStatistics",
    "MUMPSContext",
    "MUMPSError",
    "possible_orderings",
    "schur_complement",
]
