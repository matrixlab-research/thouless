"""MUMPS-compatible sparse direct-solver interface backed by Thouless."""

from __future__ import annotations

import time
import warnings

import numpy as np
import scipy.sparse as sparse
from thouless import _core


orderings = {
    "amd": 0,
    "amf": 2,
    "scotch": 3,
    "pord": 4,
    "metis": 5,
    "qamd": 6,
    "auto": 7,
}
ordering_name = [
    "amd",
    "user-defined",
    "amf",
    "scotch",
    "pord",
    "metis",
    "qamd",
]
error_messages = {
    -5: "Not enough memory during analysis phase",
    -6: "Matrix is singular in structure",
    -7: "Not enough memory during analysis phase",
    -10: "Matrix is numerically singular",
    -11: "The authors of MUMPS would like to hear about this",
    -12: "The authors of MUMPS would like to hear about this",
    -13: "Not enough memory",
}


def possible_orderings():
    """Return ordering names accepted by this compatibility backend."""
    if possible_orderings.cached is None:
        possible_orderings.cached = ["auto"]
    return possible_orderings.cached


possible_orderings.cached = None


class MUMPSError(RuntimeError):
    """Sparse factorization failure exposed through Kwant's public type."""

    def __init__(self, failure):
        if isinstance(failure, str):
            self.error = None
            message = failure
        else:
            try:
                self.error = int(failure[1])
            except (IndexError, TypeError):
                self.error = int(failure)
            description = error_messages.get(self.error)
            if description is None:
                message = f"MUMPS failed with error {self.error}."
            else:
                message = f"{description}. (MUMPS error {self.error})"
        super().__init__(message)


class AnalysisStatistics:
    """Portable subset of Kwant's MUMPS analysis statistics."""

    def __init__(
        self,
        instance=None,
        time=None,
        *,
        est_mem_incore=0,
        est_mem_ooc=0,
        est_nonzeros=0,
        est_flops=0.0,
        ordering="auto",
    ):
        if instance is not None:
            est_mem_incore = instance.infog[17]
            est_mem_ooc = instance.infog[27]
            est_nonzeros = (
                instance.infog[20]
                if instance.infog[20] > 0
                else -instance.infog[20] * 1_000_000
            )
            est_flops = instance.rinfog[1]
            ordering = ordering_name[instance.infog[7]]
        self.est_mem_incore = int(est_mem_incore)
        self.est_mem_ooc = int(est_mem_ooc)
        self.est_nonzeros = int(est_nonzeros)
        self.est_flops = float(est_flops)
        self.ordering = ordering
        self.time = time

    def __str__(self):
        parts = [
            "estimated memory for in-core factorization:",
            str(self.est_mem_incore),
            "mbytes\n",
            "estimated memory for out-of-core factorization:",
            str(self.est_mem_ooc),
            "mbytes\n",
            "estimated number of nonzeros in factors:",
            str(self.est_nonzeros),
            "\n",
            "estimated number of flops:",
            str(self.est_flops),
            "\n",
            "ordering used:",
            self.ordering,
        ]
        if self.time is not None:
            parts.extend(["\n analysis time:", str(self.time), "secs"])
        return " ".join(parts)


class FactorizationStatistics:
    """Portable subset of Kwant's MUMPS factorization statistics."""

    def __init__(
        self,
        instance=None,
        time=None,
        include_ordering=False,
        *,
        offdiag_pivots=0,
        delayed_pivots=0,
        tiny_pivots=0,
        memory=0,
        nonzeros=0,
        flops=0.0,
        ordering=None,
    ):
        if instance is not None:
            offdiag_pivots = instance.infog[12] if instance.sym == 0 else 0
            delayed_pivots = instance.infog[13]
            tiny_pivots = instance.infog[25]
            memory = instance.infog[22]
            nonzeros = (
                instance.infog[29]
                if instance.infog[29] > 0
                else -instance.infog[29] * 1_000_000
            )
            flops = instance.rinfog[3]
            if include_ordering:
                ordering = ordering_name[instance.infog[7]]
        self.offdiag_pivots = int(offdiag_pivots)
        self.delayed_pivots = int(delayed_pivots)
        self.tiny_pivots = int(tiny_pivots)
        self.memory = int(memory)
        self.nonzeros = int(nonzeros)
        self.flops = float(flops)
        if time is not None:
            self.time = time
        if ordering is not None:
            self.ordering = ordering

    def __str__(self):
        parts = [
            "off-diagonal pivots:",
            str(self.offdiag_pivots),
            "\n",
            "delayed pivots:",
            str(self.delayed_pivots),
            "\n",
            "tiny pivots:",
            str(self.tiny_pivots),
            "\n",
        ]
        if hasattr(self, "ordering"):
            parts.extend(["ordering used:", self.ordering, "\n"])
        parts.extend(
            [
                "memory used during factorization:",
                str(self.memory),
                "mbytes\n",
                "nonzeros in factored matrix:",
                str(self.nonzeros),
                "\n",
                "floating point operations:",
                str(self.flops),
            ]
        )
        if hasattr(self, "time"):
            parts.extend(["\n factorization time:", str(self.time), "secs"])
        return " ".join(parts)


def _matrix(value):
    if not sparse.isspmatrix(value):
        raise AttributeError("input matrix must provide a sparse matrix interface")
    matrix = value.tocsr().astype(np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square!")
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    return matrix


def _ordering(value):
    if value not in orderings:
        raise ValueError(f"Unknown ordering '{value}'!")


def _pivot_tolerance(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError("pivot_tol must lie in the interval [0, 1]")
    return value


def _native_matrix_arguments(matrix):
    return (
        int(matrix.shape[0]),
        int(matrix.shape[1]),
        matrix.indptr.tolist(),
        matrix.indices.tolist(),
        matrix.data.tolist(),
    )


def _storage_megabytes(nonzeros):
    return int(np.ceil(int(nonzeros) * 24 / 1_000_000))


class MUMPSContext:
    """Reusable sparse LU factorization with the Kwant MUMPS protocol."""

    def __init__(self, verbose=False):
        self.verbose = bool(verbose)
        self.mumps_instance = None
        self.dtype = None
        self.factored = False
        self.analysis_stats = None
        self.factor_stats = None
        self._analysis = None
        self._factorization = None

    def analyze(self, a, ordering="auto", overwrite_a=False):
        """Validate and record the symbolic sparse structure."""
        del overwrite_a
        matrix = _matrix(a)
        _ordering(ordering)
        started = time.process_time()
        self.n = matrix.shape[0]
        self.dtype = matrix.dtype
        try:
            self._analysis = _core.sparse_lu_analyze(
                *_native_matrix_arguments(matrix)
            )
        except RuntimeError as error:
            raise MUMPSError(str(error)) from error
        self.mumps_instance = self
        self.factored = False
        self._factorization = None
        self.analysis_stats = AnalysisStatistics(
            est_mem_incore=_storage_megabytes(
                self._analysis.input_nonzeros
            ),
            est_nonzeros=self._analysis.input_nonzeros,
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
        _ordering(ordering)
        _pivot_tolerance(pivot_tol)
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

        started = time.process_time()
        try:
            self._factorization = self._analysis.factor(
                *_native_matrix_arguments(matrix)
            )
        except RuntimeError as error:
            raise MUMPSError(str(error)) from error
        self.factored = True
        self.factor_stats = FactorizationStatistics(
            memory=_storage_megabytes(
                self._factorization.input_nonzeros
            ),
            nonzeros=self._factorization.input_nonzeros,
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
        was_vector = right_hand_side.ndim == 1
        dense = np.asarray(right_hand_side, dtype=np.complex128)
        if was_vector:
            dense = dense.reshape(self.n, 1)
        solution = np.asarray(
            self._factorization.solve(dense.tolist()),
            dtype=np.complex128,
        )
        return solution[:, 0] if was_vector else solution


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
    _ordering(ordering)
    _pivot_tolerance(pivot_tol)
    selected = np.asarray(indices)
    if selected.ndim != 1:
        raise ValueError("Schur indices must be specified in a 1d array!")
    selected = selected.astype(int, copy=False)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Schur indices must be unique")
    if np.any(selected < 0) or np.any(selected >= matrix.shape[0]):
        raise IndexError("Schur index is outside the matrix")

    started = time.process_time()
    try:
        result = _core.sparse_schur_complement(
            *_native_matrix_arguments(matrix),
            selected.tolist(),
        )
    except RuntimeError as error:
        raise MUMPSError(str(error)) from error
    result = np.asarray(result, dtype=np.complex128)
    if not calc_stats:
        return result
    statistics = FactorizationStatistics(
        memory=_storage_megabytes(matrix.nnz),
        nonzeros=int(matrix.nnz),
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
