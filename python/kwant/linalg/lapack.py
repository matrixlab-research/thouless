"""Low-level LAPACK-compatible entry points backed by Rust decompositions."""

from __future__ import annotations

import numpy as np


class LinAlgError(RuntimeError):
    """A Schur reordering or eigenvector computation failed."""


int_dtype = np.int32
logical_dtype = np.int32


def assert_fortran_mat(*matrices):
    for matrix in matrices:
        if (
            matrix is not None
            and matrix.ndim == 2
            and max(matrix.shape, default=0) > 1
            and not matrix.flags["F_CONTIGUOUS"]
        ):
            raise ValueError("Input matrix must be Fortran contiguous")


def prepare_for_lapack(overwrite, *values):
    """Convert one- or two-dimensional numeric arrays to LAPACK layout."""
    arrays = []
    for value in values:
        if value is None:
            arrays.append(None)
            continue
        array = np.asanyarray(value)
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("Argument cannot be interpreted as a numeric array")
        if array.ndim not in (1, 2):
            raise ValueError("Dimensionality of array is not 1 or 2")
        arrays.append(array)
    supplied = [array for array in arrays if array is not None]
    if not supplied:
        result = [None] * len(arrays)
        return result[0] if len(result) == 1 else tuple(result)
    dtype = np.result_type(*[array.dtype for array in supplied], np.float32)
    if dtype.kind not in "fc":
        dtype = np.dtype(np.float64)
    if dtype not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.complex64),
        np.dtype(np.complex128),
    ):
        dtype = np.dtype(
            np.complex128 if np.issubdtype(dtype, np.complexfloating)
            else np.float64
        )
    result = []
    for original, array in zip(values, arrays, strict=True):
        if array is None:
            result.append(None)
        elif (
            overwrite
            and original is array
            and array.dtype == dtype
            and (
                (array.ndim == 2 and array.flags["F_CONTIGUOUS"])
                or (array.ndim == 1 and array.flags["C_CONTIGUOUS"])
            )
        ):
            result.append(array)
        elif array.ndim == 2:
            result.append(np.array(array, dtype=dtype, order="F", copy=True))
        else:
            result.append(np.array(array, dtype=dtype, order="C", copy=True))
    return result[0] if len(result) == 1 else tuple(result)


def trsen(select, t, q, calc_ev=True):
    from .decomp_schur import order_schur

    dimension = np.asarray(t).shape[0]
    vectors = np.eye(dimension, dtype=np.asarray(t).dtype) if q is None else q
    result = order_schur(select, t, vectors, calc_ev=calc_ev)
    output = (result[0],)
    if q is not None:
        output += (result[1],)
    if calc_ev:
        output += (result[-1],)
    return output


def trevc(t, q, select, left=False, right=True):
    if not left and not right:
        return None
    from .decomp_schur import evecs_from_schur

    dimension = np.asarray(t).shape[0]
    vectors = np.eye(dimension, dtype=np.asarray(t).dtype) if q is None else q
    return evecs_from_schur(
        t,
        vectors,
        select=select,
        left=left,
        right=right,
    )


def tgsen(select, s, t, q, z, calc_ev=True):
    from .decomp_schur import order_gen_schur

    return order_gen_schur(
        select,
        s,
        t,
        q=q,
        z=z,
        calc_ev=calc_ev,
    )


def tgevc(s, t, q, z, select, left=False, right=True):
    if not left and not right:
        return None
    from .decomp_schur import evecs_from_gen_schur

    dimension = np.asarray(s).shape[0]
    left_vectors = (
        np.eye(dimension, dtype=np.asarray(s).dtype) if q is None else q
    )
    right_vectors = (
        np.eye(dimension, dtype=np.asarray(s).dtype) if z is None else z
    )
    return evecs_from_gen_schur(
        s,
        t,
        q=left_vectors,
        z=right_vectors,
        select=select,
        left=left,
        right=right,
    )


__all__ = [
    "prepare_for_lapack",
    "tgevc",
    "tgsen",
    "trevc",
    "trsen",
]
