"""Kwant Schur and generalized Schur interfaces over the Rust core."""

from __future__ import annotations

import numpy as np

from thouless import _core
from . import lapack


def _square_matrix(value, name):
    matrix = np.asarray(value)
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"{name} cannot be interpreted as a numeric array")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return matrix


def _common_square_matrices(*values):
    matrices = [
        _square_matrix(value, f"matrix {index}")
        for index, value in enumerate(values)
    ]
    if matrices and any(matrix.shape != matrices[0].shape for matrix in matrices[1:]):
        raise ValueError("all decomposition matrices must have the same shape")
    return matrices


def _rows(matrix):
    return np.asarray(matrix, dtype=np.complex128).tolist()


def _real_rows(matrix):
    return np.asarray(matrix, dtype=np.float64).tolist()


def _matrix(rows, dimension, dtype):
    return np.asarray(rows, dtype=dtype).reshape(dimension, dimension)


def _column_matrix(rows, dimension, columns, dtype):
    return np.asarray(rows, dtype=dtype).reshape(dimension, columns)


def _complex_dtype(dtype):
    dtype = np.dtype(dtype)
    return np.dtype(
        np.complex64
        if dtype in (np.dtype(np.float32), np.dtype(np.complex64))
        else np.complex128
    )


def _prepared_square_matrices(*values):
    matrices = _common_square_matrices(*values)
    prepared = lapack.prepare_for_lapack(False, *matrices)
    if len(matrices) == 1:
        return [prepared]
    return list(prepared)


def _selection(select, dimension, default_all=False):
    if select is None:
        if not default_all:
            return None
        return [True] * dimension
    if callable(select):
        return [bool(select(index)) for index in range(dimension)]
    try:
        values = np.asarray(select, dtype=bool)
    except (TypeError, ValueError) as error:
        raise ValueError("select must be either a function or an array") from error
    if values.ndim != 1 or len(values) != dimension:
        raise ValueError("select must contain one value per eigenvalue")
    return values.tolist()


def _requested_eigenvectors(
    left_rows,
    right_rows,
    dimension,
    selected,
    left,
    right,
    dtype,
):
    columns = sum(selected)
    left_vectors = (
        _column_matrix(left_rows, dimension, columns, dtype)
        if left
        else None
    )
    right_vectors = (
        _column_matrix(right_rows, dimension, columns, dtype)
        if right
        else None
    )
    if left and right:
        return left_vectors, right_vectors
    if left:
        return left_vectors
    if right:
        return right_vectors
    return ()


def schur(a, calc_q=True, calc_ev=True, overwrite_a=False):
    del overwrite_a
    matrix = _prepared_square_matrices(a)[0]
    dimension = matrix.shape[0]
    dtype = matrix.dtype
    eigenvalue_dtype = _complex_dtype(dtype)
    if np.iscomplexobj(matrix):
        form, vectors, eigenvalues = _core.dense_schur(_rows(matrix))
    else:
        form, vectors, eigenvalues = _core.dense_real_schur(_real_rows(matrix))
    result = (_matrix(form, dimension, dtype),)
    if calc_q:
        result += (_matrix(vectors, dimension, dtype),)
    if calc_ev:
        result += (np.asarray(eigenvalues, dtype=eigenvalue_dtype),)
    return result


def convert_r2c_schur(t, q):
    form, vectors = _prepared_square_matrices(t, q)
    if np.iscomplexobj(form):
        return np.array(form, copy=True), np.array(vectors, copy=True)
    if not np.any(np.diagonal(form, offset=-1)):
        return np.array(form, copy=True), np.array(vectors, copy=True)
    dimension = form.shape[0]
    dtype = _complex_dtype(form.dtype)
    converted, converted_vectors, _ = _core.dense_complexify_real_schur(
        _real_rows(form),
        _real_rows(vectors),
    )
    return (
        _matrix(converted, dimension, dtype),
        _matrix(converted_vectors, dimension, dtype),
    )


def order_schur(select, t, q, calc_ev=True, overwrite_tq=False):
    del overwrite_tq
    form, vectors = _prepared_square_matrices(t, q)
    dimension = form.shape[0]
    selected = _selection(select, dimension)
    output_dtype = form.dtype
    if np.iscomplexobj(form):
        reordered, reordered_vectors, eigenvalues = _core.dense_reorder_schur(
            _rows(form),
            _rows(vectors),
            selected,
        )
    else:
        split_pair = any(
            form[index + 1, index] != 0
            and bool(selected[index]) != bool(selected[index + 1])
            for index in range(max(0, dimension - 1))
        )
        if split_pair:
            complex_form, complex_vectors, _ = _core.dense_complexify_real_schur(
                _real_rows(form),
                _real_rows(vectors),
            )
            reordered, reordered_vectors, eigenvalues = _core.dense_reorder_schur(
                complex_form,
                complex_vectors,
                selected,
            )
            output_dtype = _complex_dtype(form.dtype)
        else:
            (
                reordered,
                reordered_vectors,
                eigenvalues,
            ) = _core.dense_reorder_real_schur(
                _real_rows(form),
                _real_rows(vectors),
                selected,
            )
    result = (
        _matrix(reordered, dimension, output_dtype),
        _matrix(reordered_vectors, dimension, output_dtype),
    )
    if calc_ev:
        result += (np.asarray(eigenvalues, dtype=_complex_dtype(form.dtype)),)
    return result


def evecs_from_schur(
    t,
    q,
    select=None,
    left=False,
    right=True,
    overwrite_tq=False,
):
    del overwrite_tq
    form, vectors = _prepared_square_matrices(t, q)
    dimension = form.shape[0]
    selected = _selection(select, dimension, default_all=True)
    if np.iscomplexobj(form):
        left_rows, right_rows = _core.dense_schur_eigenvectors(
            _rows(form),
            _rows(vectors),
            selected,
            bool(left),
            bool(right),
        )
    else:
        left_rows, right_rows = _core.dense_real_schur_eigenvectors(
            _real_rows(form),
            _real_rows(vectors),
            selected,
            bool(left),
            bool(right),
        )
    return _requested_eigenvectors(
        left_rows,
        right_rows,
        dimension,
        selected,
        bool(left),
        bool(right),
        _complex_dtype(form.dtype),
    )


def gen_schur(
    a,
    b,
    calc_q=True,
    calc_z=True,
    calc_ev=True,
    overwrite_ab=False,
):
    del overwrite_ab
    left, right = _prepared_square_matrices(a, b)
    dimension = left.shape[0]
    dtype = left.dtype
    if np.iscomplexobj(left):
        (
            left_form,
            right_form,
            left_vectors,
            right_vectors,
            alpha,
            beta,
        ) = _core.dense_generalized_schur(_rows(left), _rows(right))
        beta_dtype = dtype
    else:
        (
            left_form,
            right_form,
            left_vectors,
            right_vectors,
            alpha,
            beta,
        ) = _core.dense_generalized_real_schur(
            _real_rows(left),
            _real_rows(right),
        )
        beta_dtype = dtype
    result = (
        _matrix(left_form, dimension, dtype),
        _matrix(right_form, dimension, dtype),
    )
    if calc_q:
        result += (_matrix(left_vectors, dimension, dtype),)
    if calc_z:
        result += (_matrix(right_vectors, dimension, dtype),)
    if calc_ev:
        result += (
            np.asarray(alpha, dtype=_complex_dtype(dtype)),
            np.asarray(beta, dtype=beta_dtype),
        )
    return result


def convert_r2c_gen_schur(s, t, q=None, z=None):
    supplied = [s, t]
    if q is not None:
        supplied.append(q)
    if z is not None:
        supplied.append(z)
    matrices = _prepared_square_matrices(*supplied)
    left_form, right_form = matrices[:2]
    dimension = left_form.shape[0]
    index = 2
    left_vectors = (
        matrices[index]
        if q is not None
        else np.eye(dimension, dtype=left_form.dtype)
    )
    index += q is not None
    right_vectors = (
        matrices[index]
        if z is not None
        else np.eye(dimension, dtype=left_form.dtype)
    )

    if np.iscomplexobj(left_form) or not np.any(np.diagonal(left_form, offset=-1)):
        converted = [np.array(matrix, copy=True) for matrix in matrices]
        return tuple(converted)

    dtype = _complex_dtype(left_form.dtype)
    (
        converted_left,
        converted_right,
        converted_q,
        converted_z,
        _,
        _,
    ) = _core.dense_complexify_real_generalized_schur(
        _real_rows(left_form),
        _real_rows(right_form),
        _real_rows(left_vectors),
        _real_rows(right_vectors),
    )
    result = (
        _matrix(converted_left, dimension, dtype),
        _matrix(converted_right, dimension, dtype),
    )
    if q is not None:
        result += (_matrix(converted_q, dimension, dtype),)
    if z is not None:
        result += (_matrix(converted_z, dimension, dtype),)
    return result


def order_gen_schur(
    select,
    s,
    t,
    q=None,
    z=None,
    calc_ev=True,
    overwrite_stqz=False,
):
    del overwrite_stqz
    supplied = [s, t]
    if q is not None:
        supplied.append(q)
    if z is not None:
        supplied.append(z)
    matrices = _prepared_square_matrices(*supplied)
    left_form, right_form = matrices[:2]
    dimension = left_form.shape[0]
    index = 2
    left_vectors = (
        matrices[index]
        if q is not None
        else np.eye(dimension, dtype=left_form.dtype)
    )
    index += q is not None
    right_vectors = (
        matrices[index]
        if z is not None
        else np.eye(dimension, dtype=left_form.dtype)
    )
    selected = _selection(select, dimension)

    output_dtype = left_form.dtype
    if np.iscomplexobj(left_form):
        (
            reordered_left,
            reordered_right,
            reordered_q,
            reordered_z,
            alpha,
            beta,
        ) = _core.dense_reorder_generalized_schur(
            _rows(left_form),
            _rows(right_form),
            _rows(left_vectors),
            _rows(right_vectors),
            selected,
        )
    else:
        split_pair = any(
            left_form[index + 1, index] != 0
            and bool(selected[index]) != bool(selected[index + 1])
            for index in range(max(0, dimension - 1))
        )
        if split_pair:
            (
                complex_left,
                complex_right,
                complex_q,
                complex_z,
                _,
                _,
            ) = _core.dense_complexify_real_generalized_schur(
                _real_rows(left_form),
                _real_rows(right_form),
                _real_rows(left_vectors),
                _real_rows(right_vectors),
            )
            (
                reordered_left,
                reordered_right,
                reordered_q,
                reordered_z,
                alpha,
                beta,
            ) = _core.dense_reorder_generalized_schur(
                complex_left,
                complex_right,
                complex_q,
                complex_z,
                selected,
            )
            output_dtype = _complex_dtype(left_form.dtype)
        else:
            (
                reordered_left,
                reordered_right,
                reordered_q,
                reordered_z,
                alpha,
                beta,
            ) = _core.dense_reorder_generalized_real_schur(
                _real_rows(left_form),
                _real_rows(right_form),
                _real_rows(left_vectors),
                _real_rows(right_vectors),
                selected,
            )
    result = (
        _matrix(reordered_left, dimension, output_dtype),
        _matrix(reordered_right, dimension, output_dtype),
    )
    if q is not None:
        result += (_matrix(reordered_q, dimension, output_dtype),)
    if z is not None:
        result += (_matrix(reordered_z, dimension, output_dtype),)
    if calc_ev:
        result += (
            np.asarray(alpha, dtype=_complex_dtype(left_form.dtype)),
            np.asarray(beta, dtype=output_dtype),
        )
    return result


def evecs_from_gen_schur(
    s,
    t,
    q=None,
    z=None,
    select=None,
    left=False,
    right=True,
    overwrite_qz=False,
):
    del overwrite_qz
    if left and q is None:
        raise ValueError("Matrix q must be provided for left eigenvectors")
    if right and z is None:
        raise ValueError("Matrix z must be provided for right eigenvectors")

    supplied = [s, t]
    if q is not None:
        supplied.append(q)
    if z is not None:
        supplied.append(z)
    matrices = _prepared_square_matrices(*supplied)
    left_form, right_form = matrices[:2]
    dimension = left_form.shape[0]
    index = 2
    left_vectors = (
        matrices[index]
        if q is not None
        else np.eye(dimension, dtype=left_form.dtype)
    )
    index += q is not None
    right_vectors = (
        matrices[index]
        if z is not None
        else np.eye(dimension, dtype=left_form.dtype)
    )
    selected = _selection(select, dimension, default_all=True)

    if np.iscomplexobj(left_form):
        left_rows, right_rows = _core.dense_generalized_schur_eigenvectors(
            _rows(left_form),
            _rows(right_form),
            _rows(left_vectors),
            _rows(right_vectors),
            selected,
            bool(left),
            bool(right),
        )
    else:
        (
            left_rows,
            right_rows,
        ) = _core.dense_generalized_real_schur_eigenvectors(
            _real_rows(left_form),
            _real_rows(right_form),
            _real_rows(left_vectors),
            _real_rows(right_vectors),
            selected,
            bool(left),
            bool(right),
        )
    return _requested_eigenvectors(
        left_rows,
        right_rows,
        dimension,
        selected,
        bool(left),
        bool(right),
        _complex_dtype(left_form.dtype),
    )


__all__ = [
    "convert_r2c_gen_schur",
    "convert_r2c_schur",
    "evecs_from_gen_schur",
    "evecs_from_schur",
    "gen_schur",
    "order_gen_schur",
    "order_schur",
    "schur",
]
