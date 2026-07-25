"""Kwant Schur and generalized Schur interfaces over the Rust core."""

from __future__ import annotations

import numpy as np

from thouless import _core


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


def _matrix(rows, dimension):
    return np.asarray(rows, dtype=np.complex128).reshape(dimension, dimension)


def _column_matrix(rows, dimension, columns):
    return np.asarray(rows, dtype=np.complex128).reshape(dimension, columns)


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


def _requested_eigenvectors(left_rows, right_rows, dimension, selected, left, right):
    columns = sum(selected)
    left_vectors = (
        _column_matrix(left_rows, dimension, columns)
        if left
        else None
    )
    right_vectors = (
        _column_matrix(right_rows, dimension, columns)
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
    matrix = _square_matrix(a, "a")
    dimension = matrix.shape[0]
    form, vectors, eigenvalues = _core.dense_schur(_rows(matrix))
    result = (_matrix(form, dimension),)
    if calc_q:
        result += (_matrix(vectors, dimension),)
    if calc_ev:
        result += (np.asarray(eigenvalues, dtype=np.complex128),)
    return result


def convert_r2c_schur(t, q):
    form, vectors = _common_square_matrices(t, q)
    if np.iscomplexobj(form) and np.iscomplexobj(vectors):
        return np.array(form, copy=True), np.array(vectors, copy=True)
    dimension = form.shape[0]
    converted, converted_vectors, _ = _core.dense_complexify_schur(
        _rows(form),
        _rows(vectors),
    )
    return _matrix(converted, dimension), _matrix(converted_vectors, dimension)


def order_schur(select, t, q, calc_ev=True, overwrite_tq=False):
    del overwrite_tq
    form, vectors = _common_square_matrices(t, q)
    dimension = form.shape[0]
    selected = _selection(select, dimension)
    reordered, reordered_vectors, eigenvalues = _core.dense_reorder_schur(
        _rows(form),
        _rows(vectors),
        selected,
    )
    result = (
        _matrix(reordered, dimension),
        _matrix(reordered_vectors, dimension),
    )
    if calc_ev:
        result += (np.asarray(eigenvalues, dtype=np.complex128),)
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
    form, vectors = _common_square_matrices(t, q)
    dimension = form.shape[0]
    selected = _selection(select, dimension, default_all=True)
    left_rows, right_rows = _core.dense_schur_eigenvectors(
        _rows(form),
        _rows(vectors),
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
    left, right = _common_square_matrices(a, b)
    dimension = left.shape[0]
    (
        left_form,
        right_form,
        left_vectors,
        right_vectors,
        alpha,
        beta,
    ) = _core.dense_generalized_schur(_rows(left), _rows(right))
    result = (
        _matrix(left_form, dimension),
        _matrix(right_form, dimension),
    )
    if calc_q:
        result += (_matrix(left_vectors, dimension),)
    if calc_z:
        result += (_matrix(right_vectors, dimension),)
    if calc_ev:
        result += (
            np.asarray(alpha, dtype=np.complex128),
            np.asarray(beta, dtype=np.complex128),
        )
    return result


def convert_r2c_gen_schur(s, t, q=None, z=None):
    supplied = [s, t]
    if q is not None:
        supplied.append(q)
    if z is not None:
        supplied.append(z)
    matrices = _common_square_matrices(*supplied)
    left_form, right_form = matrices[:2]
    dimension = left_form.shape[0]
    index = 2
    left_vectors = matrices[index] if q is not None else np.eye(dimension)
    index += q is not None
    right_vectors = matrices[index] if z is not None else np.eye(dimension)

    if all(np.iscomplexobj(matrix) for matrix in matrices):
        converted = [np.array(matrix, copy=True) for matrix in matrices]
        return tuple(converted)

    (
        converted_left,
        converted_right,
        converted_q,
        converted_z,
        _,
        _,
    ) = _core.dense_complexify_generalized_schur(
        _rows(left_form),
        _rows(right_form),
        _rows(left_vectors),
        _rows(right_vectors),
    )
    result = (
        _matrix(converted_left, dimension),
        _matrix(converted_right, dimension),
    )
    if q is not None:
        result += (_matrix(converted_q, dimension),)
    if z is not None:
        result += (_matrix(converted_z, dimension),)
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
    matrices = _common_square_matrices(*supplied)
    left_form, right_form = matrices[:2]
    dimension = left_form.shape[0]
    index = 2
    left_vectors = matrices[index] if q is not None else np.eye(dimension)
    index += q is not None
    right_vectors = matrices[index] if z is not None else np.eye(dimension)
    selected = _selection(select, dimension)

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
    result = (
        _matrix(reordered_left, dimension),
        _matrix(reordered_right, dimension),
    )
    if q is not None:
        result += (_matrix(reordered_q, dimension),)
    if z is not None:
        result += (_matrix(reordered_z, dimension),)
    if calc_ev:
        result += (
            np.asarray(alpha, dtype=np.complex128),
            np.asarray(beta, dtype=np.complex128),
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
    matrices = _common_square_matrices(*supplied)
    left_form, right_form = matrices[:2]
    dimension = left_form.shape[0]
    index = 2
    left_vectors = matrices[index] if q is not None else np.eye(dimension)
    index += q is not None
    right_vectors = matrices[index] if z is not None else np.eye(dimension)
    selected = _selection(select, dimension, default_all=True)

    left_rows, right_rows = _core.dense_generalized_schur_eigenvectors(
        _rows(left_form),
        _rows(right_form),
        _rows(left_vectors),
        _rows(right_vectors),
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
