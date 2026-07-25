"""General numerical utilities exposed by the PythTB compatibility layer."""

from __future__ import annotations

from itertools import permutations
from math import factorial

import numpy as np

from thouless import _core


def levi_civita(rank, dimension):
    """Return the antisymmetric rank-``rank`` tensor in ``dimension``."""
    rank = int(rank)
    dimension = int(dimension)
    if rank < 0 or dimension < 0 or rank > dimension:
        raise ValueError("rank and dimension must satisfy 0 <= rank <= dimension")
    result = np.zeros((dimension,) * rank, dtype=int)
    for indices in permutations(range(dimension), rank):
        inversions = sum(
            indices[left] > indices[right]
            for left in range(rank)
            for right in range(left + 1, rank)
        )
        result[indices] = -1 if inversions % 2 else 1
    return result


def finite_diff_coeffs(order, derivative_order=1, mode="central"):
    """Return uniform-grid finite-difference weights and integer offsets."""
    order = int(order)
    derivative_order = int(derivative_order)
    if order < 1 or derivative_order < 1:
        raise ValueError("order and derivative_order must be positive")
    if mode not in ("central", "forward", "backward"):
        raise ValueError("Mode must be 'central', 'forward', or 'backward'.")
    point_count = order + derivative_order
    if mode == "central":
        if point_count % 2 == 0:
            point_count += 1
        half_span = point_count // 2
        offsets = np.arange(-half_span, half_span + 1)
    elif mode == "forward":
        offsets = np.arange(point_count)
    else:
        offsets = np.arange(-point_count + 1, 1)
    vandermonde = np.vander(offsets, increasing=True).T
    target = np.zeros(point_count)
    target[derivative_order] = factorial(derivative_order)
    return np.linalg.solve(vandermonde, target), offsets


def finite_difference(
    values,
    axis,
    delta,
    order,
    *,
    mode="central",
    periodic=False,
):
    """Differentiate an array on a uniformly sampled axis."""
    if not np.isfinite(delta) or delta == 0:
        raise ValueError("delta must be non-zero for finite differences.")
    values = np.asarray(values)
    data = np.moveaxis(
        values.astype(np.result_type(values.dtype, np.float64), copy=False),
        axis,
        0,
    )
    size = data.shape[0]
    coefficients, offsets = finite_diff_coeffs(order, mode=mode)
    width = len(coefficients)
    if periodic:
        if size < width:
            raise ValueError(
                f"Periodic finite differences need at least {width} samples"
            )
        result = sum(
            coefficient * np.roll(data, -int(offset), axis=0)
            for coefficient, offset in zip(
                coefficients,
                offsets,
                strict=True,
            )
        )
        return np.moveaxis(result / delta, 0, axis)
    if mode == "central" and size < 2 * width - 2:
        raise ValueError(
            f"Central differences of order {order} require at least "
            f"{2 * width - 2} samples"
        )
    if mode != "central" and size < width:
        raise ValueError(
            f"{mode.capitalize()} differences of order {order} need at "
            f"least {width} samples"
        )

    result = np.empty_like(data)
    if mode == "central":
        half_width = width // 2
        for index in range(half_width, size - half_width):
            segment = data[index - half_width : index + half_width + 1]
            result[index] = np.tensordot(
                coefficients,
                segment,
                axes=(0, 0),
            ) / delta
        forward, _ = finite_diff_coeffs(order, mode="forward")
        for index in range(len(forward) - 1):
            segment = data[index : index + len(forward)]
            result[index] = np.tensordot(
                forward,
                segment,
                axes=(0, 0),
            ) / delta
        backward, _ = finite_diff_coeffs(order, mode="backward")
        for offset in range(len(backward) - 1):
            segment = data[
                size - len(backward) - offset : size - offset
            ]
            result[size - 1 - offset] = np.tensordot(
                backward,
                segment,
                axes=(0, 0),
            ) / delta
    else:
        derivative_count = size - width + 1
        derivative = np.empty((derivative_count,) + data.shape[1:], dtype=data.dtype)
        for start in range(derivative_count):
            derivative[start] = np.tensordot(
                coefficients,
                data[start : start + width],
                axes=(0, 0),
            ) / delta
        if mode == "forward":
            result[:derivative_count] = derivative
            result[derivative_count:] = derivative[-1]
        else:
            result[-derivative_count:] = derivative
            result[:-derivative_count] = derivative[0]
    return np.moveaxis(result, 0, axis)


def is_Hermitian(matrix):
    """Return whether the trailing two axes form Hermitian matrices."""
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim == 0:
        return bool(np.allclose(matrix, matrix.conj()))
    if matrix.ndim == 1:
        return False
    return bool(np.allclose(matrix, matrix.conj().swapaxes(-1, -2)))


def get_trial_wfs(trial_functions, norb, nspin=1):
    """Construct normalized trial wavefunctions from sparse amplitudes."""
    if nspin not in (1, 2):
        raise ValueError("nspin must be 1 or 2")
    shape = (
        (len(trial_functions), int(norb))
        if nspin == 1
        else (len(trial_functions), int(norb), 2)
    )
    result = np.zeros(shape, dtype=complex)
    for trial_index, trial in enumerate(trial_functions):
        if not isinstance(trial, (list, np.ndarray)):
            raise TypeError("Trial function must be a list of tuples")
        for entry in trial:
            if nspin == 1:
                orbital, amplitude = entry
                result[trial_index, orbital] = amplitude
            else:
                orbital, spin, amplitude = entry
                result[trial_index, orbital, spin] = amplitude
        norm = np.linalg.norm(result[trial_index])
        if norm == 0:
            raise ValueError("Trial functions must have nonzero norm")
        result[trial_index] /= norm
    return result


def pauli_decompose(matrix):
    """Return coefficients in the identity and Pauli-matrix basis."""
    values = np.asarray(matrix, dtype=complex)
    if values.shape != (2, 2):
        raise ValueError("Matrix must be 2x2 for Pauli decomposition.")
    return _core.pauli_decompose(values.tolist())


__all__ = [
    "finite_diff_coeffs",
    "finite_difference",
    "get_trial_wfs",
    "is_Hermitian",
    "levi_civita",
    "pauli_decompose",
]
