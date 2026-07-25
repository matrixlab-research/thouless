"""General numerical utilities exposed by the PythTB compatibility layer."""

from __future__ import annotations

import functools
from itertools import permutations
from math import factorial
import warnings

import numpy as np

from thouless import _core

_TF_CACHE = None


def get_tensorflow():
    """Lazily load the optional TensorFlow linear-algebra entry points."""
    global _TF_CACHE
    if _TF_CACHE is None:
        try:
            import tensorflow as tf
        except ImportError as error:
            raise ImportError(
                "TensorFlow support requires pythtb[speedup] or TensorFlow."
            ) from error
        _TF_CACHE = {
            "convert_to_tensor": tf.convert_to_tensor,
            "eigvalsh": tf.linalg.eigvalsh,
            "eigh": tf.linalg.eigh,
            "complex64": tf.complex64,
            "complex128": tf.complex128,
        }
    return _TF_CACHE


def deprecated(message, category=FutureWarning):
    """Decorate a callable with a warning while preserving its metadata."""

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{function.__qualname__} is deprecated and will be "
                f"removed in a future release: {message}",
                category=category,
                stacklevel=2,
            )
            return function(*args, **kwargs)

        return wrapper

    return decorator


def copydoc(src):
    """Copy one callable's docstring to another."""

    def decorator(target):
        target.__doc__ = src.__doc__
        return target

    return decorator


def levi_civita(n, d):
    """Return the antisymmetric rank-``rank`` tensor in ``dimension``."""
    rank = int(n)
    dimension = int(d)
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


def detect_degeneracies(eigenvalues, tol=1e-8):
    """Return index groups of adjacent degenerate sorted eigenvalues."""
    values = np.sort(np.asarray(eigenvalues, dtype=float))
    if values.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional")
    groups = []
    current = [0] if len(values) else []
    for index in range(1, len(values)):
        if abs(values[index] - values[index - 1]) < tol:
            current.append(index)
        else:
            if len(current) > 1:
                groups.append(current)
            current = [index]
    if len(current) > 1:
        groups.append(current)
    return groups


def mat_exp(M):
    """Evaluate a batch-compatible matrix exponential by eigendecomposition."""
    values = np.asarray(M, dtype=complex)
    if values.ndim < 2 or values.shape[-1] != values.shape[-2]:
        raise ValueError("matrix exponential requires square trailing axes")
    eigenvalues, eigenvectors = np.linalg.eig(values)
    inverse = np.linalg.inv(eigenvectors)
    return (
        eigenvectors
        @ (np.exp(eigenvalues)[..., :, np.newaxis] * inverse)
    )


def kpath_distance(k_frac, b1, b2, b3):
    """Return cumulative Cartesian distance along a three-dimensional path."""
    points = np.asarray(k_frac, dtype=float)
    reciprocal = np.vstack((b1, b2, b3))
    cartesian = points @ reciprocal
    result = np.zeros(len(cartesian), dtype=float)
    if len(result) > 1:
        result[1:] = np.cumsum(
            np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
        )
    return result


def get_k_shell(model, nks, N_sh, report=False):
    """Deprecated model-level adapter for reciprocal neighbor shells."""
    lattice = model.lattice if hasattr(model, "lattice") else model
    return lattice.nn_k_shell(nks, N_sh, report=report)


def get_fd_weights(model, nks, dim_k, N_sh=1, report=False):
    """Deprecated model-level adapter for reciprocal shell weights."""
    if int(dim_k) != model.dim_k:
        raise ValueError("dim_k does not match the model")
    lattice = model.lattice if hasattr(model, "lattice") else model
    return lattice.k_shell_weights(
        nks,
        N_sh,
        return_shell=True,
        report=report,
    )


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


def finite_difference_periodic(M, axis, delta, order, mode="central"):
    """Differentiate a periodic array with wrapped finite differences."""
    return finite_difference(
        M,
        axis,
        delta,
        order,
        mode=mode,
        periodic=True,
    )


def finite_difference(
    M,
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
    values = np.asarray(M)
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


def is_Hermitian(M):
    """Return whether the trailing two axes form Hermitian matrices."""
    matrix = np.asarray(M, dtype=complex)
    if matrix.ndim == 0:
        return bool(np.allclose(matrix, matrix.conj()))
    if matrix.ndim == 1:
        return False
    return bool(np.allclose(matrix, matrix.conj().swapaxes(-1, -2)))


def get_trial_wfs(tf_list, norb, nspin=1):
    """Construct normalized trial wavefunctions from sparse amplitudes."""
    if nspin not in (1, 2):
        raise ValueError("nspin must be 1 or 2")
    shape = (
        (len(tf_list), int(norb))
        if nspin == 1
        else (len(tf_list), int(norb), 2)
    )
    result = np.zeros(shape, dtype=complex)
    for trial_index, trial in enumerate(tf_list):
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


def twf_generator(model, twf_list):
    """Deprecated model-shaped trial-wavefunction constructor."""
    return get_trial_wfs(twf_list, model.norb, model.nspin)


def no_2pi(x, clos):
    """Shift an angle by multiples of 2π to approach a reference."""
    while abs(clos - x) > np.pi:
        x += 2 * np.pi if clos - x > np.pi else -2 * np.pi
    return x


class PositionOperatorApproximationError(Exception):
    """Position-operator diagonality is invalid for a Wannier-derived model."""


def compute_d4k_and_d2k(delta_k):
    """Return a four-volume and all two-vector Gram areas."""
    vectors = np.asarray(delta_k, dtype=float)
    if vectors.shape != (4, 4):
        raise ValueError("delta_k must have shape (4, 4)")
    volume = abs(float(np.linalg.det(vectors)))
    areas = {}
    for first in range(4):
        for second in range(first + 1, 4):
            gram = vectors[[first, second]]
            area_squared = float(np.linalg.det(gram @ gram.T))
            areas[(first, second)] = np.sqrt(max(0.0, area_squared))
    return volume, areas


def pauli_decompose(M):
    """Return coefficients in the identity and Pauli-matrix basis."""
    values = np.asarray(M, dtype=complex)
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
