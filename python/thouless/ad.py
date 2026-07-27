"""Rust-native first-order differentiation workflows.

Finite differences are intentionally absent from this module.  Values,
directional derivatives, and gradients are evaluated by the same native Rust
rules used by the core crate.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call, complex_grid, complex_matrix, real_vector


def affine_projector_value_and_grad(
    base: npt.ArrayLike,
    directions: npt.ArrayLike,
    parameters: npt.ArrayLike,
    *,
    occupied: int,
    target: npt.ArrayLike,
    minimum_gap: float = 1.0e-8,
) -> tuple[float, np.ndarray]:
    """Return a gauge-invariant projector loss and physical gradient.

    The Hamiltonian family is ``H(theta) = base + sum_i theta[i] *
    directions[i]``.  ``target`` is the Hermitian projector defining the
    desired occupied subspace.
    """

    base_array = complex_matrix(base, name="base")
    direction_array = complex_grid(directions, name="directions")
    parameter_array = real_vector(parameters, name="parameters")
    target_array = complex_matrix(target, name="target")
    value, gradient = call(
        _core.ad_affine_projector_value_and_grad,
        base_array.tolist(),
        direction_array.tolist(),
        parameter_array.tolist(),
        int(occupied),
        target_array.tolist(),
        float(minimum_gap),
    )
    return float(value), np.asarray(gradient, dtype=np.float64)


def affine_projector_jvp(
    base: npt.ArrayLike,
    directions: npt.ArrayLike,
    parameters: npt.ArrayLike,
    direction: npt.ArrayLike,
    *,
    occupied: int,
    target: npt.ArrayLike,
    minimum_gap: float = 1.0e-8,
) -> tuple[float, float]:
    """Return the projector loss and one analytic directional derivative."""

    base_array = complex_matrix(base, name="base")
    direction_array = complex_grid(directions, name="directions")
    parameter_array = real_vector(parameters, name="parameters")
    tangent_array = real_vector(direction, name="direction")
    target_array = complex_matrix(target, name="target")
    value, derivative = call(
        _core.ad_affine_projector_jvp,
        base_array.tolist(),
        direction_array.tolist(),
        parameter_array.tolist(),
        tangent_array.tolist(),
        int(occupied),
        target_array.tolist(),
        float(minimum_gap),
    )
    return float(value), float(derivative)


__all__ = ["affine_projector_jvp", "affine_projector_value_and_grad"]
