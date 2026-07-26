"""Sampled-frame projection, localization, and interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call, complex_grid, complex_matrix


@dataclass(frozen=True)
class Spread:
    centers: np.ndarray
    spreads: np.ndarray
    invariant: float
    diagonal: float
    off_diagonal: float


@dataclass(frozen=True)
class Localization:
    frames: np.ndarray
    initial_spread: float
    final_spread: float
    gradient_norm: float
    iterations: int
    converged: bool


def project_trials(
    frames: npt.ArrayLike,
    trials: npt.ArrayLike,
    *,
    singular_tolerance: float = 1.0e-10,
) -> np.ndarray:
    return np.asarray(
        call(
            _core.wannier_project_trials,
            complex_grid(frames, name="frames").tolist(),
            complex_matrix(trials, name="trials").tolist(),
            float(singular_tolerance),
        ),
        dtype=np.complex128,
    )


def periodic_overlaps(
    mesh_shape: Sequence[int],
    frames: npt.ArrayLike,
    displacements: Sequence[Sequence[int]],
    boundary_twists: npt.ArrayLike,
) -> np.ndarray:
    return np.asarray(
        call(
            _core.wannier_periodic_overlaps,
            [int(value) for value in mesh_shape],
            complex_grid(frames, name="frames").tolist(),
            [[int(value) for value in row] for row in displacements],
            np.asarray(boundary_twists, dtype=np.complex128).tolist(),
        ),
        dtype=np.complex128,
    )


def spread_decomposition(
    overlaps: npt.ArrayLike,
    neighbor_vectors: npt.ArrayLike,
    neighbor_weights: npt.ArrayLike,
) -> Spread:
    centers, spreads, invariant, diagonal, off_diagonal = call(
        _core.wannier_spread_decomposition,
        np.asarray(overlaps, dtype=np.complex128).tolist(),
        np.asarray(neighbor_vectors, dtype=np.float64).tolist(),
        np.asarray(neighbor_weights, dtype=np.float64).tolist(),
    )
    return Spread(
        np.asarray(centers, dtype=np.float64),
        np.asarray(spreads, dtype=np.float64),
        float(invariant),
        float(diagonal),
        float(off_diagonal),
    )


def maximize_localization(
    mesh_shape: Sequence[int],
    frames: npt.ArrayLike,
    displacements: Sequence[Sequence[int]],
    boundary_twists: npt.ArrayLike,
    neighbor_vectors: npt.ArrayLike,
    neighbor_weights: npt.ArrayLike,
    *,
    step_scale: float = 0.5,
    max_iterations: int = 1000,
    spread_tolerance: float = 1.0e-5,
    gradient_tolerance: float = 1.0e-3,
) -> Localization:
    result = call(
        _core.wannier_maximize_localization,
        [int(value) for value in mesh_shape],
        complex_grid(frames, name="frames").tolist(),
        [[int(value) for value in row] for row in displacements],
        np.asarray(boundary_twists, dtype=np.complex128).tolist(),
        np.asarray(neighbor_vectors, dtype=np.float64).tolist(),
        np.asarray(neighbor_weights, dtype=np.float64).tolist(),
        float(step_scale),
        int(max_iterations),
        float(spread_tolerance),
        float(gradient_tolerance),
    )
    return Localization(
        np.asarray(result[0], dtype=np.complex128),
        float(result[1]),
        float(result[2]),
        float(result[3]),
        int(result[4]),
        bool(result[5]),
    )


def inverse_bloch_transform(
    mesh_shape: Sequence[int],
    frames: npt.ArrayLike,
) -> np.ndarray:
    return np.asarray(
        call(
            _core.wannier_inverse_bloch_transform,
            [int(value) for value in mesh_shape],
            complex_grid(frames, name="frames").tolist(),
        ),
        dtype=np.complex128,
    )


def interpolate_matrices(
    mesh_shape: Sequence[int],
    samples: npt.ArrayLike,
    points: npt.ArrayLike,
) -> np.ndarray:
    return np.asarray(
        call(
            _core.wannier_interpolate_matrices,
            [int(value) for value in mesh_shape],
            complex_grid(samples, name="samples").tolist(),
            np.asarray(points, dtype=np.float64).tolist(),
        ),
        dtype=np.complex128,
    )


__all__ = [
    "Localization",
    "Spread",
    "interpolate_matrices",
    "inverse_bloch_transform",
    "maximize_localization",
    "periodic_overlaps",
    "project_trials",
    "spread_decomposition",
]
