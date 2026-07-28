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
    """Wannier centers and Marzari-Vanderbilt spread decomposition.

    Attributes:
        centers: Cartesian center of each Wannier function.
        spreads: Total spread of each Wannier function.
        invariant: Gauge-invariant contribution ``Ω_I``.
        diagonal: Gauge-dependent diagonal contribution ``Ω_D``.
        off_diagonal: Gauge-dependent off-diagonal contribution ``Ω_OD``.
    """

    centers: np.ndarray
    spreads: np.ndarray
    invariant: float
    diagonal: float
    off_diagonal: float


@dataclass(frozen=True)
class Localization:
    """Result and convergence diagnostics of spread minimization.

    Attributes:
        frames: Gauge-rotated Bloch frames on the input mesh.
        initial_spread: Gauge-dependent spread before minimization.
        final_spread: Gauge-dependent spread after the last accepted step.
        gradient_norm: Norm of the final anti-Hermitian gauge gradient.
        iterations: Number of attempted optimization iterations.
        converged: Whether the spread and gradient criteria were satisfied.
    """

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
    """Project localized trial orbitals into sampled Bloch subspaces.

    Each projected frame is Löwdin-orthonormalized. A sample whose trial
    overlap is rank deficient below ``singular_tolerance`` is rejected.
    """
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
    """Build neighbor overlaps on a periodic reciprocal mesh.

    Args:
        mesh_shape: Sample count on each reciprocal axis.
        frames: Orthonormal Bloch frames in flattened mesh order.
        displacements: Integer neighbor displacements.
        boundary_twists: Orbital phase factors applied when a displacement
            crosses each periodic boundary.

    Returns:
        Overlap matrices indexed by mesh point and displacement.
    """
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
    """Evaluate the discrete Marzari-Vanderbilt spread decomposition.

    Args:
        overlaps: Neighbor overlap matrices on the reciprocal mesh.
        neighbor_vectors: Cartesian neighbor vectors.
        neighbor_weights: Finite-difference weight of every neighbor.

    Returns:
        Wannier centers, per-orbital spreads, and invariant, diagonal, and
        off-diagonal total contributions.
    """
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
    """Minimize the gauge-dependent Wannier spread on a periodic mesh.

    The optimization updates only unitary rotations inside the sampled
    subspaces. It stops when both spread change and gradient norm satisfy their
    tolerances or when ``max_iterations`` is reached.
    """
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
    """Apply an inverse discrete Bloch transform to mesh-ordered frames.

    Returns cell-resolved Wannier amplitudes with the same internal and
    subspace dimensions as the sampled frames.
    """
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
    """Fourier-interpolate matrix samples from a uniform reciprocal mesh.

    Args:
        mesh_shape: Shape of the uniform source mesh.
        samples: Matrix values in flattened mesh order.
        points: Reduced reciprocal coordinates at which to interpolate.

    Returns:
        One interpolated complex matrix per requested point.
    """
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
