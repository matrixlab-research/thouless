"""Gauge-covariant topology and quantum geometry."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call, complex_grid, complex_matrix, real_matrix
from .model import Model


@dataclass(frozen=True)
class ChernResult:
    """First-Chern values over optional spectator momentum axes.

    Attributes:
        values: Integrated Chern values in flattened spectator-grid order.
        spectator_shape: Grid shape corresponding to ``values``.
    """

    values: np.ndarray
    spectator_shape: tuple[int, ...]


@dataclass(frozen=True)
class SecondChernResult:
    """Four-dimensional second-Chern integral and slice-resolved densities.

    Attributes:
        slice_densities: Integrated density for each fourth-coordinate slice.
        value: Total second-Chern value after the fourth-axis quadrature.
    """

    slice_densities: np.ndarray
    value: float


def wilson_phase(frames: npt.ArrayLike) -> float:
    """Return the gauge-invariant determinant Wilson-loop phase.

    ``frames`` is a closed sequence of occupied column frames. Neighboring
    overlaps are unitarized before their determinants are accumulated.
    """
    return float(
        call(
            _core.wilson_phase,
            complex_grid(frames, name="frames").tolist(),
        )
    )


def wilson_eigenphases(frames: npt.ArrayLike) -> np.ndarray:
    """Return eigenphases of the non-Abelian Wilson loop for closed frames.

    Args:
        frames: Sequence of equal-rank orthonormal column frames.

    Returns:
        Sorted phases in radians on the principal branch.
    """
    return np.asarray(
        call(
            _core.wilson_eigenphases,
            complex_grid(frames, name="frames").tolist(),
        ),
        dtype=np.float64,
    )


def parallel_transport(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
) -> np.ndarray:
    """Return the unitary link that parallel-transports ``right`` to ``left``.

    Both inputs store equal-dimensional orthonormal frames as columns. The
    overlap is projected to its closest unitary factor.
    """
    return np.asarray(
        call(
            _core.transport_link,
            complex_matrix(left, name="left").tolist(),
            complex_matrix(right, name="right").tolist(),
        ),
        dtype=np.complex128,
    )


def berry_flux(corners: npt.ArrayLike) -> float:
    """Evaluate gauge-invariant Berry flux through one oriented plaquette.

    The first axis enumerates the four corner frames in cyclic order.
    """
    return float(
        call(
            _core.berry_flux,
            complex_grid(corners, name="corners").tolist(),
        )
    )


def chern_numbers(
    model: Model,
    samples: Sequence[int],
    plane: tuple[int, int],
    occupied_states: Sequence[int],
) -> ChernResult:
    """Integrate first Chern numbers on a uniform model momentum grid.

    Args:
        model: Periodic tight-binding model.
        samples: Grid size on every periodic axis.
        plane: Pair of zero-based periodic axes spanning each integration plane.
        occupied_states: Zero-based band indices forming the occupied frame.

    Returns:
        Integrated values and the shape of all spectator-axis samples.
    """
    values, shape = call(
        _core.uniform_grid_chern,
        *model._export(),
        [int(value) for value in samples],
        [int(plane[0]), int(plane[1])],
        [int(value) for value in occupied_states],
    )
    return ChernResult(
        np.asarray(values, dtype=np.float64),
        tuple(int(value) for value in shape),
    )


def quantum_geometric_tensor(
    hamiltonians: npt.ArrayLike,
    derivatives: npt.ArrayLike,
    occupied_states: Sequence[int],
) -> np.ndarray:
    """Evaluate the occupied-subspace quantum geometric tensor by Kubo sums.

    ``hamiltonians`` is a batch of square Hermitian matrices and
    ``derivatives`` stores the corresponding momentum derivatives. The final
    tensor combines its symmetric quantum metric and antisymmetric Berry
    curvature.
    """
    hamiltonian_batch = np.asarray(hamiltonians, dtype=np.complex128)
    derivative_batch = np.asarray(derivatives, dtype=np.complex128)
    return np.asarray(
        call(
            _core.quantum_geometric_tensor_kubo,
            hamiltonian_batch.tolist(),
            derivative_batch.tolist(),
            [int(value) for value in occupied_states],
        ),
        dtype=np.complex128,
    )


def local_chern_marker(
    hamiltonian: npt.ArrayLike,
    positions: npt.ArrayLike,
    occupied_states: Sequence[int],
    cell_area: float,
) -> np.ndarray:
    """Evaluate the real-space local Chern marker for a finite Hamiltonian.

    Args:
        hamiltonian: Finite square Hermitian Hamiltonian.
        positions: Cartesian position of each basis state; at least two columns
            are required.
        occupied_states: Zero-based eigenstate indices below the chosen gap.
        cell_area: Positive real-space normalization area.

    Returns:
        One dimensionless local marker per basis state.
    """
    return np.asarray(
        call(
            _core.local_chern_marker_kubo,
            complex_matrix(hamiltonian, name="hamiltonian").tolist(),
            real_matrix(positions, name="positions").tolist(),
            [int(value) for value in occupied_states],
            float(cell_area),
        ),
        dtype=np.float64,
    )


def second_chern(
    hamiltonians: npt.ArrayLike,
    derivatives: npt.ArrayLike,
    grid_shape: Sequence[int],
    coordinate_steps: npt.ArrayLike,
    *,
    fourth_axis_periodic: bool,
    occupied_states: Sequence[int],
) -> SecondChernResult:
    """Integrate the second Chern density on a four-coordinate grid.

    Args:
        hamiltonians: Grid of square Hermitian matrices.
        derivatives: Four Hamiltonian derivatives at every grid point.
        grid_shape: Four-dimensional grid shape.
        coordinate_steps: Physical spacing of the four coordinates.
        fourth_axis_periodic: Whether the last integration axis closes
            periodically.
        occupied_states: Zero-based occupied-band indices.

    Returns:
        Slice-resolved densities along the fourth coordinate and their total
        second-Chern value.
    """
    slices, value = call(
        _core.second_chern_kubo,
        np.asarray(hamiltonians, dtype=np.complex128).tolist(),
        np.asarray(derivatives, dtype=np.complex128).tolist(),
        [int(item) for item in grid_shape],
        np.asarray(coordinate_steps, dtype=np.float64).tolist(),
        bool(fourth_axis_periodic),
        [int(item) for item in occupied_states],
    )
    return SecondChernResult(np.asarray(slices, dtype=np.float64), float(value))


__all__ = [
    "ChernResult",
    "SecondChernResult",
    "berry_flux",
    "chern_numbers",
    "local_chern_marker",
    "parallel_transport",
    "quantum_geometric_tensor",
    "second_chern",
    "wilson_eigenphases",
    "wilson_phase",
]
