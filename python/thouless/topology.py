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
    values: np.ndarray
    spectator_shape: tuple[int, ...]


@dataclass(frozen=True)
class SecondChernResult:
    slice_densities: np.ndarray
    value: float


def wilson_phase(frames: npt.ArrayLike) -> float:
    return float(
        call(
            _core.wilson_phase,
            complex_grid(frames, name="frames").tolist(),
        )
    )


def wilson_eigenphases(frames: npt.ArrayLike) -> np.ndarray:
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
    return np.asarray(
        call(
            _core.transport_link,
            complex_matrix(left, name="left").tolist(),
            complex_matrix(right, name="right").tolist(),
        ),
        dtype=np.complex128,
    )


def berry_flux(corners: npt.ArrayLike) -> float:
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
