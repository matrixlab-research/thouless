"""Steady-state coherent transport and periodic lead modes."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call, complex_matrix


@dataclass(frozen=True)
class Lead:
    """One semi-infinite periodic lead coupled to a finite device."""

    cell_hamiltonian: np.ndarray
    inter_cell_hopping: np.ndarray
    coupling: np.ndarray

    def __init__(
        self,
        cell_hamiltonian: npt.ArrayLike,
        inter_cell_hopping: npt.ArrayLike,
        coupling: npt.ArrayLike,
    ) -> None:
        object.__setattr__(
            self,
            "cell_hamiltonian",
            complex_matrix(cell_hamiltonian, name="cell_hamiltonian"),
        )
        object.__setattr__(
            self,
            "inter_cell_hopping",
            complex_matrix(inter_cell_hopping, name="inter_cell_hopping"),
        )
        object.__setattr__(
            self,
            "coupling",
            complex_matrix(coupling, name="coupling"),
        )

    def _native_input(self) -> tuple[list[list[complex]], ...]:
        return (
            self.cell_hamiltonian.tolist(),
            self.inter_cell_hopping.tolist(),
            self.coupling.tolist(),
        )


@dataclass(frozen=True)
class ScatteringResult:
    retarded_green: np.ndarray
    self_energies: np.ndarray
    broadenings: np.ndarray
    transmissions: np.ndarray


@dataclass(frozen=True)
class PropagatingModes:
    wave_functions: np.ndarray
    velocities: np.ndarray
    momenta: np.ndarray
    incoming_count: int
    stabilized_vectors: np.ndarray
    stabilized_vectors_lambda_inverse: np.ndarray
    square_root_hopping: np.ndarray


def solve(
    device_hamiltonian: npt.ArrayLike,
    leads: Sequence[Lead],
    energy: float,
    *,
    broadening: float | None = None,
) -> ScatteringResult:
    result = call(
        _core.open_system_solution,
        complex_matrix(device_hamiltonian, name="device_hamiltonian").tolist(),
        [lead._native_input() for lead in leads],
        float(energy),
        None if broadening is None else float(broadening),
    )
    return ScatteringResult(
        np.asarray(result[0], dtype=np.complex128),
        np.asarray(result[1], dtype=np.complex128),
        np.asarray(result[2], dtype=np.complex128),
        np.asarray(result[3], dtype=np.float64),
    )


def lead_self_energy(
    cell_hamiltonian: npt.ArrayLike,
    inter_cell_hopping: npt.ArrayLike,
    energy: float = 0.0,
    *,
    broadening: float | None = None,
    maximum_rank: int | None = None,
) -> np.ndarray:
    return np.asarray(
        call(
            _core.lead_retarded_self_energy,
            complex_matrix(cell_hamiltonian, name="cell_hamiltonian").tolist(),
            complex_matrix(
                inter_cell_hopping,
                name="inter_cell_hopping",
            ).tolist(),
            float(energy),
            None if broadening is None else float(broadening),
            maximum_rank,
        ),
        dtype=np.complex128,
    )


def propagating_modes(
    cell_hamiltonian: npt.ArrayLike,
    inter_cell_hopping: npt.ArrayLike,
) -> PropagatingModes:
    result = call(
        _core.lead_propagating_modes,
        complex_matrix(cell_hamiltonian, name="cell_hamiltonian").tolist(),
        complex_matrix(inter_cell_hopping, name="inter_cell_hopping").tolist(),
    )
    return PropagatingModes(
        np.asarray(result[0], dtype=np.complex128),
        np.asarray(result[1], dtype=np.float64),
        np.asarray(result[2], dtype=np.float64),
        int(result[3]),
        np.asarray(result[4], dtype=np.complex128),
        np.asarray(result[5], dtype=np.complex128),
        np.asarray(result[6], dtype=np.complex128),
    )


def partition_shot_noise(reflection_amplitudes: npt.ArrayLike) -> float:
    return float(
        call(
            _core.reflection_shot_noise,
            complex_matrix(
                reflection_amplitudes,
                name="reflection_amplitudes",
            ).tolist(),
        )
    )


__all__ = [
    "Lead",
    "PropagatingModes",
    "ScatteringResult",
    "lead_self_energy",
    "partition_shot_noise",
    "propagating_modes",
    "solve",
]
