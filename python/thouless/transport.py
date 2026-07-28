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
    """One semi-infinite periodic lead coupled to a finite device.

    Args:
        cell_hamiltonian: Square Hermitian principal-cell Hamiltonian.
        inter_cell_hopping: Hopping from one lead cell to the next.
        coupling: Matrix mapping lead-cell states into device states.
    """

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
    """Open-system Green function and lead-resolved transport observables.

    Attributes:
        retarded_green: Device retarded Green function.
        self_energies: Lead self-energy matrices, one per lead.
        broadenings: Lead broadening matrices ``i(Σ-Σᴴ)``.
        transmissions: Lead-to-lead Caroli transmission matrix.
    """

    retarded_green: np.ndarray
    self_energies: np.ndarray
    broadenings: np.ndarray
    transmissions: np.ndarray


@dataclass(frozen=True)
class PropagatingModes:
    """Propagating and stabilized modes of a periodic lead.

    Attributes:
        wave_functions: Current-normalized propagating modes as columns.
        velocities: Signed group velocities in mode order.
        momenta: Reduced crystal momenta in mode order.
        incoming_count: Number of modes directed toward the device.
        stabilized_vectors: Full stabilized transfer basis.
        stabilized_vectors_lambda_inverse: Stabilized basis multiplied by
            inverse Bloch factors.
        square_root_hopping: Rank-revealing hopping factor used by the mode
            solver.
    """

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
    """Solve a coherent finite device attached to semi-infinite leads.

    Args:
        device_hamiltonian: Square Hermitian device Hamiltonian.
        leads: Lead definitions and their lead-to-device couplings.
        energy: Real scattering energy.
        broadening: Optional positive retarded regulator. The native default is
            used when omitted.

    Returns:
        Retarded Green function, self energies, broadenings, and all pairwise
        transmission probabilities.
    """
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
    """Compute the retarded surface self energy of a periodic lead.

    Args:
        cell_hamiltonian: Principal-cell Hamiltonian.
        inter_cell_hopping: Hopping from one principal cell to the next.
        energy: Real evaluation energy.
        broadening: Optional positive retarded regulator.
        maximum_rank: Optional cap for the rank-revealing stabilized basis.

    Returns:
        Retarded self-energy matrix acting on the principal cell.
    """
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
    """Solve current-normalized propagating and evanescent lead modes.

    Incoming modes precede outgoing modes in the propagating arrays. The
    stabilized arrays retain the evanescent information required by transport
    assembly.
    """
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
    """Return dimensionless partition shot noise from a reflection matrix.

    The value is ``Tr[T(1-T)]`` with transmission eigenvalues inferred from
    ``T = I - rᴴr``.
    """
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
