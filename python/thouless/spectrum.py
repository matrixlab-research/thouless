"""Dense periodic spectra and one-dimensional lead bands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call, complex_matrix


@dataclass(frozen=True)
class Eigensystem:
    """Ascending eigenvalues and column eigenvectors."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass(frozen=True)
class BandEvaluation:
    """Lead-band energies and requested derivatives."""

    energies: np.ndarray
    first_derivatives: np.ndarray | None
    second_derivatives: np.ndarray | None
    eigenvectors: np.ndarray | None


class PeriodicBands:
    """One principal-cell Hamiltonian and neighboring-cell hopping."""

    def __init__(
        self,
        cell_hamiltonian: npt.ArrayLike,
        inter_cell_hopping: npt.ArrayLike,
    ) -> None:
        self._cell = complex_matrix(cell_hamiltonian, name="cell_hamiltonian")
        self._hopping = complex_matrix(
            inter_cell_hopping,
            name="inter_cell_hopping",
        )
        call(_core.validate_periodic_bands, self._cell.tolist(), self._hopping.tolist())

    def evaluate(
        self,
        momentum: float,
        *,
        derivative_order: int = 0,
        eigenvectors: bool = False,
    ) -> BandEvaluation:
        energies, first, second, vectors = call(
            _core.lead_band_evaluation,
            self._cell.tolist(),
            self._hopping.tolist(),
            float(momentum),
            int(derivative_order),
            bool(eigenvectors),
        )
        return BandEvaluation(
            np.asarray(energies, dtype=np.float64),
            None if first is None else np.asarray(first, dtype=np.float64),
            None if second is None else np.asarray(second, dtype=np.float64),
            None if vectors is None else np.asarray(vectors, dtype=np.complex128),
        )


def hermitian_eigensystem(matrix: npt.ArrayLike) -> Eigensystem:
    """Diagonalize a dense Hermitian matrix."""
    value = complex_matrix(matrix, name="matrix")
    energies, vectors = call(_core.matrix_eigensystem, value.tolist())
    return Eigensystem(
        np.asarray(energies, dtype=np.float64),
        np.asarray(vectors, dtype=np.complex128),
    )


__all__ = [
    "BandEvaluation",
    "Eigensystem",
    "PeriodicBands",
    "hermitian_eigensystem",
]
