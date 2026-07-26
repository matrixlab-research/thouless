"""Kernel-polynomial spectra and correlation response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call, complex_grid, complex_matrix


@dataclass(frozen=True)
class RescaledHamiltonian:
    matrix: np.ndarray
    half_width: float
    center: float


@dataclass(frozen=True)
class Reconstruction:
    energies: np.ndarray
    densities: np.ndarray
    gammas: np.ndarray
    moments: np.ndarray


def rescale(
    hamiltonian: npt.ArrayLike,
    *,
    strict_margin: float = 0.05,
    bounds: tuple[float, float] | None = None,
) -> RescaledHamiltonian:
    matrix, half_width, center = call(
        _core.kpm_rescale_hamiltonian,
        complex_matrix(hamiltonian, name="hamiltonian").tolist(),
        float(strict_margin),
        bounds,
    )
    return RescaledHamiltonian(
        np.asarray(matrix, dtype=np.complex128),
        float(half_width),
        float(center),
    )


def chebyshev_vectors(
    rescaled_hamiltonian: npt.ArrayLike,
    initial_vectors: npt.ArrayLike,
    moment_count: int,
) -> np.ndarray:
    return np.asarray(
        call(
            _core.kpm_chebyshev_vectors,
            complex_matrix(
                rescaled_hamiltonian,
                name="rescaled_hamiltonian",
            ).tolist(),
            np.asarray(initial_vectors, dtype=np.complex128).tolist(),
            int(moment_count),
        ),
        dtype=np.complex128,
    )


def scalar_moments(
    initial_vectors: npt.ArrayLike,
    chebyshev: npt.ArrayLike,
    operator: npt.ArrayLike | None = None,
) -> np.ndarray:
    return np.asarray(
        call(
            _core.kpm_scalar_moments,
            np.asarray(initial_vectors, dtype=np.complex128).tolist(),
            complex_grid(chebyshev, name="chebyshev").tolist(),
            None
            if operator is None
            else complex_matrix(operator, name="operator").tolist(),
        ),
        dtype=np.complex128,
    )


def reconstruct(
    raw_moments: npt.ArrayLike,
    half_width: float,
    center: float,
    *,
    kernel: str = "jackson",
    kernel_strength: float | None = None,
    mean: bool = True,
) -> Reconstruction:
    result = call(
        _core.kpm_reconstruct,
        complex_grid(raw_moments, name="raw_moments").tolist(),
        float(half_width),
        float(center),
        str(kernel),
        kernel_strength,
        bool(mean),
    )
    return Reconstruction(
        np.asarray(result[0], dtype=np.float64),
        np.asarray(result[1], dtype=np.complex128),
        np.asarray(result[2], dtype=np.complex128),
        np.asarray(result[3], dtype=np.complex128),
    )


def fermi_distribution(
    energies: npt.ArrayLike,
    chemical_potential: float,
    temperature: float,
) -> np.ndarray:
    return np.asarray(
        call(
            _core.kpm_fermi_distribution,
            np.asarray(energies, dtype=np.float64).tolist(),
            float(chemical_potential),
            float(temperature),
        ),
        dtype=np.float64,
    )


__all__ = [
    "Reconstruction",
    "RescaledHamiltonian",
    "chebyshev_vectors",
    "fermi_distribution",
    "reconstruct",
    "rescale",
    "scalar_moments",
]
