"""Kernel-polynomial spectra and correlation response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call, complex_grid, complex_matrix


@dataclass(frozen=True)
class RescaledHamiltonian:
    """Hamiltonian mapped into the open Chebyshev interval.

    Attributes:
        matrix: Rescaled Hermitian matrix.
        half_width: Positive energy scale used in ``H' = (H-center)/half_width``.
        center: Energy shift applied before rescaling.
    """

    matrix: np.ndarray
    half_width: float
    center: float


@dataclass(frozen=True)
class Reconstruction:
    """Kernel-polynomial reconstruction on the native Chebyshev grid.

    Attributes:
        energies: Physical energy samples.
        densities: Reconstructed values, optionally averaged over probes.
        gammas: Kernel-damped moments.
        moments: Input moments after the requested probe reduction.
    """

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
    """Map a Hermitian Hamiltonian strictly inside ``[-1, 1]``.

    Args:
        hamiltonian: Square Hermitian matrix.
        strict_margin: Fractional padding beyond the estimated spectral bounds.
        bounds: Optional explicit ``(lower, upper)`` spectral bounds. When
            omitted, the native eigensolver determines exact dense bounds.

    Returns:
        The rescaled matrix together with its physical half-width and center.
    """
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
    """Generate the Chebyshev recurrence for one or more probe vectors.

    Args:
        rescaled_hamiltonian: Square matrix whose spectrum lies within
            ``[-1, 1]``.
        initial_vectors: Probe vectors as rows with shape
            ``(probe_count, dimension)``.
        moment_count: Number of Chebyshev vectors, including orders zero and
            one.

    Returns:
        Complex array indexed by probe, moment, and state.
    """
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
    """Contract Chebyshev vectors into scalar moments.

    Args:
        initial_vectors: Probe vectors used to start the recurrence.
        chebyshev: Output of :func:`chebyshev_vectors`.
        operator: Optional operator inserted between the bra probes and
            Chebyshev vectors.

    Returns:
        Complex array indexed by probe, moment, and observable component.
    """
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
    """Reconstruct a spectral function from Chebyshev moments.

    Args:
        raw_moments: One moment sequence or a grid of probe sequences.
        half_width: Physical scale returned by :func:`rescale`.
        center: Physical energy shift returned by :func:`rescale`.
        kernel: Damping kernel, currently ``"jackson"`` or ``"lorentz"``.
        kernel_strength: Kernel-specific strength; the native default is used
            when omitted.
        mean: Average independent probe estimates before returning densities.

    Returns:
        Energy samples, reconstructed densities, damped moments, and reduced
        raw moments.
    """
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
    """Evaluate the Fermi-Dirac occupation with a stable zero-temperature limit.

    Args:
        energies: Physical energy samples.
        chemical_potential: Fermi level in the same units as ``energies``.
        temperature: Nonnegative thermal energy ``k_B T``.

    Returns:
        Occupations in ``[0, 1]`` with the same flattened length as
        ``energies``.
    """
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
