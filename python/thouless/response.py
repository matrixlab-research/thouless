"""Intrinsic band geometry and Berry-curvature response."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call, real_vector
from .model import Model


class BandResponse:
    """Rust-owned band response at one momentum.

    Arrays expose band energies, occupations, optional Fermi-surface weights,
    group velocities, and band-resolved Berry-curvature tensors computed from
    one shared eigensystem.
    """

    def __init__(self, native: object) -> None:
        if not isinstance(native, _core.NativeBandResponse):
            raise TypeError("BandResponse objects are produced by band_response")
        self._native = native

    @property
    def energies(self) -> np.ndarray:
        """Ascending band energies."""
        return np.asarray(self._native.energies, dtype=np.float64)

    @property
    def occupations(self) -> np.ndarray:
        """Fermi occupations at the requested chemical potential and temperature."""
        return np.asarray(self._native.occupations, dtype=np.float64)

    @property
    def negative_occupation_derivatives(self) -> np.ndarray | None:
        """Values of ``-df/dE``, or ``None`` at exact zero temperature."""
        values = self._native.negative_occupation_derivatives
        return None if values is None else np.asarray(values, dtype=np.float64)

    @property
    def group_velocities(self) -> np.ndarray:
        """Band-energy derivatives indexed by band and momentum direction."""
        return np.asarray(self._native.group_velocities, dtype=np.float64)

    @property
    def berry_curvatures(self) -> np.ndarray:
        """Band-resolved antisymmetric Berry-curvature tensors."""
        return np.asarray(self._native.berry_curvatures, dtype=np.float64)


def band_response(
    model: Model,
    momentum: npt.ArrayLike,
    *,
    chemical_potential: float,
    temperature: float,
    cartesian: bool = False,
    degeneracy_tolerance: float = 1.0e-10,
) -> BandResponse:
    """Evaluate band energies, occupations, velocities, and Berry curvatures.

    Args:
        model: Periodic tight-binding model.
        momentum: Reduced reciprocal coordinate.
        chemical_potential: Fermi level.
        temperature: Nonnegative thermal energy ``k_B T``.
        cartesian: Return derivatives in Cartesian reciprocal coordinates.
        degeneracy_tolerance: Energy threshold used to group degenerate bands
            in the gauge-covariant response.

    Returns:
        Rust-owned response arrays for one momentum.
    """
    return BandResponse(
        call(
            model._native.band_response,
            real_vector(momentum, name="momentum").tolist(),
            float(chemical_potential),
            float(temperature),
            bool(cartesian),
            float(degeneracy_tolerance),
        )
    )


def intrinsic_curvature(
    model: Model,
    momentum: npt.ArrayLike,
    *,
    chemical_potential: float,
    temperature: float,
    cartesian: bool = False,
    degeneracy_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Sum occupation-weighted Berry curvature at one momentum.

    Returns an antisymmetric tensor over momentum directions. Degenerate
    subspaces are treated covariantly before band occupations are applied.
    """
    return np.asarray(
        call(
            model._native.intrinsic_curvature,
            real_vector(momentum, name="momentum").tolist(),
            float(chemical_potential),
            float(temperature),
            bool(cartesian),
            float(degeneracy_tolerance),
        ),
        dtype=np.float64,
    )


def integrated_intrinsic_curvature(
    model: Model,
    shape: Sequence[int],
    fractional_offsets: npt.ArrayLike,
    *,
    chemical_potential: float,
    temperature: float,
    cartesian: bool = False,
    degeneracy_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Average intrinsic Berry curvature on a uniform reciprocal grid.

    Args:
        model: Periodic model.
        shape: Sample count on every periodic axis.
        fractional_offsets: Fractional grid offset on every axis.
        chemical_potential: Fermi level.
        temperature: Nonnegative thermal energy ``k_B T``.
        cartesian: Express the tensor in Cartesian reciprocal coordinates.
        degeneracy_tolerance: Degenerate-subspace energy threshold.

    Returns:
        Brillouin-zone mean of the occupation-weighted curvature tensor.
    """
    return np.asarray(
        call(
            model._native.integrated_intrinsic_curvature,
            [int(value) for value in shape],
            real_vector(fractional_offsets, name="fractional_offsets").tolist(),
            float(chemical_potential),
            float(temperature),
            bool(cartesian),
            float(degeneracy_tolerance),
        ),
        dtype=np.float64,
    )


def berry_curvature_dipole(
    samples: Sequence[BandResponse],
    weights: npt.ArrayLike,
    derivative_direction: int,
    curvature_first: int,
    curvature_second: int,
) -> float:
    """Integrate one Berry-curvature-dipole component.

    The integrand is ``(-df/dE) v_a Ω_bc`` using the selected derivative
    direction ``a`` and curvature plane ``(b, c)``. ``weights`` supplies one
    quadrature measure per momentum sample.
    """
    sample_weights = real_vector(weights, name="weights")
    if len(samples) != sample_weights.size:
        raise ValueError("samples and weights must have the same length")
    return float(
        call(
            _core.native_response_dipole,
            [sample._native for sample in samples],
            sample_weights.tolist(),
            int(derivative_direction),
            int(curvature_first),
            int(curvature_second),
        )
    )


def occupation_weighted_curvature(
    samples: Sequence[BandResponse],
    weights: npt.ArrayLike,
    first: int,
    second: int,
) -> float:
    """Integrate one occupation-weighted Berry-curvature component.

    ``first`` and ``second`` select zero-based tensor directions and
    ``weights`` supplies one quadrature measure per :class:`BandResponse`.
    """
    sample_weights = real_vector(weights, name="weights")
    if len(samples) != sample_weights.size:
        raise ValueError("samples and weights must have the same length")
    return float(
        call(
            _core.native_response_curvature_integral,
            [sample._native for sample in samples],
            sample_weights.tolist(),
            int(first),
            int(second),
        )
    )


__all__ = [
    "BandResponse",
    "band_response",
    "berry_curvature_dipole",
    "integrated_intrinsic_curvature",
    "intrinsic_curvature",
    "occupation_weighted_curvature",
]
