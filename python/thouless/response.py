"""Intrinsic band geometry and Berry-curvature response."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call, real_vector
from .model import Model


class BandResponse:
    """Rust-owned band response at one momentum."""

    def __init__(self, native: object) -> None:
        if not isinstance(native, _core.NativeBandResponse):
            raise TypeError("BandResponse objects are produced by band_response")
        self._native = native

    @property
    def energies(self) -> np.ndarray:
        return np.asarray(self._native.energies, dtype=np.float64)

    @property
    def occupations(self) -> np.ndarray:
        return np.asarray(self._native.occupations, dtype=np.float64)

    @property
    def negative_occupation_derivatives(self) -> np.ndarray | None:
        values = self._native.negative_occupation_derivatives
        return None if values is None else np.asarray(values, dtype=np.float64)

    @property
    def group_velocities(self) -> np.ndarray:
        return np.asarray(self._native.group_velocities, dtype=np.float64)

    @property
    def berry_curvatures(self) -> np.ndarray:
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
    """Integrate the Fermi-surface dipole in the shared Rust core."""
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
    """Integrate an occupation-weighted curvature component in Rust."""
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
