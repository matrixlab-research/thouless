"""Random-matrix ensembles and deterministic random-access variates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import _core
from ._binding import call


def uniform_pair(data: bytes, salt: bytes = b"") -> tuple[float, float]:
    """Map arbitrary bytes deterministically to two independent uniforms.

    The output lies in the half-open interval ``[0, 1)`` and is stable across processes
    and supported languages. ``salt`` defines an independent random-access
    stream without mutable generator state.
    """
    first, second = call(_core.digest_uniform_pair, list(data), list(salt))
    return float(first), float(second)


def uniform(data: bytes, salt: bytes = b"") -> float:
    """Map arbitrary bytes deterministically to one uniform in ``[0, 1)``."""
    return float(call(_core.digest_uniform, list(data), list(salt)))


def gaussian(data: bytes, salt: bytes = b"") -> float:
    """Map arbitrary bytes deterministically to a standard-normal variate."""
    return float(call(_core.digest_gaussian, list(data), list(salt)))


def gaussian_matrix(
    dimension: int,
    symmetry_class: str,
    variance: float,
    real_components: Sequence[float],
    imaginary_components: Sequence[float],
) -> np.ndarray:
    """Project independent normal components onto a Gaussian symmetry ensemble.

    Args:
        dimension: Matrix dimension.
        symmetry_class: Altland-Zirnbauer label such as ``"A"``, ``"AII"``,
            or ``"DIII"``.
        variance: Target variance scale.
        real_components: Independent standard-normal real components.
        imaginary_components: Independent standard-normal imaginary components.

    Returns:
        Hermitian matrix satisfying the selected symmetry constraints.

    Notes:
        Random-number generation is deliberately external. The native kernel
        validates component counts and performs only deterministic projection.
    """
    return np.asarray(
        call(
            _core.rmt_gaussian,
            int(dimension),
            str(symmetry_class),
            float(variance),
            [float(value) for value in real_components],
            [float(value) for value in imaginary_components],
        ),
        dtype=np.complex128,
    )


def circular_matrix(
    dimension: int,
    symmetry_class: str,
    real_components: Sequence[float],
    imaginary_components: Sequence[float],
    random_bits: Sequence[bool],
    *,
    topological_sector: int | None = None,
) -> np.ndarray:
    """Project independent components onto a circular symmetry ensemble.

    Args:
        dimension: Matrix dimension.
        symmetry_class: Altland-Zirnbauer class label.
        real_components: Independent standard-normal real components.
        imaginary_components: Independent standard-normal imaginary components.
        random_bits: Independent signs used by disconnected ensemble sectors.
        topological_sector: Optional class-specific topological sector.

    Returns:
        Unitary matrix satisfying the selected symmetry constraints.
    """
    return np.asarray(
        call(
            _core.rmt_circular,
            int(dimension),
            str(symmetry_class),
            [float(value) for value in real_components],
            [float(value) for value in imaginary_components],
            [bool(value) for value in random_bits],
            topological_sector,
        ),
        dtype=np.complex128,
    )


__all__ = [
    "circular_matrix",
    "gaussian",
    "gaussian_matrix",
    "uniform",
    "uniform_pair",
]
