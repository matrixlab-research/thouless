"""Random-matrix ensembles and deterministic random-access variates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import _core
from ._binding import call


def uniform_pair(data: bytes, salt: bytes = b"") -> tuple[float, float]:
    first, second = call(_core.digest_uniform_pair, list(data), list(salt))
    return float(first), float(second)


def uniform(data: bytes, salt: bytes = b"") -> float:
    return float(call(_core.digest_uniform, list(data), list(salt)))


def gaussian(data: bytes, salt: bytes = b"") -> float:
    return float(call(_core.digest_gaussian, list(data), list(salt)))


def gaussian_matrix(
    dimension: int,
    symmetry_class: str,
    variance: float,
    real_components: Sequence[float],
    imaginary_components: Sequence[float],
) -> np.ndarray:
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
