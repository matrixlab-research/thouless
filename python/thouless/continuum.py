"""Finite-difference and magnetic ladder primitives."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from . import _core
from ._binding import call


@dataclass(frozen=True)
class StencilTerm:
    wave_offset: np.ndarray
    weight: complex
    inverse_spacing_powers: np.ndarray
    shifted_coefficients: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]


def finite_difference_stencil(
    dimension: int,
    factors: Sequence[tuple[int | None, int, int]],
) -> list[StencilTerm]:
    result = call(
        _core.continuum_finite_difference_stencil,
        int(dimension),
        [
            (
                None if axis is None else int(axis),
                int(identifier),
                int(power),
            )
            for axis, identifier, power in factors
        ],
    )
    return [
        StencilTerm(
            np.asarray(offset, dtype=np.int32),
            complex(weight),
            np.asarray(powers, dtype=np.uint32),
            tuple(
                (
                    int(identifier),
                    tuple(
                        (int(numerator), int(denominator))
                        for numerator, denominator in shifts
                    ),
                )
                for identifier, shifts in coefficients
            ),
        )
        for offset, weight, powers, coefficients in result
    ]


def landau_ladder_coefficient(
    ladder_powers: Sequence[int],
    initial_level: int,
    magnetic_field: float,
) -> float:
    return float(
        call(
            _core.continuum_landau_ladder_coefficient,
            [int(value) for value in ladder_powers],
            int(initial_level),
            float(magnetic_field),
        )
    )


__all__ = [
    "StencilTerm",
    "finite_difference_stencil",
    "landau_ladder_coefficient",
]
