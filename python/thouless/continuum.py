"""Finite-difference and magnetic ladder primitives."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from . import _core
from ._binding import call


@dataclass(frozen=True)
class StencilTerm:
    """One contribution to a coordinate-independent finite-difference stencil.

    Attributes:
        wave_offset: Integer displacement of the wave-function sample.
        weight: Dimensionless complex central-difference coefficient.
        inverse_spacing_powers: Power of the inverse grid spacing on each axis.
        shifted_coefficients: Symbolic coefficient identifiers and their exact
            rational evaluation shifts, represented as ``(numerator,
            denominator)`` pairs for every axis.
    """

    wave_offset: np.ndarray
    weight: complex
    inverse_spacing_powers: np.ndarray
    shifted_coefficients: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]


def finite_difference_stencil(
    dimension: int,
    factors: Sequence[tuple[int | None, int, int]],
) -> list[StencilTerm]:
    """Discretize an ordered product of coefficients and momentum operators.

    Args:
        dimension: Number of discretized coordinate axes.
        factors: Ordered factors. ``(None, identifier, 1)`` denotes an opaque
            coefficient and ``(axis, identifier, power)`` denotes
            ``(-i d/dx_axis)**power``; the identifier is ignored for momentum
            factors. Axes use zero-based Python indexing.

    Returns:
        Exact stencil terms whose offsets, spacing powers, coefficient shifts,
        and complex weights can be assembled into a lattice Hamiltonian.

    Raises:
        ThoulessError: If the dimension, axis, or operator power is invalid.
    """
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
    """Evaluate an ordered Landau-level ladder-operator matrix element.

    Args:
        ladder_powers: Ordered signed ladder powers. Positive values apply
            creation operators and negative values apply annihilation
            operators.
        initial_level: Nonnegative Landau level on which the product acts.
        magnetic_field: Signed magnetic-field strength in the convention used
            by the continuum Hamiltonian.

    Returns:
        The real scalar multiplying the surviving Landau-level state, or zero
        if an annihilation step crosses below level zero.

    Raises:
        ThoulessError: If a level, power, or field value is invalid.
    """
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
