"""Discrete unitary and antiunitary symmetry constraints."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call, complex_matrix


def _optional_matrix(value: npt.ArrayLike | None, name: str) -> object:
    return None if value is None else complex_matrix(value, name=name).tolist()


@dataclass(frozen=True)
class DiscreteSymmetry:
    """Validated conservation, time-reversal, particle-hole, and chiral data.

    Args:
        projectors: Optional complete orthogonal projectors for conserved
            sectors.
        time_reversal: Unitary part of an antiunitary time-reversal operation.
        particle_hole: Unitary part of an antiunitary particle-hole operation.
        chiral: Unitary chiral-symmetry operator.

    The constructor normalizes matrices in Rust and rejects inconsistent
    dimensions, non-unitary operators, invalid antiunitary squares, or
    incompatible combinations.
    """

    projectors: tuple[np.ndarray, ...] | None = None
    time_reversal: np.ndarray | None = None
    particle_hole: np.ndarray | None = None
    chiral: np.ndarray | None = None

    def __init__(
        self,
        *,
        projectors: Sequence[npt.ArrayLike] | None = None,
        time_reversal: npt.ArrayLike | None = None,
        particle_hole: npt.ArrayLike | None = None,
        chiral: npt.ArrayLike | None = None,
    ) -> None:
        normalized = call(
            _core.discrete_symmetry_normalize,
            None
            if projectors is None
            else [
                complex_matrix(value, name="projector").tolist()
                for value in projectors
            ],
            _optional_matrix(time_reversal, "time_reversal"),
            _optional_matrix(particle_hole, "particle_hole"),
            _optional_matrix(chiral, "chiral"),
        )
        object.__setattr__(
            self,
            "projectors",
            None
            if normalized[0] is None
            else tuple(
                np.asarray(value, dtype=np.complex128)
                for value in normalized[0]
            ),
        )
        object.__setattr__(
            self,
            "time_reversal",
            None
            if normalized[1] is None
            else np.asarray(normalized[1], dtype=np.complex128),
        )
        object.__setattr__(
            self,
            "particle_hole",
            None
            if normalized[2] is None
            else np.asarray(normalized[2], dtype=np.complex128),
        )
        object.__setattr__(
            self,
            "chiral",
            None
            if normalized[3] is None
            else np.asarray(normalized[3], dtype=np.complex128),
        )

    def validate(self, matrix: npt.ArrayLike) -> tuple[str, ...]:
        """Return labels of declared symmetries violated by ``matrix``.

        A square matrix is treated as an onsite operator. A left-aligned
        rectangular matrix is treated as a hopping block between compatible
        symmetry sectors.
        """
        return tuple(
            call(
                _core.discrete_symmetry_validate,
                None
                if self.projectors is None
                else [value.tolist() for value in self.projectors],
                None
                if self.time_reversal is None
                else self.time_reversal.tolist(),
                None
                if self.particle_hole is None
                else self.particle_hole.tolist(),
                None if self.chiral is None else self.chiral.tolist(),
                complex_matrix(matrix, name="matrix").tolist(),
            )
        )


def particle_hole_basis(
    wave_functions: npt.ArrayLike,
    particle_hole: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a canonical basis for a particle-hole-closed subspace.

    Args:
        wave_functions: Column wave functions spanning the subspace.
        particle_hole: Unitary part of the antiunitary particle-hole operator.

    Returns:
        Canonicalized column vectors and their deterministic source ordering.

    Raises:
        ThoulessError: If the subspace is not closed, has odd dimension, or a
            stable canonical basis cannot be constructed.
    """
    vectors, ordering = call(
        _core.particle_hole_basis,
        complex_matrix(wave_functions, name="wave_functions").tolist(),
        complex_matrix(particle_hole, name="particle_hole").tolist(),
    )
    return (
        np.asarray(vectors, dtype=np.complex128),
        np.asarray(ordering, dtype=np.int64),
    )


__all__ = ["DiscreteSymmetry", "particle_hole_basis"]
