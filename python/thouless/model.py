"""First-class tight-binding model objects backed by Rust ownership."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call, complex_matrix, real_matrix, real_vector


@dataclass(frozen=True)
class Lattice:
    """A Cartesian primitive-vector frame and the axes treated as periodic."""

    primitive_vectors: np.ndarray
    periodic_axes: tuple[int, ...]

    def __init__(
        self,
        primitive_vectors: npt.ArrayLike,
        periodic_axes: Sequence[int],
    ) -> None:
        vectors = real_matrix(primitive_vectors, name="primitive_vectors")
        axes = tuple(int(axis) for axis in periodic_axes)
        call(_core.NativeModelBuilder, vectors.tolist(), list(axes))
        object.__setattr__(self, "primitive_vectors", vectors)
        object.__setattr__(self, "periodic_axes", axes)

    @classmethod
    def finite(cls, dimension: int) -> "Lattice":
        """Construct a finite identity coordinate frame."""
        return cls(np.eye(dimension, dtype=np.float64), ())

    @property
    def real_dimension(self) -> int:
        return int(self.primitive_vectors.shape[0])

    @property
    def periodic_dimension(self) -> int:
        return len(self.periodic_axes)


class ModelBuilder:
    """Exclusive mutable construction of an immutable Rust model."""

    def __init__(self, lattice: Lattice) -> None:
        self._native = call(
            _core.NativeModelBuilder,
            lattice.primitive_vectors.tolist(),
            list(lattice.periodic_axes),
        )

    def add_orbital(
        self,
        label: str,
        reduced_position: npt.ArrayLike,
        *,
        degrees_of_freedom: int = 1,
    ) -> int:
        position = real_vector(reduced_position, name="reduced_position")
        return int(
            call(
                self._native.add_orbital,
                str(label),
                position.tolist(),
                int(degrees_of_freedom),
            )
        )

    def set_onsite(self, orbital: int, energy: float) -> "ModelBuilder":
        call(self._native.set_onsite, int(orbital), float(energy))
        return self

    def set_onsite_block(
        self,
        orbital: int,
        block: npt.ArrayLike,
    ) -> "ModelBuilder":
        matrix = complex_matrix(block, name="block")
        call(self._native.set_onsite_block, int(orbital), matrix.tolist())
        return self

    def add_hopping(
        self,
        target: int,
        source: int,
        cell_offset: Sequence[int],
        amplitude: complex,
    ) -> "ModelBuilder":
        call(
            self._native.add_hopping,
            int(target),
            int(source),
            [int(value) for value in cell_offset],
            complex(amplitude),
        )
        return self

    def add_hopping_block(
        self,
        target: int,
        source: int,
        cell_offset: Sequence[int],
        amplitude: npt.ArrayLike,
    ) -> "ModelBuilder":
        matrix = complex_matrix(amplitude, name="amplitude")
        call(
            self._native.add_hopping_block,
            int(target),
            int(source),
            [int(value) for value in cell_offset],
            matrix.tolist(),
        )
        return self

    def build(self) -> "Model":
        return Model._from_native(call(self._native.build))


class Model:
    """Immutable Rust-owned tight-binding model."""

    def __init__(self, native: Any) -> None:
        if not isinstance(native, _core.NativeModel):
            raise TypeError("Model instances are created by ModelBuilder")
        self._native = native

    @classmethod
    def _from_native(cls, native: Any) -> "Model":
        return cls(native)

    @property
    def state_count(self) -> int:
        return int(self._native.state_count)

    @property
    def real_dimension(self) -> int:
        return int(self._native.real_dimension)

    @property
    def periodic_dimension(self) -> int:
        return int(self._native.periodic_dimension)

    @property
    def lattice(self) -> Lattice:
        return Lattice(self._native.primitive_vectors, self._native.periodic_axes)

    def hamiltonian(self, momentum: npt.ArrayLike = ()) -> np.ndarray:
        point = real_vector(momentum, name="momentum")
        return np.asarray(
            call(self._native.hamiltonian, point.tolist()),
            dtype=np.complex128,
        )

    def eigensystem(self, momentum: npt.ArrayLike = ()) -> Any:
        from .spectrum import Eigensystem

        point = real_vector(momentum, name="momentum")
        values, vectors = call(self._native.eigensystem, point.tolist())
        return Eigensystem(
            np.asarray(values, dtype=np.float64),
            np.asarray(vectors, dtype=np.complex128),
        )

    def band_structure(self, momenta: npt.ArrayLike) -> list[Any]:
        from .spectrum import Eigensystem

        points = real_matrix(momenta, name="momenta")
        return [
            Eigensystem(
                np.asarray(values, dtype=np.float64),
                np.asarray(vectors, dtype=np.complex128),
            )
            for values, vectors in call(
                self._native.band_structure,
                points.tolist(),
            )
        ]

    def momentum_derivatives(
        self,
        momentum: npt.ArrayLike,
        *,
        cartesian: bool = False,
    ) -> np.ndarray:
        point = real_vector(momentum, name="momentum")
        return np.asarray(
            call(
                self._native.momentum_derivatives,
                point.tolist(),
                bool(cartesian),
            ),
            dtype=np.complex128,
        )

    def _export(self) -> tuple[Any, ...]:
        return call(self._native.export)


__all__ = ["Lattice", "Model", "ModelBuilder"]
