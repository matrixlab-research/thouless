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
    """A Cartesian primitive-vector frame and the axes treated as periodic.

    Primitive vectors are Cartesian rows. Orbital positions use reduced
    coordinates in this frame, while ``periodic_axes`` selects which rows
    correspond to reduced momentum components.
    """

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
        """Number of Cartesian components in the primitive-vector frame."""
        return int(self.primitive_vectors.shape[0])

    @property
    def periodic_dimension(self) -> int:
        """Number of reciprocal coordinates accepted by the model."""
        return len(self.periodic_axes)


class ModelBuilder:
    """Exclusive mutable construction of an immutable Rust model.

    A builder owns its native state and is consumed by :meth:`build`. Hopping
    calls automatically add the Hermitian-conjugate physical counterpart.
    """

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
        """Add one orbital block to the primitive cell.

        Args:
            label: Stable human-readable orbital label.
            reduced_position: Coordinates in the primitive-vector frame.
            degrees_of_freedom: Positive number of internal states carried by
                the orbital.

        Returns:
            Zero-based orbital index for subsequent onsite and hopping calls.
        """
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
        """Set a scalar onsite energy on an orbital block.

        The scalar multiplies the identity in the orbital's internal space.
        Returns this builder to support explicit method chaining.
        """
        call(self._native.set_onsite, int(orbital), float(energy))
        return self

    def set_onsite_block(
        self,
        orbital: int,
        block: npt.ArrayLike,
    ) -> "ModelBuilder":
        """Set the full Hermitian onsite block for one orbital.

        ``block`` must match the orbital's declared degrees of freedom.
        Returns this builder.
        """
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
        """Add a scalar hopping and its Hermitian-conjugate counterpart.

        Args:
            target: Zero-based target orbital.
            source: Zero-based source orbital.
            cell_offset: Integer target-cell displacement in periodic-axis
                order.
            amplitude: Scalar source-to-target matrix element.

        Returns:
            This builder.
        """
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
        """Add a block hopping and its Hermitian-conjugate counterpart.

        The block shape is ``(target_dof, source_dof)`` and ``cell_offset`` is
        expressed in periodic-axis order. Returns this builder.
        """
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
        """Consume the builder and return an immutable Rust-owned model.

        A builder is single-use; any subsequent mutation or build attempt
        raises :class:`~thouless.errors.ThoulessError`.
        """
        return Model._from_native(call(self._native.build))


class Model:
    """Immutable Rust-owned tight-binding model.

    Models are created by :class:`ModelBuilder` or geometry transformations.
    Hamiltonian assembly, eigensystems, derivatives, and downstream scientific
    workflows all execute in Rust.
    """

    def __init__(self, native: Any) -> None:
        if not isinstance(native, _core.NativeModel):
            raise TypeError("Model instances are created by ModelBuilder")
        self._native = native

    @classmethod
    def _from_native(cls, native: Any) -> "Model":
        return cls(native)

    @property
    def state_count(self) -> int:
        """Total Hilbert-space dimension of one model cell."""
        return int(self._native.state_count)

    @property
    def real_dimension(self) -> int:
        """Number of Cartesian dimensions in the lattice frame."""
        return int(self._native.real_dimension)

    @property
    def periodic_dimension(self) -> int:
        """Number of reduced momentum coordinates expected by this model."""
        return int(self._native.periodic_dimension)

    @property
    def lattice(self) -> Lattice:
        """Copy the model's lattice metadata into a Python value object."""
        return Lattice(self._native.primitive_vectors, self._native.periodic_axes)

    def hamiltonian(self, momentum: npt.ArrayLike = ()) -> np.ndarray:
        """Evaluate the Bloch Hamiltonian at reduced momentum ``momentum``.

        Finite models accept the default empty momentum. Periodic models
        require exactly :attr:`periodic_dimension` components.
        """
        point = real_vector(momentum, name="momentum")
        return np.asarray(
            call(self._native.hamiltonian, point.tolist()),
            dtype=np.complex128,
        )

    def eigensystem(self, momentum: npt.ArrayLike = ()) -> Any:
        """Diagonalize the Hermitian Bloch Hamiltonian.

        Returns:
            An :class:`thouless.spectrum.Eigensystem` with ascending energies
            and normalized eigenvectors stored as columns.
        """
        from .spectrum import Eigensystem

        point = real_vector(momentum, name="momentum")
        values, vectors = call(self._native.eigensystem, point.tolist())
        return Eigensystem(
            np.asarray(values, dtype=np.float64),
            np.asarray(vectors, dtype=np.complex128),
        )

    def band_structure(self, momenta: npt.ArrayLike) -> list[Any]:
        """Diagonalize the model independently along a momentum path.

        Args:
            momenta: Array with shape ``(sample_count, periodic_dimension)``.

        Returns:
            One ascending eigensystem per momentum sample.
        """
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
        """Differentiate the Bloch Hamiltonian analytically with momentum.

        Args:
            momentum: Reduced reciprocal coordinate.
            cartesian: If true, transform derivatives to Cartesian reciprocal
                components; otherwise retain reduced-coordinate derivatives.

        Returns:
            Complex array whose leading axis selects the derivative direction.
        """
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
