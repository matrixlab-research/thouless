"""Finite transformations, reciprocal sampling, and lattice geometry."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call, complex_matrix, real_matrix
from .model import Lattice, Model


@dataclass(frozen=True)
class ReciprocalPath:
    """Reduced path points and Cartesian reciprocal distances.

    Attributes:
        points: Sampled reduced reciprocal coordinates.
        distances: Cumulative Cartesian reciprocal distance at every sample.
        node_distances: Cumulative distance of each requested path node.
    """

    points: np.ndarray
    distances: np.ndarray
    node_distances: np.ndarray


@dataclass(frozen=True)
class Supercell:
    """A transformed model and old-cell representatives.

    Attributes:
        model: Immutable model in the requested integer supercell basis.
        translations: Old-cell translation represented by every new orbital.
    """

    model: Model
    translations: np.ndarray


def reciprocal_path(
    lattice: Lattice,
    nodes: npt.ArrayLike,
    sample_count: int,
) -> ReciprocalPath:
    """Sample a piecewise-linear path in reduced reciprocal coordinates.

    Args:
        lattice: Real-space primitive-vector frame and periodic axes.
        nodes: Path vertices with shape ``(node_count, periodic_dimension)``.
        sample_count: Total number of path samples, including both endpoints.

    Returns:
        Reduced path points, cumulative Cartesian reciprocal distances, and
        the distance assigned to each requested node.

    Raises:
        ValueError: If ``nodes`` is not a two-dimensional array.
        ThoulessError: If dimensions or sample count are inconsistent.
    """
    requested = real_matrix(nodes, name="nodes")
    points, distances, node_distances = call(
        _core.reciprocal_path,
        lattice.primitive_vectors.tolist(),
        list(lattice.periodic_axes),
        requested.tolist(),
        int(sample_count),
    )
    return ReciprocalPath(
        np.asarray(points, dtype=np.float64),
        np.asarray(distances, dtype=np.float64),
        np.asarray(node_distances, dtype=np.float64),
    )


def finite_cluster(model: Model, cells: npt.ArrayLike) -> Model:
    """Extract complete source cells into an open finite model.

    Args:
        model: Source model; periodic boundary conditions are removed.
        cells: Integer source-cell coordinates with shape
            ``(cell_count, periodic_dimension)``.

    Returns:
        A finite model containing every source orbital in each selected cell.
    """
    cell_array = np.asarray(cells, dtype=np.int32)
    if cell_array.ndim != 2:
        raise ValueError("cells must have shape (site_count, dimension)")
    return Model._from_native(
        call(model._native.finite_cluster, cell_array.tolist()),
    )


def finite_geometry(
    model: Model,
    sites: Sequence[tuple[Sequence[int], int]],
) -> Model:
    """Extract arbitrary source sites into an open finite model.

    Args:
        model: Source model.
        sites: Ordered ``(cell, orbital)`` pairs using zero-based source
            orbital indices.

    Returns:
        A finite model that preserves the requested order and supports
        incomplete cells and vacancies.
    """
    converted = [
        ([int(component) for component in cell], int(orbital))
        for cell, orbital in sites
    ]
    return Model._from_native(call(model._native.finite_geometry, converted))


def remove_orbitals(model: Model, removed: Sequence[int]) -> Model:
    """Return a model without the selected source orbitals.

    Hoppings touching a removed orbital are discarded and surviving orbital
    indices are compacted while preserving their relative order.

    Args:
        model: Immutable source model.
        removed: Zero-based orbital indices to delete.

    Returns:
        A new Rust-owned model; ``model`` is unchanged.
    """
    return Model._from_native(
        call(model._native.remove_orbitals, [int(value) for value in removed]),
    )


def supercell(
    model: Model,
    integer_basis: npt.ArrayLike,
    *,
    move_periodic_to_home: bool = True,
) -> Supercell:
    """Transform a periodic model to an integer supercell basis.

    Args:
        model: Source periodic model.
        integer_basis: Integer rows expressing new periodic vectors in the old
            cell basis.
        move_periodic_to_home: Whether to choose representatives in the new
            home cell for periodic directions.

    Returns:
        The transformed model and the old-cell translation assigned to each
        new orbital.

    Raises:
        ValueError: If ``integer_basis`` is not two-dimensional.
        ThoulessError: If the basis is singular or dimensionally incompatible.
    """
    basis = np.asarray(integer_basis, dtype=np.int32)
    if basis.ndim != 2:
        raise ValueError("integer_basis must be two-dimensional")
    native, translations = call(
        model._native.supercell,
        basis.tolist(),
        bool(move_periodic_to_home),
    )
    return Supercell(
        Model._from_native(native),
        np.asarray(translations, dtype=np.int32),
    )


def fold_terms(
    terms: Sequence[tuple[npt.ArrayLike, Sequence[int], bool]],
    momentum: npt.ArrayLike,
) -> np.ndarray:
    """Fold translated matrix terms into a Bloch Hamiltonian.

    Args:
        terms: Sequence of ``(matrix, translation, include_adjoint)`` records.
            Each term is multiplied by its Bloch phase; when
            ``include_adjoint`` is true its Hermitian-conjugate counterpart is
            added as well.
        momentum: Reduced reciprocal coordinate used for all phases.

    Returns:
        Complex matrix equal to the sum of all phased terms.
    """
    converted = [
        (
            complex_matrix(value, name="term").tolist(),
            [int(component) for component in translation],
            bool(include_adjoint),
        )
        for value, translation, include_adjoint in terms
    ]
    point = np.asarray(momentum, dtype=np.float64)
    return np.asarray(
        call(_core.periodic_fold_terms, converted, point.tolist()),
        dtype=np.complex128,
    )


def lll_reduce(
    basis: npt.ArrayLike,
    *,
    reduction_parameter: float = 1.34,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a row-vector lattice basis with the LLL algorithm.

    Args:
        basis: Real basis vectors stored as rows.
        reduction_parameter: Lovász reduction parameter. It must exceed
            ``4/3`` in the convention used by the native implementation.

    Returns:
        ``(reduced_basis, transformation)`` where the exact integer
        transformation maps original rows to reduced rows.

    Raises:
        ThoulessError: If the basis is empty, dependent, non-finite, or the
            reduction parameter is invalid.
    """
    vectors = real_matrix(basis, name="basis")
    reduced, transformation = call(
        _core.lattice_lll,
        vectors.tolist(),
        float(reduction_parameter),
    )
    return (
        np.asarray(reduced, dtype=np.float64),
        np.asarray(transformation, dtype=np.int64),
    )


def closest_lattice_vectors(
    target: npt.ArrayLike,
    basis: npt.ArrayLike,
    *,
    neighbor_count: int = 1,
    group_by_length: bool = False,
    relative_tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Find integer coefficients of lattice vectors closest to a target.

    Args:
        target: Cartesian target vector.
        basis: Real lattice basis vectors stored as rows.
        neighbor_count: Minimum number of nearest coefficient vectors.
        group_by_length: If true, include complete distance-degenerate shells
            even when this returns more than ``neighbor_count`` rows.
        relative_tolerance: Relative tolerance used to group equal distances.

    Returns:
        Integer coefficient vectors ordered by increasing target distance.
    """
    return np.asarray(
        call(
            _core.lattice_cvp,
            np.asarray(target, dtype=np.float64).tolist(),
            real_matrix(basis, name="basis").tolist(),
            int(neighbor_count),
            bool(group_by_length),
            float(relative_tolerance),
        ),
        dtype=np.int64,
    )


def voronoi_neighbors(
    basis: npt.ArrayLike,
    *,
    reduced: bool = False,
    relative_tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Enumerate coefficient vectors whose lattice points share the origin cell.

    Args:
        basis: Real lattice basis vectors stored as rows.
        reduced: Set true only when ``basis`` is already LLL-reduced.
        relative_tolerance: Relative tolerance for Voronoi-boundary tests.

    Returns:
        Integer lattice coefficients of the Voronoi-neighbor vectors.
    """
    return np.asarray(
        call(
            _core.lattice_voronoi,
            real_matrix(basis, name="basis").tolist(),
            bool(reduced),
            float(relative_tolerance),
        ),
        dtype=np.int64,
    )


def embedded_neighbors(
    primitive_vectors: npt.ArrayLike,
    basis_offsets: npt.ArrayLike,
    order: int,
    *,
    relative_tolerance: float = 1.0e-9,
) -> list[tuple[np.ndarray, int, int]]:
    """Enumerate one distance shell of an embedded lattice.

    Args:
        primitive_vectors: Primitive real-space vectors stored as rows.
        basis_offsets: Cartesian positions of sites within the primitive cell.
        order: One-based neighbor-shell number.
        relative_tolerance: Relative tolerance for grouping equal distances.

    Returns:
        ``(cell_displacement, first_site, second_site)`` relations in
        deterministic order. Site indices are zero-based.
    """
    relations = call(
        _core.embedded_lattice_neighbors,
        real_matrix(primitive_vectors, name="primitive_vectors").tolist(),
        real_matrix(basis_offsets, name="basis_offsets").tolist(),
        int(order),
        float(relative_tolerance),
    )
    return [
        (np.asarray(displacement, dtype=np.int64), int(first), int(second))
        for displacement, first, second in relations
    ]


__all__ = [
    "ReciprocalPath",
    "Supercell",
    "closest_lattice_vectors",
    "embedded_neighbors",
    "finite_cluster",
    "finite_geometry",
    "fold_terms",
    "lll_reduce",
    "reciprocal_path",
    "remove_orbitals",
    "supercell",
    "voronoi_neighbors",
]
