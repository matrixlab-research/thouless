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
    """Reduced path points and Cartesian reciprocal distances."""

    points: np.ndarray
    distances: np.ndarray
    node_distances: np.ndarray


@dataclass(frozen=True)
class Supercell:
    """A transformed model and old-cell representatives."""

    model: Model
    translations: np.ndarray


def reciprocal_path(
    lattice: Lattice,
    nodes: npt.ArrayLike,
    sample_count: int,
) -> ReciprocalPath:
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
    """Extract complete source cells into an open finite model."""
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
    """Extract an arbitrary set of source-cell and source-orbital pairs."""
    converted = [
        ([int(component) for component in cell], int(orbital))
        for cell, orbital in sites
    ]
    return Model._from_native(call(model._native.finite_geometry, converted))


def remove_orbitals(model: Model, removed: Sequence[int]) -> Model:
    return Model._from_native(
        call(model._native.remove_orbitals, [int(value) for value in removed]),
    )


def supercell(
    model: Model,
    integer_basis: npt.ArrayLike,
    *,
    move_periodic_to_home: bool = True,
) -> Supercell:
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
