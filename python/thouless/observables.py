"""Local density, current, source, and operator evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call, complex_matrix, complex_vector


class LocalOperatorSet:
    """Rust-owned sparse local operator components."""

    def __init__(self, native: object) -> None:
        if not isinstance(native, _core.LocalOperatorSet):
            raise TypeError("LocalOperatorSet objects are created by module functions")
        self._native = native

    @property
    def dimension(self) -> int:
        return int(self._native.dimension)

    def matrix_elements(
        self,
        bra: npt.ArrayLike,
        ket: npt.ArrayLike,
    ) -> np.ndarray:
        return np.asarray(
            call(
                self._native.matrix_elements,
                complex_vector(bra, name="bra").tolist(),
                complex_vector(ket, name="ket").tolist(),
            ),
            dtype=np.complex128,
        )

    def apply(self, ket: npt.ArrayLike) -> np.ndarray:
        return np.asarray(
            call(
                self._native.apply_total,
                complex_vector(ket, name="ket").tolist(),
            ),
            dtype=np.complex128,
        )

    def matrix(self) -> np.ndarray:
        return np.asarray(call(self._native.total_matrix), dtype=np.complex128)

    def component_matrices(self) -> np.ndarray:
        return np.asarray(
            call(self._native.component_matrices),
            dtype=np.complex128,
        )


def densities(
    site_dimensions: Sequence[int],
    terms: Sequence[tuple[int, npt.ArrayLike]],
) -> LocalOperatorSet:
    converted = [
        (int(site), complex_matrix(observable, name="observable").tolist())
        for site, observable in terms
    ]
    return LocalOperatorSet(
        call(
            _core.local_density_operators,
            [int(value) for value in site_dimensions],
            converted,
        )
    )


def currents(
    site_dimensions: Sequence[int],
    terms: Sequence[
        tuple[int, int, npt.ArrayLike, npt.ArrayLike]
    ],
) -> LocalOperatorSet:
    converted = [
        (
            int(site),
            int(neighbor),
            complex_matrix(observable, name="observable").tolist(),
            complex_matrix(hopping, name="hopping").tolist(),
        )
        for site, neighbor, observable, hopping in terms
    ]
    return LocalOperatorSet(
        call(
            _core.bond_current_operators,
            [int(value) for value in site_dimensions],
            converted,
        )
    )


def sources(
    site_dimensions: Sequence[int],
    terms: Sequence[tuple[int, npt.ArrayLike, npt.ArrayLike]],
) -> LocalOperatorSet:
    converted = [
        (
            int(site),
            complex_matrix(observable, name="observable").tolist(),
            complex_matrix(onsite, name="onsite").tolist(),
        )
        for site, observable, onsite in terms
    ]
    return LocalOperatorSet(
        call(
            _core.local_source_operators,
            [int(value) for value in site_dimensions],
            converted,
        )
    )


def project_diagonal(
    states: npt.ArrayLike,
    diagonal: npt.ArrayLike,
) -> np.ndarray:
    return np.asarray(
        call(
            _core.diagonal_observable_matrix,
            complex_matrix(states, name="states").tolist(),
            np.asarray(diagonal, dtype=np.float64).tolist(),
        ),
        dtype=np.complex128,
    )


def pauli_coefficients(matrix: npt.ArrayLike) -> np.ndarray:
    return np.asarray(
        call(
            _core.pauli_decompose,
            complex_matrix(matrix, name="matrix").tolist(),
        ),
        dtype=np.complex128,
    )


__all__ = [
    "LocalOperatorSet",
    "currents",
    "densities",
    "pauli_coefficients",
    "project_diagonal",
    "sources",
]
