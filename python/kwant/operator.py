"""Local densities, currents, and sources from the continuity equation."""

from __future__ import annotations

import copy
import inspect

import numpy as np
from thouless import _core

from .builder import _block, _evaluate


def _validate_call_arguments(args, params):
    if args and params is not None:
        raise TypeError("'args' and 'params' are mutually exclusive")


class _LocalOperator:
    _where_rank = 1
    onsite = None

    def __init__(
        self,
        syst,
        onsite=1,
        where=None,
        sum=False,
        check_hermiticity=True,
    ):
        self.syst = syst
        self.sum = bool(sum)
        self.check_hermiticity = bool(check_hermiticity)
        self._bound_args = ()
        self._bound_params = None
        self._onsite_input = onsite

        cell_size = getattr(syst, "cell_size", len(syst.sites))
        cell_sites = syst.sites[:cell_size]
        if any(site.family.norbs is None for site in cell_sites):
            raise ValueError("Local operators require defined orbital counts")
        self._cell_size = cell_size
        self.where = self._normalize_where(where)
        self.onsite = self._public_onsite(onsite)

        if not callable(onsite) and not isinstance(onsite, dict):
            for index in self._onsite_sites():
                self._onsite_matrix(index, (), None)

    def _default_where(self):
        if self._where_rank == 1:
            return np.arange(self._cell_size, dtype=int).reshape(-1, 1)
        return np.asarray(
            [
                (first, second)
                for first, second in self.syst.graph
                if first < self._cell_size and second < self._cell_size
            ],
            dtype=int,
        ).reshape(-1, 2)

    def _normalize_site(self, value):
        if isinstance(value, (int, np.integer)):
            index = int(value)
        else:
            try:
                index = self.syst.id_by_site[value]
            except (KeyError, TypeError) as error:
                raise ValueError(f"Unknown operator site {value!r}") from error
        if not 0 <= index < self._cell_size:
            raise ValueError(f"Operator site index {index} is out of range")
        return index

    def _normalize_where(self, where):
        if where is None:
            return self._default_where()
        if callable(where):
            if self._where_rank == 1:
                values = [
                    index
                    for index, site in enumerate(self.syst.sites[: self._cell_size])
                    if where(site)
                ]
            else:
                values = [
                    (first, second)
                    for first, second in self._default_where()
                    if where(self.syst.sites[first], self.syst.sites[second])
                ]
        else:
            values = list(where)
        if self._where_rank == 1:
            return np.asarray(
                [self._normalize_site(value) for value in values],
                dtype=int,
            ).reshape(-1, 1)
        edges = []
        for value in values:
            if len(value) != 2:
                raise ValueError("Current where entries must be hoppings")
            first = self._normalize_site(value[0])
            second = self._normalize_site(value[1])
            if not self.syst.graph.has_edge(first, second):
                raise ValueError(
                    f"Sites {first} and {second} do not form a system hopping"
                )
            edges.append((first, second))
        return np.asarray(edges, dtype=int).reshape(-1, 2)

    def _onsite_sites(self):
        if self._where_rank == 1:
            return np.asarray(self.where).reshape(-1)
        return np.asarray(self.where)[:, 0]

    def _public_onsite(self, onsite):
        if callable(onsite) or isinstance(onsite, dict):
            return self._onsite_public
        first = next(iter(self._onsite_sites()), 0)
        dofs = self.syst.sites[int(first)].family.norbs
        return _block(onsite, dofs, dofs)

    def _onsite_public(self, index, *args, **kwargs):
        params = kwargs.pop("params", None)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {tuple(kwargs)}")
        return self._onsite_value(int(index), args, params)

    def _onsite_value(self, index, args, params):
        site = self.syst.sites[index]
        value = self._onsite_input
        if isinstance(value, dict):
            try:
                value = value[site.family]
            except KeyError as error:
                raise ValueError(
                    f"No onsite operator specified for family {site.family!r}"
                ) from error
        if callable(value):
            value = _evaluate(value, (site,), args, params)
        return value

    def _onsite_matrix(self, index, args, params):
        site = self.syst.sites[index]
        value = self._onsite_value(index, args, params)
        matrix = _block(value, site.family.norbs, site.family.norbs)
        if self.check_hermiticity and not np.allclose(matrix, matrix.conj().T):
            raise ValueError("Onsite operator is not Hermitian")
        return matrix

    def _arguments(self, args, params):
        _validate_call_arguments(args, params)
        if self._bound_args or self._bound_params is not None:
            if args or params is not None:
                raise ValueError("Arguments were already bound to this operator")
            return self._bound_args, self._bound_params
        return args, params

    def _offsets(self):
        offsets = [0]
        for site in self.syst.sites[: self._cell_size]:
            offsets.append(offsets[-1] + site.family.norbs)
        return offsets

    def _site_dimensions(self):
        return [
            int(site.family.norbs)
            for site in self.syst.sites[: self._cell_size]
        ]

    @staticmethod
    def _native_matrices(operator):
        return (
            np.asarray(operator.total_matrix(), dtype=complex),
            [
                np.asarray(component, dtype=complex)
                for component in operator.component_matrices()
            ],
        )

    def _native_operator(self, args, params):
        raise NotImplementedError

    def _matrix(self, args, params):
        return self._native_matrices(self._native_operator(args, params))

    def __call__(self, bra, ket=None, args=(), *, params=None):
        args, params = self._arguments(args, params)
        operator = self._native_operator(args, params)
        bra = np.asarray(bra, dtype=complex)
        if bra.ndim != 1 or bra.shape[0] != operator.dimension:
            raise ValueError("Wave function has incompatible size")
        diagonal = ket is None
        ket = bra if ket is None else np.asarray(ket, dtype=complex)
        if ket.ndim != 1 or ket.shape[0] != operator.dimension:
            raise ValueError("Wave function has incompatible size")
        values = np.asarray(
            operator.matrix_elements(bra.tolist(), ket.tolist()),
            dtype=complex,
        )
        if self.sum:
            values = values.sum()
        if diagonal and self.check_hermiticity:
            values = np.real(values)
        return values

    def act(self, ket, args=(), *, params=None):
        args, params = self._arguments(args, params)
        operator = self._native_operator(args, params)
        ket = np.asarray(ket, dtype=complex)
        if ket.ndim != 1 or ket.shape[0] != operator.dimension:
            raise ValueError("Wave function has incompatible size")
        return np.asarray(operator.apply_total(ket.tolist()), dtype=complex)

    def bind(self, args=(), *, params=None):
        _validate_call_arguments(args, params)
        result = copy.copy(self)
        result._bound_args = tuple(args)
        result._bound_params = None if params is None else dict(params)
        if callable(self._onsite_input):
            signature = inspect.signature(self._onsite_input)
            parameter_names = list(signature.parameters)[1:]
            if params is not None:
                missing = [name for name in parameter_names if name not in params]
                if missing:
                    raise TypeError(f"Missing required arguments: {missing}")
        return result

    def tocoo(self, args=(), *, params=None):
        args, params = self._arguments(args, params)
        operator = self._native_operator(args, params)
        from scipy.sparse import coo_matrix

        return coo_matrix(np.asarray(operator.total_matrix(), dtype=complex))


class Density(_LocalOperator):
    """Site-resolved matrix elements of a local observable."""

    def _native_operator(self, args, params):
        densities = [
            (
                int(index),
                self._onsite_matrix(index, args, params).tolist(),
            )
            for (index,) in self.where
        ]
        return _core.local_density_operators(
            self._site_dimensions(),
            densities,
        )


class Current(_LocalOperator):
    """Bond currents generated by a local density through system hoppings."""

    _where_rank = 2

    def _native_operator(self, args, params):
        offsets = self._offsets()
        currents = []
        for first, second in self.where:
            first_slice = slice(offsets[first], offsets[first + 1])
            second_slice = slice(offsets[second], offsets[second + 1])
            density = self._onsite_matrix(first, args, params)
            hopping = _block(
                self.syst.hamiltonian(first, second, *args, params=params),
                first_slice.stop - first_slice.start,
                second_slice.stop - second_slice.start,
            )
            currents.append(
                (
                    int(first),
                    int(second),
                    density.tolist(),
                    hopping.tolist(),
                )
            )
        return _core.bond_current_operators(
            self._site_dimensions(),
            currents,
        )


class Source(_LocalOperator):
    """Onsite production rate of a local density."""

    def _native_operator(self, args, params):
        offsets = self._offsets()
        sources = []
        for (index,) in self.where:
            site_slice = slice(offsets[index], offsets[index + 1])
            density = self._onsite_matrix(index, args, params)
            onsite = _block(
                self.syst.hamiltonian(index, index, *args, params=params),
                site_slice.stop - site_slice.start,
                site_slice.stop - site_slice.start,
                onsite=True,
            )
            sources.append(
                (
                    int(index),
                    density.tolist(),
                    onsite.tolist(),
                )
            )
        return _core.local_source_operators(
            self._site_dimensions(),
            sources,
        )


__all__ = ["Current", "Density", "Source"]
