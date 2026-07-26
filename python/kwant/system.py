"""Low-level Kwant system protocols and reusable default operations."""

from __future__ import annotations

import abc
import copy

import numpy as np


def _block(value, rows, columns):
    array = np.asarray(value, dtype=complex)
    if array.ndim == 0:
        if rows != columns:
            raise ValueError("A scalar Hamiltonian block must be square")
        return np.eye(rows, dtype=complex) * array
    if array.shape != (rows, columns):
        raise ValueError(
            f"Hamiltonian block has shape {array.shape}, "
            f"expected {(rows, columns)}"
        )
    return array


def _orbital_counts(system, args, params):
    ranges = getattr(system, "site_ranges", None)
    if ranges:
        counts = np.empty(system.graph.num_nodes, dtype=int)
        for (start, norbs, _), (stop, _, _) in zip(
            ranges[:-1],
            ranges[1:],
            strict=True,
        ):
            counts[int(start) : int(stop)] = int(norbs)
    else:
        counts = np.ones(system.graph.num_nodes, dtype=int)
    for index in range(system.graph.num_nodes):
        onsite = np.asarray(
            system.hamiltonian(index, index, *args, params=params)
        )
        if onsite.ndim == 0:
            continue
        if onsite.ndim != 2 or onsite.shape[0] != onsite.shape[1]:
            raise ValueError(
                f"Onsite Hamiltonian block has shape {onsite.shape}, "
                "expected a square matrix"
            )
        counts[index] = onsite.shape[0]
    return counts


class System(metaclass=abc.ABCMeta):
    """Abstract low-level graph Hamiltonian."""

    @abc.abstractmethod
    def hamiltonian(self, i, j, *args, params=None):
        """Return an onsite or hopping block."""

    def discrete_symmetry(self, args=(), *, params=None):
        del args, params
        from .physics import DiscreteSymmetry

        return DiscreteSymmetry()

    def hamiltonian_submatrix(
        self,
        args=(),
        to_sites=None,
        from_sites=None,
        sparse=False,
        return_norb=False,
        *,
        params=None,
    ):
        """Assemble an orbital Hamiltonian from the low-level graph protocol."""
        if args and params is not None:
            raise TypeError("'args' and 'params' are mutually exclusive")
        args = tuple(args)
        node_count = int(self.graph.num_nodes)
        rows = list(range(node_count)) if to_sites is None else list(to_sites)
        columns = (
            list(range(node_count))
            if from_sites is None
            else list(from_sites)
        )
        counts = _orbital_counts(self, args, params)
        row_offsets = np.cumsum([0, *(int(counts[index]) for index in rows)])
        column_offsets = np.cumsum(
            [0, *(int(counts[index]) for index in columns)]
        )
        if sparse:
            from scipy import sparse as scipy_sparse

            row_positions = {}
            for position, site in enumerate(rows):
                row_positions.setdefault(int(site), []).append(position)
            column_positions = {}
            for position, site in enumerate(columns):
                column_positions.setdefault(int(site), []).append(position)
            data = []
            row_indices = []
            column_indices = []

            def append_block(row_site, column_site, value):
                block = _block(
                    value,
                    int(counts[row_site]),
                    int(counts[column_site]),
                )
                for row_position in row_positions.get(row_site, ()):
                    row_start = int(row_offsets[row_position])
                    for column_position in column_positions.get(
                        column_site, ()
                    ):
                        column_start = int(
                            column_offsets[column_position]
                        )
                        for local_row, local_column in zip(
                            *np.nonzero(block), strict=True
                        ):
                            data.append(
                                block[local_row, local_column]
                            )
                            row_indices.append(
                                row_start + int(local_row)
                            )
                            column_indices.append(
                                column_start + int(local_column)
                            )

            for site in row_positions.keys() & column_positions.keys():
                append_block(
                    site,
                    site,
                    self.hamiltonian(
                        site, site, *args, params=params
                    ),
                )

            try:
                graph_edges = iter(self.graph)
            except TypeError:
                graph_edges = (
                    (row_site, column_site)
                    for row_site in row_positions
                    for column_site in column_positions
                    if row_site != column_site
                    and self.graph.has_edge(row_site, column_site)
                )
            seen = set()
            for row_site, column_site in graph_edges:
                row_site = int(row_site)
                column_site = int(column_site)
                edge = row_site, column_site
                if (
                    edge in seen
                    or row_site not in row_positions
                    or column_site not in column_positions
                ):
                    continue
                seen.add(edge)
                try:
                    value = self.hamiltonian(
                        row_site,
                        column_site,
                        *args,
                        params=params,
                    )
                except KeyError:
                    continue
                append_block(row_site, column_site, value)

            output = scipy_sparse.coo_matrix(
                (data, (row_indices, column_indices)),
                shape=(
                    int(row_offsets[-1]),
                    int(column_offsets[-1]),
                ),
                dtype=complex,
            )
            if return_norb:
                return (
                    output,
                    counts[np.asarray(rows, dtype=int)],
                    counts[np.asarray(columns, dtype=int)],
                )
            return output

        matrix = np.zeros(
            (int(row_offsets[-1]), int(column_offsets[-1])),
            dtype=complex,
        )
        for row_position, row_site in enumerate(rows):
            for column_position, column_site in enumerate(columns):
                if row_site != column_site:
                    try:
                        if not self.graph.has_edge(row_site, column_site):
                            continue
                    except AttributeError:
                        pass
                try:
                    value = self.hamiltonian(
                        row_site,
                        column_site,
                        *args,
                        params=params,
                    )
                except KeyError:
                    continue
                block = _block(
                    value,
                    int(counts[row_site]),
                    int(counts[column_site]),
                )
                matrix[
                    row_offsets[row_position] : row_offsets[row_position + 1],
                    column_offsets[
                        column_position
                    ] : column_offsets[column_position + 1],
                ] = block
        if return_norb:
            return (
                matrix,
                counts[np.asarray(rows, dtype=int)],
                counts[np.asarray(columns, dtype=int)],
            )
        return matrix

    def __str__(self):
        return (
            f"<{self.__class__.__name__} with "
            f"{self.graph.num_nodes} sites and "
            f"{self.graph.num_edges // 2} hoppings>"
        )


class FiniteSystem(System, metaclass=abc.ABCMeta):
    """Abstract finite low-level system, optionally connected to leads."""

    def precalculate(
        self,
        energy=0,
        args=(),
        leads=None,
        what="modes",
        *,
        params=None,
    ):
        if what not in {"modes", "selfenergy", "all"}:
            raise ValueError(f"Invalid value of argument 'what': {what}")
        selected = (
            set(range(len(self.leads)))
            if leads is None
            else {int(lead) for lead in leads}
        )
        result = copy.copy(self)
        result.leads = list(self.leads)
        for index in selected:
            lead = self.leads[index]
            modes = (
                lead.modes(energy, args=args, params=params)
                if what in {"modes", "all"}
                else None
            )
            selfenergy = None
            if what in {"selfenergy", "all"}:
                selfenergy = (
                    modes[1].selfenergy()
                    if modes is not None
                    else lead.selfenergy(energy, args=args, params=params)
                )
            result.leads[index] = PrecalculatedLead(modes, selfenergy)
        return result

    def validate_symmetries(self, args=(), *, params=None):
        symmetry = self.discrete_symmetry(args=args, params=params)
        hamiltonian = self.hamiltonian_submatrix(
            args=args,
            sparse=True,
            params=params,
        )
        return symmetry.validate(hamiltonian)


class InfiniteSystem(System, metaclass=abc.ABCMeta):
    """Abstract one-directional periodic low-level system."""

    def cell_hamiltonian(self, args=(), sparse=False, *, params=None):
        sites = range(self.cell_size)
        return self.hamiltonian_submatrix(
            args,
            sites,
            sites,
            sparse=sparse,
            params=params,
        )

    def inter_cell_hopping(self, args=(), sparse=False, *, params=None):
        cell = range(self.cell_size)
        interface = range(self.cell_size, self.graph.num_nodes)
        return self.hamiltonian_submatrix(
            args,
            cell,
            interface,
            sparse=sparse,
            params=params,
        )

    def modes(self, energy=0, args=(), *, params=None):
        from . import physics

        cell = np.asarray(
            self.cell_hamiltonian(args=args, params=params),
            dtype=complex,
        )
        hopping = self.inter_cell_hopping(args=args, params=params)
        cell.flat[:: cell.shape[0] + 1] -= float(energy)
        return physics.modes(cell, hopping)

    def selfenergy(self, energy=0, args=(), *, params=None):
        from . import physics

        cell = np.asarray(
            self.cell_hamiltonian(args=args, params=params),
            dtype=complex,
        )
        hopping = self.inter_cell_hopping(args=args, params=params)
        cell.flat[:: cell.shape[0] + 1] -= float(energy)
        return physics.selfenergy(cell, hopping)

    def validate_symmetries(self, args=(), *, params=None):
        symmetry = self.discrete_symmetry(args=args, params=params)
        broken = set(
            symmetry.validate(
                self.cell_hamiltonian(
                    args=args,
                    sparse=True,
                    params=params,
                )
            )
        )
        broken.update(
            symmetry.validate(
                self.inter_cell_hopping(
                    args=args,
                    sparse=True,
                    params=params,
                )
            )
        )
        return list(broken)


def is_selfenergy_lead(lead):
    """Return whether a lead supplies self-energy but no propagating modes."""
    return hasattr(lead, "selfenergy") and not hasattr(lead, "modes")


class PrecalculatedLead:
    """Lead with cached modes and/or self-energy."""

    def __init__(self, modes=None, selfenergy=None):
        if modes is None and selfenergy is None:
            raise ValueError("No precalculated values provided.")
        self._modes = modes
        self._selfenergy = selfenergy
        self.parameters = frozenset()

    def modes(self, energy=0, args=(), *, params=None):
        del energy, args, params
        if self._modes is None:
            raise ValueError("No precalculated modes were provided.")
        return self._modes

    def selfenergy(self, energy=0, args=(), *, params=None):
        del energy, args, params
        if self._selfenergy is None:
            raise ValueError("No precalculated selfenergy was provided.")
        return self._selfenergy


__all__ = ["FiniteSystem", "InfiniteSystem", "System"]
