"""Steady-state solver entry points backed by Thouless Rust transport."""

from __future__ import annotations

import sys
import types

import numpy as np

from thouless import _core


class BlockResult:
    """Common block access and transport statistics for solver results."""

    def __init__(
        self,
        data,
        lead_info,
        out_leads,
        in_leads,
        sizes,
        current_conserving=False,
    ):
        self.data = np.asarray(data, dtype=complex)
        self.lead_info = tuple(lead_info)
        self.out_leads = list(out_leads)
        self.in_leads = list(in_leads)
        self.sizes = np.asarray(sizes, dtype=int)
        self.current_conserving = bool(current_conserving)
        self.in_offsets = np.cumsum(
            [0, *(self.sizes[index] for index in self.in_leads)]
        )
        self.out_offsets = np.cumsum(
            [0, *(self.sizes[index] for index in self.out_leads)]
        )

    def block_coords(self, out_lead, in_lead):
        return (
            self.out_block_coords(out_lead),
            self.in_block_coords(in_lead),
        )

    def out_block_coords(self, out_lead):
        position = self.out_leads.index(int(out_lead))
        return slice(
            self.out_offsets[position],
            self.out_offsets[position + 1],
        )

    def in_block_coords(self, in_lead):
        position = self.in_leads.index(int(in_lead))
        return slice(
            self.in_offsets[position],
            self.in_offsets[position + 1],
        )

    def submatrix(self, out_lead, in_lead):
        return self.data[self.block_coords(out_lead, in_lead)]

    def num_propagating(self, lead):
        """Return the number of propagating channels in a lead."""
        raise NotImplementedError

    def transmission(self, out_lead, in_lead):
        chosen = [int(out_lead), int(in_lead)]
        present = self.out_leads, self.in_leads
        available = [
            int(lead in leads)
            for lead, leads in zip(chosen, present, strict=True)
        ]
        if all(available):
            return self._transmission(*chosen)

        all_but_one = len(self.lead_info) - 1
        if self.current_conserving:
            if sum(available) == 1:
                sum_axis, available_axis = available
                if len(present[sum_axis]) == all_but_one:
                    return self.num_propagating(
                        chosen[available_axis]
                    ) - sum(
                        self._transmission(*chosen)
                        for chosen[sum_axis] in present[sum_axis]
                    )
            elif all(
                len(leads) == all_but_one for leads in present
            ):
                return sum(
                    self._transmission(out_index, in_index)
                    - (
                        self.num_propagating(out_index)
                        if out_index == in_index
                        else 0
                    )
                    for out_index in present[0]
                    for in_index in present[1]
                )
        raise ValueError(
            "Insufficient matrix elements to compute "
            f"transmission({chosen[0]}, {chosen[1]})"
        )

    def conductance_matrix(self):
        lead_count = len(self.lead_info)
        matrix = np.asarray(
            [
                [
                    -self.transmission(drain, source)
                    if drain != source
                    else 0.0
                    for source in range(lead_count)
                ]
                for drain in range(lead_count)
            ],
            dtype=float,
        )
        matrix.flat[:: lead_count + 1] = -matrix.sum(axis=0)
        return matrix


class SMatrix(BlockResult):
    """Scattering amplitudes in the propagating-mode basis of each lead."""

    def __init__(
        self,
        data,
        transmissions,
        lead_slices,
        lead_info=(),
        out_leads=None,
        in_leads=None,
        current_conserving=True,
    ):
        self._transmissions = np.asarray(transmissions, dtype=float)
        lead_info = tuple(lead_info)
        lead_count = len(lead_info)
        out_leads = list(
            range(lead_count) if out_leads is None else out_leads
        )
        in_leads = list(
            range(lead_count) if in_leads is None else in_leads
        )
        sizes = [
            int(lead_slices[index].stop - lead_slices[index].start)
            for index in range(lead_count)
        ]
        super().__init__(
            data,
            lead_info,
            out_leads,
            in_leads,
            sizes,
            current_conserving,
        )

    def out_block_coords(self, out_lead):
        if np.isscalar(out_lead):
            return super().out_block_coords(out_lead)
        lead, block = map(int, out_lead)
        sizes = getattr(self.lead_info[lead], "block_nmodes", None)
        if sizes is None:
            raise IndexError(f"Lead {lead} has no conservation-law blocks")
        base = super().out_block_coords(lead).start
        offsets = np.cumsum([0, *sizes])
        return slice(base + offsets[block], base + offsets[block + 1])

    def in_block_coords(self, in_lead):
        if np.isscalar(in_lead):
            return super().in_block_coords(in_lead)
        lead, block = map(int, in_lead)
        sizes = getattr(self.lead_info[lead], "block_nmodes", None)
        if sizes is None:
            raise IndexError(f"Lead {lead} has no conservation-law blocks")
        base = super().in_block_coords(lead).start
        offsets = np.cumsum([0, *sizes])
        return slice(base + offsets[block], base + offsets[block + 1])

    def _transmission(self, out_lead, in_lead):
        if np.isscalar(out_lead) and np.isscalar(in_lead):
            return float(
                self._transmissions[int(out_lead), int(in_lead)]
            )
        return float(
            np.sum(np.abs(self.submatrix(out_lead, in_lead)) ** 2)
        )

    def transmission(self, out_lead, in_lead):
        if np.isscalar(out_lead) and np.isscalar(in_lead):
            return super().transmission(out_lead, in_lead)
        return self._transmission(out_lead, in_lead)

    def submatrix(self, out_lead, in_lead):
        return super().submatrix(out_lead, in_lead)

    def num_propagating(self, lead):
        info = self.lead_info[int(lead)]
        return len(getattr(info, "momenta", ())) // 2


class GreensFunction(BlockResult):
    """Retarded device Green function and lead-to-lead transmissions."""

    def __init__(
        self,
        data,
        transmissions,
        selfenergies,
        broadenings,
        channel_counts,
        out_leads=None,
        in_leads=None,
        current_conserving=True,
    ):
        self._transmissions = np.asarray(transmissions, dtype=float)
        self.selfenergies = tuple(
            np.asarray(value, dtype=complex) for value in selfenergies
        )
        self.broadenings = tuple(
            np.asarray(value, dtype=complex) for value in broadenings
        )
        self._channel_counts = tuple(channel_counts)
        lead_count = len(self.selfenergies)
        out_leads = list(
            range(lead_count) if out_leads is None else out_leads
        )
        in_leads = list(
            range(lead_count) if in_leads is None else in_leads
        )
        super().__init__(
            data,
            self.selfenergies,
            out_leads,
            in_leads,
            [matrix.shape[0] for matrix in self.selfenergies],
            current_conserving,
        )

    def _a_ttdagger_a_inv(self, lead_out, lead_in):
        green = self.submatrix(lead_out, lead_in)
        return (
            self.broadenings[int(lead_out)]
            @ green
            @ self.broadenings[int(lead_in)]
            @ green.conj().T
        )

    def _transmission(self, lead_out, lead_in):
        return float(
            self._transmissions[int(lead_out), int(lead_in)]
        )

    def num_propagating(self, lead):
        return int(self._channel_counts[int(lead)])


def _solution(syst, energy, args, params, channel_counts=None):
    device, leads = syst._transport_data(args=args, params=params)
    selfenergies = [
        np.asarray(value, dtype=complex)
        for value in _core.open_system_extrapolated_self_energies(
            device.tolist(),
            leads,
            float(energy),
        )
    ]
    counts = []
    for index, lead in enumerate(syst.leads):
        if hasattr(lead, "modes"):
            propagating, stabilized = lead.modes(
                energy, args=args, params=params
            )
            count = len(propagating.momenta) // 2
            counts.append(count)
            if getattr(lead, "_uses_stabilized_selfenergy", False):
                interface_selfenergy = np.asarray(
                    stabilized.selfenergy(),
                    dtype=complex,
                )
                selfenergies[index] = _embed_interface_matrix(
                    syst,
                    index,
                    interface_selfenergy,
                    args,
                    params,
                )
            selfenergies[index] = np.asarray(
                _core.regularize_embedded_self_energy(
                    selfenergies[index].tolist(),
                    count,
                ),
                dtype=complex,
            )
        elif hasattr(lead, "selfenergy"):
            counts.append(None)
            interface_selfenergy = np.asarray(
                lead.selfenergy(energy, args=args, params=params),
                dtype=complex,
            )
            selfenergies[index] = _embed_interface_matrix(
                syst, index, interface_selfenergy, args, params
            )
        else:
            raise ValueError(
                f"Lead {index} provides neither modes nor selfenergy"
            )
    if channel_counts is not None:
        expected = [
            None if count is None else int(count)
            for count in channel_counts
        ]
        actual = [
            None if count is None else int(count) for count in counts
        ]
        if any(
            expected_count not in (0, actual_count)
            for expected_count, actual_count in zip(
                expected, actual, strict=True
            )
        ):
            raise ValueError("Lead channel count is inconsistent with modes")
    native = _core.open_system_from_self_energies(
        device.tolist(),
        [value.tolist() for value in selfenergies],
        float(energy),
    )
    return native, tuple(counts)


def _embed_interface_matrix(syst, lead_index, matrix, args, params):
    offsets = syst._site_slices(args, params)
    interface = syst.lead_interfaces[lead_index]
    device_basis = np.concatenate(
        [
            np.arange(offsets[index], offsets[index + 1])
            for index in interface
        ]
    )
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (len(device_basis), len(device_basis)):
        raise ValueError(
            f"Self-energy dimension for lead {lead_index} does not "
            "match its interface"
        )
    result = np.zeros(
        (offsets[-1], offsets[-1]), dtype=complex
    )
    result[np.ix_(device_basis, device_basis)] = matrix
    return result


def _check_precalculated(syst, allowed):
    what = getattr(syst, "_precalculated_what", None)
    if what is not None and what not in allowed:
        raise ValueError(
            f"System precalculated with {what!r}, expected one of {tuple(allowed)!r}"
        )


def _mode_factors(
    syst, lead_info, selfenergies, native, args, params
):
    device, lead_data = syst._transport_data(args=args, params=params)
    offsets = syst._site_slices(args, params)
    incoming = []
    outgoing = []
    for index, (data, info, selfenergy, interface) in enumerate(
        zip(
            lead_data,
            lead_info,
            selfenergies,
            syst.lead_interfaces,
            strict=True,
        )
    ):
        if not hasattr(info, "momenta"):
            empty = np.empty((device.shape[0], 0), dtype=complex)
            incoming.append(empty)
            outgoing.append(empty)
            continue
        if getattr(
            syst.leads[index],
            "_uses_stabilized_selfenergy",
            False,
        ):
            mode_count = len(info.momenta) // 2
            factors = np.asarray(
                native.broadening_factor(index, mode_count),
                dtype=complex,
            )
            incoming.append(factors)
            outgoing.append(factors)
            continue
        coupling = np.asarray(data[2], dtype=complex)
        sigma = np.asarray(selfenergy, dtype=complex)
        device_basis = np.concatenate(
            [
                np.arange(offsets[index], offsets[index + 1])
                for index in interface
            ]
        )
        boundary = np.zeros(
            (device.shape[0], coupling.shape[1]), dtype=complex
        )
        boundary[
            np.ix_(device_basis, np.arange(len(device_basis)))
        ] = np.eye(len(device_basis))
        mode_count = len(info.momenta) // 2
        incoming_waves = info.wave_functions[:, :mode_count]
        outgoing_waves = info.wave_functions[:, mode_count:]
        incoming_lambdas = np.exp(1j * info.momenta[:mode_count])
        outgoing_lambdas = np.exp(1j * info.momenta[mode_count:])
        incoming.append(
            coupling @ incoming_waves
            - sigma @ (boundary @ (incoming_waves / incoming_lambdas))
        )
        outgoing.append(
            coupling @ outgoing_waves
            - sigma.conj().T
            @ (boundary @ (outgoing_waves / outgoing_lambdas))
        )
    return incoming, outgoing


def _lead_selection(lead_count, out_leads, in_leads):
    out_leads = list(
        range(lead_count) if out_leads is None else out_leads
    )
    in_leads = list(
        range(lead_count) if in_leads is None else in_leads
    )
    if (
        np.any(np.diff(out_leads) <= 0)
        or np.any(np.diff(in_leads) <= 0)
    ):
        raise ValueError("Lead lists must be sorted and with unique entries.")
    if not out_leads or not in_leads:
        raise ValueError("No output is requested.")
    return out_leads, in_leads


def _interface_bases(syst, args, params):
    offsets = syst._site_slices(args, params)
    return tuple(
        np.concatenate(
            [
                np.arange(offsets[index], offsets[index + 1])
                for index in interface
            ]
        )
        for interface in syst.lead_interfaces
    )


def smatrix(
    syst,
    energy=0,
    args=(),
    out_leads=None,
    in_leads=None,
    check_hermiticity=True,
    *,
    params=None,
    **kwargs,
):
    del kwargs
    _check_precalculated(syst, {"modes", "all"})
    lead_count = len(syst.leads)
    out_leads, in_leads = _lead_selection(
        lead_count, out_leads, in_leads
    )
    selected = set(out_leads) | set(in_leads)
    if any(
        not hasattr(syst.leads[index], "modes") for index in selected
    ):
        raise ValueError(
            "Scattering matrix blocks require propagating lead modes"
        )
    preliminary_info = tuple(
        (
            lead.modes(energy, args=args, params=params)[0]
            if hasattr(lead, "modes")
            else None
        )
        for lead in syst.leads
    )
    channel_counts = tuple(
        len(info.momenta) // 2 if info is not None else 0
        for info in preliminary_info
    )
    native, _ = _solution(
        syst,
        energy,
        args,
        params,
        channel_counts,
    )
    selfenergies = tuple(
        np.asarray(value, dtype=complex)
        for value in native.self_energies
    )
    lead_info = tuple(
        selfenergies[index] if info is None else info
        for index, info in enumerate(preliminary_info)
    )
    incoming_factors, outgoing_factors = _mode_factors(
        syst, lead_info, selfenergies, native, args, params
    )
    data, lead_offsets, physical_transmissions = (
        native.scattering_matrix(
            [value.tolist() for value in incoming_factors],
            [value.tolist() for value in outgoing_factors],
        )
    )
    total_channels = int(lead_offsets[-1])
    data = np.asarray(data, dtype=complex).reshape(
        (total_channels, total_channels)
    )
    lead_slices = tuple(
        slice(lead_offsets[index], lead_offsets[index + 1])
        for index in range(lead_count)
    )
    row_indices = np.concatenate(
        [np.arange(data.shape[0])[lead_slices[index]] for index in out_leads]
    )
    column_indices = np.concatenate(
        [np.arange(data.shape[1])[lead_slices[index]] for index in in_leads]
    )
    return SMatrix(
        data[np.ix_(row_indices, column_indices)],
        physical_transmissions,
        lead_slices,
        lead_info,
        out_leads,
        in_leads,
        check_hermiticity,
    )


def greens_function(
    syst,
    energy=0,
    args=(),
    out_leads=None,
    in_leads=None,
    check_hermiticity=True,
    *,
    params=None,
    **kwargs,
):
    del kwargs
    _check_precalculated(syst, {"selfenergy", "all"})
    lead_count = len(syst.leads)
    out_leads, in_leads = _lead_selection(
        lead_count, out_leads, in_leads
    )
    native, channel_counts = _solution(
        syst, energy, args, params
    )
    green = np.asarray(native.retarded_green, dtype=complex)
    selfenergies = tuple(
        np.asarray(value, dtype=complex)
        for value in native.self_energies
    )
    broadenings = tuple(
        np.asarray(value, dtype=complex)
        for value in native.broadenings
    )
    channel_counts = tuple(
        native.channel_counts(list(channel_counts))
    )
    transmissions = np.asarray(
        native.green_function_transmissions(list(channel_counts)),
        dtype=float,
    )
    interface_bases = _interface_bases(syst, args, params)
    interface_selfenergies = tuple(
        matrix[np.ix_(basis, basis)]
        for matrix, basis in zip(
            selfenergies, interface_bases, strict=True
        )
    )
    interface_broadenings = tuple(
        matrix[np.ix_(basis, basis)]
        for matrix, basis in zip(
            broadenings, interface_bases, strict=True
        )
    )
    rows = np.concatenate(
        [interface_bases[index] for index in out_leads]
    )
    columns = np.concatenate(
        [interface_bases[index] for index in in_leads]
    )
    return GreensFunction(
        green[np.ix_(rows, columns)],
        transmissions,
        interface_selfenergies,
        interface_broadenings,
        channel_counts,
        out_leads,
        in_leads,
        check_hermiticity,
    )


def ldos(syst, energy=0, args=(), *, params=None, **kwargs):
    _check_precalculated(syst, {"modes", "all"})
    if any(not hasattr(lead, "modes") for lead in syst.leads):
        raise NotImplementedError("LDOS requires propagating lead modes")
    channel_counts = tuple(
        len(lead.modes(energy, args=args, params=params)[0].momenta) // 2
        for lead in syst.leads
    )
    native, _ = _solution(
        syst, energy, args, params, channel_counts
    )
    return np.asarray(native.local_density_of_states, dtype=float)


class WaveFunction:
    def __init__(self, states):
        self._states = tuple(
            None if value is None else np.asarray(value, dtype=complex)
            for value in states
        )
        first = next(
            (value for value in self._states if value is not None), None
        )
        self.num_orb = 0 if first is None else first.shape[1]

    def __call__(self, lead):
        state = self._states[int(lead)]
        if state is None:
            raise ValueError(
                "Scattering wave functions require propagating lead modes"
            )
        return state


def wave_function(syst, energy=0, args=(), *, params=None, **kwargs):
    _check_precalculated(syst, {"modes", "all"})
    channel_counts = tuple(
        (
            len(
                lead.modes(
                    energy, args=args, params=params
                )[0].momenta
            )
            // 2
            if hasattr(lead, "modes")
            else 0
        )
        for lead in syst.leads
    )
    native, _ = _solution(
        syst, energy, args, params, channel_counts
    )
    selfenergies = tuple(
        np.asarray(value, dtype=complex)
        for value in native.self_energies
    )
    lead_info = tuple(
        (
            lead.modes(energy, args=args, params=params)[0]
            if hasattr(lead, "modes")
            else selfenergies[index]
        )
        for index, lead in enumerate(syst.leads)
    )
    incoming_factors, _ = _mode_factors(
        syst,
        lead_info,
        selfenergies,
        native,
        args,
        params,
    )
    native_states = native.scattering_states(
        [value.tolist() for value in incoming_factors]
    )
    states = [
        (
            None
            if not hasattr(info, "momenta")
            else np.asarray(state, dtype=complex).reshape(
                (factor.shape[1], factor.shape[0])
            )
        )
        for info, state, factor in zip(
            lead_info,
            native_states,
            incoming_factors,
            strict=True,
        )
    ]
    return WaveFunction(states)


class SparseSolver:
    """Reusable sparse-solver facade over the Thouless transport entry points."""

    smatrix = staticmethod(smatrix)
    greens_function = staticmethod(greens_function)
    ldos = staticmethod(ldos)
    wave_function = staticmethod(wave_function)


class Solver(SparseSolver):
    """Sparse-solver compatible facade over the shared transport kernel."""

    lhsformat = "csc"
    rhsformat = "csc"

    def __init__(self):
        self.nrhs = 6
        self.ordering = "auto"
        self.sparse_rhs = False

    def options(self, nrhs=None, ordering=None, sparse_rhs=None):
        old = {
            "nrhs": self.nrhs,
            "ordering": self.ordering,
            "sparse_rhs": self.sparse_rhs,
        }
        if nrhs is not None:
            if int(nrhs) != nrhs or int(nrhs) < 1:
                raise ValueError("nrhs must be a positive integer")
            self.nrhs = int(nrhs)
        if ordering is not None:
            if ordering not in {
                "amd",
                "amf",
                "auto",
                "kwant_decides",
                "metis",
                "pord",
                "scotch",
            }:
                raise ValueError(f"Invalid ordering: {ordering}")
            self.ordering = (
                "auto" if ordering == "kwant_decides" else ordering
            )
        if sparse_rhs is not None:
            self.sparse_rhs = bool(sparse_rhs)
        return old

    def reset_options(self):
        old = {
            "nrhs": self.nrhs,
            "ordering": self.ordering,
            "sparse_rhs": self.sparse_rhs,
        }
        self.nrhs = 6
        self.ordering = "auto"
        self.sparse_rhs = False
        return old


sparse = types.ModuleType(f"{__name__}.sparse")
sparse.Solver = Solver
sparse.smatrix = smatrix
sparse.greens_function = greens_function
sparse.ldos = ldos
sparse.wave_function = wave_function
sparse.__all__ = [
    "smatrix",
    "greens_function",
    "ldos",
    "wave_function",
    "Solver",
]
sys.modules[sparse.__name__] = sparse

common = types.ModuleType(f"{__name__}.common")
common.BlockResult = BlockResult
common.GreensFunction = GreensFunction
common.SMatrix = SMatrix
common.SparseSolver = SparseSolver
common.__all__ = ["SparseSolver", "SMatrix", "GreensFunction"]
sys.modules[common.__name__] = common

default = types.ModuleType(f"{__name__}.default")
default.greens_function = greens_function
default.ldos = ldos
default.smatrix = smatrix
default.wave_function = wave_function
default.__all__ = [
    "smatrix",
    "ldos",
    "wave_function",
    "greens_function",
]
sys.modules[default.__name__] = default

mumps = types.ModuleType(f"{__name__}.mumps")
mumps.Solver = Solver
mumps.default_solver = Solver()
mumps.greens_function = greens_function
mumps.ldos = ldos
mumps.smatrix = smatrix
mumps.wave_function = wave_function
mumps.options = mumps.default_solver.options
mumps.reset_options = mumps.default_solver.reset_options
mumps.__all__ = [
    "smatrix",
    "ldos",
    "wave_function",
    "greens_function",
    "options",
    "Solver",
]
sys.modules[mumps.__name__] = mumps


__all__ = [
    "BlockResult",
    "GreensFunction",
    "SMatrix",
    "Solver",
    "SparseSolver",
    "WaveFunction",
    "common",
    "default",
    "greens_function",
    "ldos",
    "mumps",
    "smatrix",
    "sparse",
    "wave_function",
]
