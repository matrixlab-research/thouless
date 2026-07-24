"""Steady-state solver entry points backed by Thouless Rust transport."""

from __future__ import annotations

import sys
import types

import numpy as np

from thouless import _core

_SURFACE_BROADENING = 1.0e-6


class SMatrix:
    """Scattering amplitudes in the propagating-mode basis of each lead."""

    def __init__(
        self,
        data,
        transmissions,
        lead_slices,
        lead_info=(),
        out_leads=None,
        in_leads=None,
    ):
        self.data = np.asarray(data, dtype=complex)
        self._transmissions = np.asarray(transmissions, dtype=float)
        self.lead_info = tuple(lead_info)
        lead_count = len(self.lead_info)
        self.out_leads = list(
            range(lead_count) if out_leads is None else out_leads
        )
        self.in_leads = list(
            range(lead_count) if in_leads is None else in_leads
        )
        sizes = [
            int(lead_slices[index].stop - lead_slices[index].start)
            for index in range(lead_count)
        ]
        out_offsets = np.cumsum(
            [0, *(sizes[index] for index in self.out_leads)]
        )
        in_offsets = np.cumsum(
            [0, *(sizes[index] for index in self.in_leads)]
        )
        self._out_slices = {
            lead: slice(out_offsets[position], out_offsets[position + 1])
            for position, lead in enumerate(self.out_leads)
        }
        self._in_slices = {
            lead: slice(in_offsets[position], in_offsets[position + 1])
            for position, lead in enumerate(self.in_leads)
        }

    def transmission(self, out_lead, in_lead):
        _require_transmission(
            int(out_lead),
            int(in_lead),
            self.out_leads,
            self.in_leads,
            len(self.lead_info),
        )
        return float(self._transmissions[int(out_lead), int(in_lead)])

    def submatrix(self, out_lead, in_lead):
        return self.data[
            self._out_slices[int(out_lead)],
            self._in_slices[int(in_lead)],
        ]

    def num_propagating(self, lead):
        info = self.lead_info[int(lead)]
        return len(getattr(info, "momenta", ())) // 2

    def conductance_matrix(self):
        return _conductance_matrix(self)


class GreensFunction:
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
    ):
        self.data = np.asarray(data, dtype=complex)
        self._transmissions = np.asarray(transmissions, dtype=float)
        self.selfenergies = tuple(
            np.asarray(value, dtype=complex) for value in selfenergies
        )
        self.lead_info = self.selfenergies
        self.broadenings = tuple(
            np.asarray(value, dtype=complex) for value in broadenings
        )
        self._channel_counts = tuple(channel_counts)
        lead_count = len(self.selfenergies)
        self.out_leads = list(
            range(lead_count) if out_leads is None else out_leads
        )
        self.in_leads = list(
            range(lead_count) if in_leads is None else in_leads
        )

    def transmission(self, out_lead, in_lead):
        _require_transmission(
            int(out_lead),
            int(in_lead),
            self.out_leads,
            self.in_leads,
            len(self.lead_info),
        )
        return float(self._transmission(out_lead, in_lead))

    def _a_ttdagger_a_inv(self, lead_out, lead_in):
        return (
            self.broadenings[int(lead_out)]
            @ self.data
            @ self.broadenings[int(lead_in)]
            @ self.data.conj().T
        )

    def _transmission(self, lead_out, lead_in):
        lead_out = int(lead_out)
        lead_in = int(lead_in)
        result = np.trace(
            self._a_ttdagger_a_inv(lead_out, lead_in)
        ).real
        if lead_out == lead_in:
            gamma = self.broadenings[lead_in]
            result += (
                2 * np.trace(gamma @ self.data).imag
                + self.num_propagating(lead_in)
            )
        return float(result)

    def num_propagating(self, lead):
        count = self._channel_counts[int(lead)]
        if count is not None:
            return int(count)
        gamma = self.broadenings[int(lead)]
        scale = max(np.linalg.norm(gamma, np.inf), 1.0)
        return int(
            np.sum(
                np.linalg.eigvalsh(
                    0.5 * (gamma + gamma.conj().T)
                )
                > 1.0e-10 * scale
            )
        )

    def conductance_matrix(self):
        return _conductance_matrix(self)


def _require_transmission(
    out_lead, in_lead, out_leads, in_leads, lead_count
):
    out_present = out_lead in out_leads
    in_present = in_lead in in_leads
    if out_present and in_present:
        return
    all_but_one = lead_count - 1
    if out_present != in_present:
        missing_axis = in_leads if out_present else out_leads
        if len(missing_axis) == all_but_one:
            return
    elif (
        len(out_leads) == all_but_one
        and len(in_leads) == all_but_one
    ):
        return
    raise ValueError(
        f"Insufficient matrix elements to compute "
        f"transmission({out_lead}, {in_lead})"
    )


def _conductance_matrix(result):
    lead_count = len(result.lead_info)
    matrix = np.asarray(
        [
            [
                -result.transmission(drain, source)
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


def _solution(syst, energy, args, params, channel_counts=None):
    device, leads = syst._transport_data(args=args, params=params)
    _, narrow_selfenergies, _, _ = _core.open_system_solution(
        device.tolist(),
        leads,
        float(energy),
    )
    _, wider_selfenergies, _, _ = _core.open_system_solution(
        device.tolist(),
        leads,
        float(energy),
        2 * _SURFACE_BROADENING,
    )
    selfenergies = [
        2 * np.asarray(narrow, dtype=complex)
        - np.asarray(wide, dtype=complex)
        for narrow, wide in zip(
            narrow_selfenergies, wider_selfenergies, strict=True
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
            gamma = 1j * (
                selfenergies[index] - selfenergies[index].conj().T
            )
            projected, _ = _physical_selfenergies(
                [selfenergies[index]], [gamma], [count]
            )
            selfenergies[index] = projected[0]
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
    broadenings = [
        1j * (value - value.conj().T) for value in selfenergies
    ]
    inverse_green = (
        float(energy) * np.eye(device.shape[0], dtype=complex)
        - device
        - sum(
            (np.asarray(value, dtype=complex) for value in selfenergies),
            start=np.zeros_like(device, dtype=complex),
        )
    )
    green = np.linalg.inv(inverse_green)
    transmissions = np.asarray(
        [
            [
                np.trace(
                    np.asarray(drain)
                    @ green
                    @ np.asarray(source)
                    @ green.conj().T
                ).real
                for source in broadenings
            ]
            for drain in broadenings
        ],
        dtype=float,
    )
    return (
        np.asarray(green, dtype=complex),
        selfenergies,
        broadenings,
        transmissions,
    )


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


def _physical_selfenergies(selfenergies, broadenings, channel_counts):
    projected_selfenergies = []
    projected_broadenings = []
    for selfenergy, broadening, channel_count in zip(
        selfenergies, broadenings, channel_counts, strict=True
    ):
        sigma = np.asarray(selfenergy, dtype=complex)
        gamma = np.asarray(broadening, dtype=complex)
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (gamma + gamma.conj().T)
        )
        order = np.argsort(eigenvalues)[::-1][: int(channel_count)]
        if len(order):
            vectors = eigenvectors[:, order]
            physical_gamma = (
                vectors
                * np.maximum(eigenvalues[order], 0.0)
            ) @ vectors.conj().T
        else:
            physical_gamma = np.zeros_like(gamma)
        hermitian_part = 0.5 * (sigma + sigma.conj().T)
        physical_sigma = hermitian_part - 0.5j * physical_gamma
        projected_selfenergies.append(physical_sigma)
        projected_broadenings.append(physical_gamma)
    return projected_selfenergies, projected_broadenings


def _check_precalculated(syst, allowed):
    what = getattr(syst, "_precalculated_what", None)
    if what is not None and what not in allowed:
        raise ValueError(
            f"System precalculated with {what!r}, expected one of {tuple(allowed)!r}"
        )


def _scattering_matrix(green, incoming_factors, outgoing_factors):
    offsets = np.cumsum(
        [0, *(factor.shape[1] for factor in incoming_factors)]
    )
    data = np.zeros((offsets[-1], offsets[-1]), dtype=complex)
    slices = tuple(
        slice(offsets[index], offsets[index + 1])
        for index in range(len(incoming_factors))
    )
    for drain, outgoing_factor in enumerate(outgoing_factors):
        for source, incoming_factor in enumerate(incoming_factors):
            block = (
                -1j
                * outgoing_factor.conj().T
                @ green
                @ incoming_factor
            )
            if drain == source:
                block += np.linalg.pinv(outgoing_factor) @ incoming_factor
            data[slices[drain], slices[source]] = block
    return data, slices


def _mode_factors(syst, lead_info, selfenergies, args, params):
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
            gamma = 1j * (
                np.asarray(selfenergy)
                - np.asarray(selfenergy).conj().T
            )
            eigenvalues, eigenvectors = np.linalg.eigh(
                0.5 * (gamma + gamma.conj().T)
            )
            mode_count = len(info.momenta) // 2
            order = np.argsort(eigenvalues)[::-1][:mode_count]
            factors = (
                eigenvectors[:, order]
                * np.sqrt(np.maximum(eigenvalues[order], 0.0))
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


def smatrix(syst, energy=0, args=(), out_leads=None, in_leads=None, *, params=None, **kwargs):
    _check_precalculated(syst, {"modes", "all"})
    lead_count = len(syst.leads)
    out_leads = list(
        range(lead_count) if out_leads is None else out_leads
    )
    in_leads = list(
        range(lead_count) if in_leads is None else in_leads
    )
    if not out_leads or not in_leads:
        raise ValueError("At least one incoming and outgoing lead is required")
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
    green, selfenergies, _, _ = _solution(
        syst,
        energy,
        args,
        params,
        channel_counts,
    )
    lead_info = tuple(
        selfenergies[index] if info is None else info
        for index, info in enumerate(preliminary_info)
    )
    incoming_factors, outgoing_factors = _mode_factors(
        syst, lead_info, selfenergies, args, params
    )
    data, lead_slices = _scattering_matrix(
        green, incoming_factors, outgoing_factors
    )
    physical_transmissions = np.asarray(
        [
            [
                np.linalg.norm(
                    data[lead_slices[drain], lead_slices[source]]
                )
                ** 2
                for source in range(lead_count)
            ]
            for drain in range(lead_count)
        ]
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
    )


def greens_function(
    syst,
    energy=0,
    args=(),
    out_leads=None,
    in_leads=None,
    *,
    params=None,
    **kwargs,
):
    _check_precalculated(syst, {"selfenergy", "all"})
    green, selfenergies, broadenings, transmissions = _solution(
        syst, energy, args, params
    )
    channel_counts = tuple(
        (
            len(
                lead.modes(
                    energy, args=args, params=params
                )[0].momenta
            )
            // 2
            if hasattr(lead, "modes")
            else None
        )
        for lead in syst.leads
    )
    return GreensFunction(
        green,
        transmissions,
        selfenergies,
        broadenings,
        channel_counts,
        out_leads,
        in_leads,
    )


def ldos(syst, energy=0, args=(), *, params=None, **kwargs):
    _check_precalculated(syst, {"modes", "all"})
    if any(not hasattr(lead, "modes") for lead in syst.leads):
        raise NotImplementedError("LDOS requires propagating lead modes")
    channel_counts = tuple(
        len(lead.modes(energy, args=args, params=params)[0].momenta) // 2
        for lead in syst.leads
    )
    green, _, _, _ = _solution(
        syst, energy, args, params, channel_counts
    )
    return -np.imag(np.diag(green)) / np.pi


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
    green, selfenergies, _, _ = _solution(
        syst, energy, args, params, channel_counts
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
        args,
        params,
    )
    states = [
        None if not hasattr(info, "momenta") else (green @ factor).T
        for info, factor in zip(
            lead_info, incoming_factors, strict=True
        )
    ]
    return WaveFunction(states)


default = sys.modules[__name__]


class Solver:
    """Sparse-solver compatible facade over the shared transport kernel."""

    smatrix = staticmethod(smatrix)
    greens_function = staticmethod(greens_function)
    ldos = staticmethod(ldos)
    wave_function = staticmethod(wave_function)


sparse = types.ModuleType(f"{__name__}.sparse")
sparse.Solver = Solver
sys.modules[sparse.__name__] = sparse


__all__ = [
    "GreensFunction",
    "SMatrix",
    "Solver",
    "WaveFunction",
    "default",
    "greens_function",
    "ldos",
    "smatrix",
    "sparse",
    "wave_function",
]
