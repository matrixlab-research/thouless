"""Steady-state solver entry points backed by Thouless Rust transport."""

from __future__ import annotations

import sys

import numpy as np

from thouless import _core


class SMatrix:
    """Scattering amplitudes in a lead-resolved broadening eigenchannel basis."""

    def __init__(self, data, transmissions, lead_slices):
        self.data = np.asarray(data, dtype=complex)
        self._transmissions = np.asarray(transmissions, dtype=float)
        self._lead_slices = tuple(lead_slices)

    def transmission(self, out_lead, in_lead):
        return float(self._transmissions[int(out_lead), int(in_lead)])

    def submatrix(self, out_lead, in_lead):
        return self.data[
            self._lead_slices[int(out_lead)],
            self._lead_slices[int(in_lead)],
        ]


class GreensFunction:
    """Retarded device Green function and lead-to-lead transmissions."""

    def __init__(self, data, transmissions, selfenergies, broadenings):
        self.data = np.asarray(data, dtype=complex)
        self._transmissions = np.asarray(transmissions, dtype=float)
        self.selfenergies = tuple(
            np.asarray(value, dtype=complex) for value in selfenergies
        )
        self.broadenings = tuple(
            np.asarray(value, dtype=complex) for value in broadenings
        )

    def transmission(self, out_lead, in_lead):
        return float(self._transmissions[int(out_lead), int(in_lead)])


def _solution(syst, energy, args, params):
    fixed_energy = getattr(syst, "_precalculated_energy", None)
    if fixed_energy is not None and float(energy) != fixed_energy:
        raise ValueError(
            f"System was precalculated at energy {fixed_energy}, not {energy}"
        )
    device, leads = syst._transport_data(args=args, params=params)
    green, selfenergies, broadenings, transmissions = (
        _core.open_system_solution(
            device.tolist(),
            leads,
            float(energy),
        )
    )
    return (
        np.asarray(green, dtype=complex),
        selfenergies,
        broadenings,
        transmissions,
    )


def _scattering_matrix(green, broadenings):
    factors = []
    for broadening in broadenings:
        matrix = np.asarray(broadening, dtype=complex)
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (matrix + matrix.conj().T)
        )
        scale = max(float(np.max(eigenvalues, initial=0.0)), 1.0)
        propagating = eigenvalues > 1.0e-10 * scale
        factors.append(
            eigenvectors[:, propagating]
            * np.sqrt(np.maximum(eigenvalues[propagating], 0.0))
        )
    offsets = np.cumsum([0, *(factor.shape[1] for factor in factors)])
    data = np.zeros((offsets[-1], offsets[-1]), dtype=complex)
    slices = tuple(
        slice(offsets[index], offsets[index + 1])
        for index in range(len(factors))
    )
    for drain, drain_factor in enumerate(factors):
        for source, source_factor in enumerate(factors):
            block = -1j * drain_factor.conj().T @ green @ source_factor
            if drain == source:
                block += np.eye(block.shape[0])
            data[slices[drain], slices[source]] = block
    return data, slices


def smatrix(syst, energy=0, args=(), out_leads=None, in_leads=None, *, params=None, **kwargs):
    green, _, broadenings, transmissions = _solution(
        syst,
        energy,
        args,
        params,
    )
    data, lead_slices = _scattering_matrix(green, broadenings)
    return SMatrix(data, transmissions, lead_slices)


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
    green, selfenergies, broadenings, transmissions = _solution(
        syst, energy, args, params
    )
    return GreensFunction(green, transmissions, selfenergies, broadenings)


def ldos(syst, energy=0, args=(), *, params=None, **kwargs):
    green, _, _, _ = _solution(syst, energy, args, params)
    return -np.imag(np.diag(green)) / np.pi


default = sys.modules[__name__]


__all__ = [
    "GreensFunction",
    "SMatrix",
    "default",
    "greens_function",
    "ldos",
    "smatrix",
]
