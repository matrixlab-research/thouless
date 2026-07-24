"""Steady-state solver entry points backed by Thouless Rust transport."""

from __future__ import annotations

import numpy as np

from thouless import _core


class SMatrix:
    """Lead-to-lead transmission summary at one energy."""

    def __init__(self, transmissions):
        self._transmissions = np.asarray(transmissions, dtype=float)

    def transmission(self, out_lead, in_lead):
        return float(self._transmissions[int(out_lead), int(in_lead)])


def smatrix(syst, energy=0, args=(), out_leads=None, in_leads=None, *, params=None, **kwargs):
    device, leads = syst._transport_data(args=args, params=params)
    transmissions = _core.open_system_transmissions(
        device.tolist(),
        leads,
        float(energy),
    )
    return SMatrix(transmissions)


__all__ = ["SMatrix", "smatrix"]
