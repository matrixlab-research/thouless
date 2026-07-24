"""Clean-room smoke contracts for the Kwant-compatible entry point.

These tests are not a substitute for the pinned upstream test suite.
"""

from __future__ import annotations

import pytest

from conftest import require_compat_module


ISSUE_URL = "https://github.com/matrixlab-research/thouless/issues/5"


def test_builder_and_finalized_system_contract() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    lattice = kwant.lattice.chain(norbs=1)
    system = kwant.Builder()
    system[lattice(0)] = 0.0
    system[lattice(1)] = 0.0
    system[lattice(0), lattice(1)] = -1.0

    hamiltonian = system.finalized().hamiltonian_submatrix()
    assert hamiltonian.shape == (2, 2)
    assert hamiltonian[0, 1] == pytest.approx(-1.0)
    assert hamiltonian[1, 0] == pytest.approx(-1.0)


def test_ballistic_chain_scattering_contract() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    lattice = kwant.lattice.chain(norbs=1)
    system = kwant.Builder()
    for index in range(3):
        system[lattice(index)] = 0.0
    system[lattice.neighbors()] = -1.0

    lead = kwant.Builder(kwant.TranslationalSymmetry(lattice.vec((-1,))))
    lead[lattice(0)] = 0.0
    lead[lattice.neighbors()] = -1.0
    system.attach_lead(lead)
    system.attach_lead(lead.reversed())

    scattering = kwant.smatrix(system.finalized(), energy=0.0)
    assert scattering.transmission(1, 0) == pytest.approx(1.0)


def test_local_operator_entry_points_exist() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    assert callable(kwant.operator.Density)
    assert callable(kwant.operator.Current)
    assert callable(kwant.operator.Source)
