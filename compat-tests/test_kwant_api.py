"""Clean-room smoke contracts for the Kwant-compatible entry point.

These tests are not a substitute for the pinned upstream test suite.
"""

from __future__ import annotations

import numpy as np
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
    sparse_hamiltonian = system.finalized().hamiltonian_submatrix(sparse=True)
    assert sparse_hamiltonian.format == "coo"
    np.testing.assert_allclose(sparse_hamiltonian.toarray(), hamiltonian)


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


def test_two_channel_ballistic_scattering_contract() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    lattice = kwant.lattice.square(norbs=1)
    system = kwant.Builder()
    for x in range(3):
        for y in range(2):
            system[lattice(x, y)] = 0.0
            if x:
                system[lattice(x - 1, y), lattice(x, y)] = -1.0

    lead = kwant.Builder(kwant.TranslationalSymmetry(lattice.vec((-1, 0))))
    for y in range(2):
        lead[lattice(0, y)] = 0.0
        lead[lattice(0, y), lattice(-1, y)] = -1.0
    system.leads.append(
        kwant.builder.BuilderLead(
            lead,
            [lattice(0, 0), lattice(0, 1)],
        )
    )
    system.leads.append(
        kwant.builder.BuilderLead(
            lead.reversed(),
            [lattice(2, 0), lattice(2, 1)],
        )
    )

    scattering = kwant.smatrix(system.finalized(), energy=0.0)
    assert scattering.transmission(1, 0) == pytest.approx(2.0)


def test_green_function_and_ldos_contract() -> None:
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
    finalized = system.finalized()

    green = kwant.greens_function(finalized, energy=0.0)
    assert green.data.shape == (3, 3)
    assert green.transmission(1, 0) == pytest.approx(1.0)
    density = kwant.ldos(finalized, energy=0.0)
    assert density.shape == (3,)
    assert np.all(density > 0)


def test_local_operator_entry_points_exist() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    assert callable(kwant.operator.Density)
    assert callable(kwant.operator.Current)
    assert callable(kwant.operator.Source)
