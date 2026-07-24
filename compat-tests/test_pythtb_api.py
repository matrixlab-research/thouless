"""Clean-room smoke contracts for the PythTB-compatible entry point.

These tests are not a substitute for the pinned upstream test suite.
"""

from __future__ import annotations

import pytest

from conftest import require_compat_module


ISSUE_URL = "https://github.com/matrixlab-research/thouless/issues/4"


def test_periodic_model_construction_and_spectrum_contract() -> None:
    pythtb = require_compat_module("pythtb", ISSUE_URL)
    model = pythtb.tb_model(1, 1, lat=[[1.0]], orb=[[0.0]])
    model.set_onsite([0.25])
    model.set_hop(-1.0, 0, 0, [1])

    energies = model.solve_one([0.0])
    assert float(energies[0]) == pytest.approx(-1.75)


def test_finite_model_and_cut_contract() -> None:
    pythtb = require_compat_module("pythtb", ISSUE_URL)
    model = pythtb.tb_model(1, 1, lat=[[1.0]], orb=[[0.0]])
    model.set_hop(-1.0, 0, 0, [1])

    finite = model.cut_piece(2, 0, glue_edgs=False)
    energies = sorted(float(value) for value in finite.solve_all())
    assert energies == pytest.approx([-1.0, 1.0])


def test_wavefunction_array_entry_points_exist() -> None:
    pythtb = require_compat_module("pythtb", ISSUE_URL)
    model = pythtb.tb_model(1, 1, lat=[[1.0]], orb=[[0.0]])
    model.set_hop(-1.0, 0, 0, [1])

    wavefunctions = pythtb.wf_array(model, [9])
    assert callable(wavefunctions.solve_on_grid)
    assert callable(wavefunctions.berry_phase)
