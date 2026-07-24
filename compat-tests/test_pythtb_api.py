"""Clean-room smoke contracts for the PythTB-compatible entry point.

These tests are not a substitute for the pinned upstream test suite.
"""

from __future__ import annotations

import numpy as np
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


def test_reciprocal_path_uses_cartesian_arc_length() -> None:
    pythtb = require_compat_module("pythtb", ISSUE_URL)
    lattice = pythtb.Lattice(
        [[2.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0]],
        periodic_dirs=[0, 1],
    )

    points, distances, nodes = lattice.k_path(
        [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]],
        7,
    )

    assert points[0] == pytest.approx([0.0, 0.0])
    assert points[-1] == pytest.approx([0.5, 0.5])
    assert nodes == pytest.approx([0.0, 0.5 * np.pi, 1.5 * np.pi])
    assert distances[-1] == pytest.approx(nodes[-1])


def test_finite_position_projection_uses_rust_observable_core() -> None:
    pythtb = require_compat_module("pythtb", ISSUE_URL)
    model = pythtb.tb_model(
        0,
        1,
        lat=[[1.0]],
        orb=[[0.0], [2.0]],
    )
    eigenvectors = np.array(
        [
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            [1 / np.sqrt(2), -1 / np.sqrt(2)],
        ],
        dtype=complex,
    )

    position = model.position_matrix(eigenvectors, 0)
    expectation = model.position_expectation(eigenvectors, 0)

    np.testing.assert_allclose(position, [[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(expectation, [1.0, 1.0])
