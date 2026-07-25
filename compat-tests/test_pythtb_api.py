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

    endpoint_mesh = lattice.k_uniform_mesh(
        [3, 2],
        gamma_centered=True,
        include_endpoints=True,
    )
    np.testing.assert_allclose(endpoint_mesh[0], [-0.5, -0.5])
    np.testing.assert_allclose(endpoint_mesh[-1], [0.5, 0.5])
    open_mesh = lattice.k_uniform_mesh(
        [3, 2],
        include_endpoints=False,
    )
    assert np.all(open_mesh < 1.0)

    path_distance, node_distance, node_indices, label_indices = (
        lattice.get_kpath_distance(
            points,
            k_nodes=points[[0, -1]],
            labels=["start", "finish"],
        )
    )
    np.testing.assert_allclose(path_distance, distances)
    np.testing.assert_allclose(node_distance, distances[[0, -1]])
    assert node_indices[str(points[0])] == [0]
    assert label_indices["finish"] == pytest.approx(points[-1])


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


def test_massive_dirac_quantum_geometry_uses_rust_kubo_core() -> None:
    pythtb = require_compat_module("pythtb", ISSUE_URL)
    lattice = pythtb.Lattice(
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0]],
        periodic_dirs=[0, 1],
    )
    model = pythtb.TBModel(lattice, spinful=True)
    mass = 2.0
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    model.set_onsite([[0.0, 0.0, 0.0, mass]])
    model.set_hop(-0.5j * sigma_x, 0, 0, [1, 0])
    model.set_hop(-0.5j * sigma_y, 0, 0, [0, 1])

    tensor = model.quantum_geometric_tensor(
        [[0.0, 0.0]],
        occ_idxs=[0],
    )
    curvature = model.berry_curvature(
        [[0.0, 0.0]],
        occ_idxs=[0],
        plane=(0, 1),
    )
    metric = model.quantum_metric(
        [[0.0, 0.0]],
        occ_idxs=[0],
        plane=(0, 0),
    )

    diagonal = np.pi**2 / mass**2
    assert tensor.shape == (2, 2, 1)
    assert tensor[0, 0, 0] == pytest.approx(diagonal)
    assert tensor[0, 1, 0] == pytest.approx(-1j * diagonal)
    assert curvature[0] == pytest.approx(2 * diagonal)
    assert metric[0] == pytest.approx(diagonal)


def test_predefined_models_cover_the_pythtb_constructor_set() -> None:
    pythtb = require_compat_module("pythtb", ISSUE_URL)
    from pythtb import models

    constructors = {
        "checkerboard",
        "fu_kane_mele",
        "graphene",
        "haldane",
        "kane_mele",
        "ssh",
    }
    assert constructors == set(models.__all__)
    assert models.checkerboard(0.2, -1.0).nstate == 2
    assert models.graphene(0.2, -1.0).nstate == 2
    assert models.haldane(0.2, -1.0, 0.1).nstate == 2
    assert models.kane_mele(0.2, -1.0, 0.1, 0.05).nstate == 4
    assert models.fu_kane_mele(1.0, 0.1).nstate == 4

    ssh = models.ssh(1.0, 2.0)
    np.testing.assert_allclose(
        ssh.solve_ham([[0.0], [0.5]]),
        [[-3.0, 3.0], [-1.0, 1.0]],
        atol=1e-12,
    )
    assert pythtb.TBModel is type(ssh)

    from pythtb.models.graphene import graphene
    from pythtb.models.ssh import ssh as ssh_constructor

    assert graphene(0.2, -1.0).nstate == 2
    assert ssh_constructor(1.0, 2.0).nstate == 2
    assert callable(models.graphene)
    assert callable(models.ssh)


def test_hopping_table_supports_general_mutation_and_flattening() -> None:
    require_compat_module("pythtb", ISSUE_URL)
    from pythtb.hoptable import HoppingTable

    table = HoppingTable(dim_r=2, spinful=False)
    row = table.append(1 + 2j, 0, 1, [1, 0])
    table.extend([-0.5], [1], [0], [[0, 1]])
    assert row == 0
    assert table.find(0, 1, [1, 0]) == 0
    assert len(table) == 2

    table.add(0, -1j)
    table.update(1, R=[0, -1])
    np.testing.assert_allclose(table.amplitudes, [1 + 1j, -0.5])
    assert table.find(1, 0, [0, -1]) == 1
    cache = table.flatten_cache(norb=2)
    np.testing.assert_array_equal(cache["uniq"], [1, 2])
    np.testing.assert_array_equal(cache["cols_transposed"], [2, 1])


def test_wannier_projection_spread_and_interpolation_are_general() -> None:
    pythtb = require_compat_module("pythtb", ISSUE_URL)
    from pythtb.models.ssh import ssh

    model = ssh(1.0, 2.0)
    mesh = pythtb.Mesh(["k"], dim_k=1)
    mesh.build_grid((31,), k_endpoints=False)
    states = pythtb.WFArray(
        model.lattice,
        mesh,
        nstates=model.nstate,
        spinful=False,
    )
    states.solve_model(model)

    wannier = pythtb.Wannier(states)
    assert wannier.project([[(0, 1.0)]], band_idxs=[0]) is None
    np.testing.assert_allclose(
        np.sum(np.abs(wannier.wannier) ** 2),
        1.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(wannier.WFs, wannier.wannier)
    np.testing.assert_allclose(wannier.centers, [[-0.25]], atol=1e-12)
    np.testing.assert_allclose(wannier.spread, [0.14472717], atol=1e-8)
    assert wannier.Omega_I == pytest.approx(0.10337476098363751)
    assert wannier.Omega_D == pytest.approx(0.041352405470848536)
    assert wannier.Omega_OD == pytest.approx(0.0, abs=1e-12)

    path = [[0.0], [0.5]]
    interpolated = wannier.interp_bands(path, n_interp=11)
    k_points, _, _ = model.k_path(path, 11)
    direct = model.solve_ham(k_points)[:, :1]
    # A finite 31-point Fourier representation truncates the exact square-root
    # dispersion; the observed uniform error is below one part per million.
    np.testing.assert_allclose(interpolated, direct, atol=1e-6)

    initial_spread = float(np.sum(wannier.spread))
    assert wannier.maxloc() is None
    assert float(np.sum(wannier.spread)) <= initial_spread + 1e-12

    optimized = pythtb.Wannier(states)
    optimized.set_trial_wfs([[(0, 1.0)]])
    assert (
        optimized.min_spread(
            n_wfs=1,
            outer_window="all",
            inner_window={"bands": [0]},
            max_iter=3,
            max_iter_dis=3,
        )
        is None
    )
    assert optimized.tilde_states.nstates == 1
    assert np.isfinite(optimized.Omega_I)


def test_mesh_model_and_wannier_visualization_entry_points() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pythtb = require_compat_module("pythtb", ISSUE_URL)
    from matplotlib import pyplot as plt
    from pythtb import models
    from pythtb.models.haldane import haldane
    from pythtb.visualization import plot_bands, plot_lattice, plot_tbmodel

    model = haldane(0.2, -1.0, 0.1)
    for result in (
        plot_lattice(model.lattice),
        model.lattice.visualize(),
        plot_tbmodel(model, annotate_onsite=True),
        model.visualize(draw_hoppings=True),
        plot_bands(model, [[0.0, 0.0], [0.5, 0.0]], nk=7),
        model.plot_bands([[0.0, 0.0], [0.5, 0.0]], nk=7),
    ):
        fig, ax = result
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    model_3d = models.fu_kane_mele(1.0, 0.1)
    lattice_figure = model_3d.lattice.visualize_3d()
    model_figure = model_3d.visualize_3d(show=False)
    assert len(lattice_figure.data) >= 2
    assert len(model_figure.data) > len(lattice_figure.data)

    mesh = pythtb.Mesh(["k", "k"], dim_k=2)
    mesh.build_grid((5, 5), k_endpoints=False)
    states = pythtb.WFArray(
        model.lattice,
        mesh,
        nstates=model.nstate,
        spinful=False,
    )
    states.solve_model(model)
    wannier = pythtb.Wannier(states)
    wannier.project([[(0, 1.0)]], band_idxs=[0])
    for result in (
        wannier.plot_density(0, cbar=False),
        wannier.plot_decay(0),
        wannier.plot_centers(legend=False),
    ):
        fig, ax = result
        assert fig is not None
        assert ax is not None
        plt.close(fig)


def test_root_numerical_utilities_cover_source_exports() -> None:
    pythtb = require_compat_module("pythtb", ISSUE_URL)
    tensor = pythtb.levi_civita(3, 3)
    assert tensor[0, 1, 2] == 1
    assert tensor[1, 0, 2] == -1
    assert tensor[0, 0, 1] == 0

    coordinates = np.linspace(-1.0, 1.0, 9)
    derivative = pythtb.finite_difference(
        coordinates**3,
        axis=0,
        delta=coordinates[1] - coordinates[0],
        order=2,
    )
    np.testing.assert_allclose(derivative, 3 * coordinates**2, atol=0.125)
    assert pythtb.is_Hermitian([[1.0, 1j], [-1j, 2.0]])
    assert not pythtb.is_Hermitian([1.0, 2.0])
    np.testing.assert_allclose(
        pythtb.get_trial_wfs([[(0, 1.0), (1, 1.0)]], 2),
        [[1 / np.sqrt(2), 1 / np.sqrt(2)]],
    )


def test_wannier90_and_qe_text_import_builds_a_general_model(tmp_path) -> None:
    pythtb = require_compat_module("pythtb", ISSUE_URL)
    prefix = "chain"
    (tmp_path / f"{prefix}.win").write_text(
        "\n".join(
            [
                "begin unit_cell_cart",
                "ang",
                "1 0 0",
                "0 1 0",
                "0 0 1",
                "end unit_cell_cart",
                "begin kpoint_path",
                "G 0 0 0 X 0.5 0 0",
                "end kpoint_path",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / f"{prefix}_centres.xyz").write_text(
        "1\ncentres\nX 0 0 0\n",
        encoding="utf-8",
    )
    (tmp_path / f"{prefix}_hr.dat").write_text(
        "\n".join(
            [
                "generated fixture",
                "1",
                "3",
                "1 1 1",
                "-1 0 0 1 1 -1.0 0.0",
                "0 0 0 1 1 0.5 0.0",
                "1 0 0 1 1 -1.0 0.0",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / f"{prefix}_band.kpt").write_text(
        "2\n0 0 0 1\n0.5 0 0 1\n",
        encoding="utf-8",
    )
    (tmp_path / f"{prefix}_band.dat").write_text(
        "0.0 -1.5\n1.0 2.5\n",
        encoding="utf-8",
    )
    (tmp_path / f"{prefix}_bands.dat").write_text(
        "&plot nbnd=1, nks=2 /\n"
        "0 0 0\n"
        "-1.5\n"
        "0.5 0 0\n"
        "2.5\n",
        encoding="utf-8",
    )

    imported = pythtb.W90(tmp_path, prefix)
    model = imported.model()
    assert model.from_w90
    assert not model.assume_position_operator_diagonal
    np.testing.assert_allclose(
        model.solve_ham([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        [[-1.5], [2.5]],
        atol=1e-12,
    )
    k_points, energies, distance, nodes, labels = imported.bands_w90(
        return_k_dist=True,
        return_k_nodes=True,
    )
    np.testing.assert_allclose(k_points, [[0, 0, 0], [0.5, 0, 0]])
    np.testing.assert_allclose(energies, [[-1.5], [2.5]])
    np.testing.assert_allclose(distance, [0.0, np.pi])
    np.testing.assert_allclose(nodes, [[0, 0, 0], [0.5, 0, 0]])
    assert labels == [r"$\Gamma$", "$X$"]

    qe_k, qe_energies, metadata = imported.bands_qe(return_meta=True)
    np.testing.assert_allclose(qe_k, [[0, 0, 0], [0.5, 0, 0]])
    np.testing.assert_allclose(qe_energies, [[-1.5], [2.5]])
    assert metadata == {"nbnd": 1, "nks": 2}


def test_model_mutation_neighbor_shells_and_parameter_copies_are_general() -> None:
    pythtb = require_compat_module("pythtb", ISSUE_URL)
    lattice = pythtb.Lattice(
        [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]],
        [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]],
        periodic_dirs=[0, 1],
    )
    model = pythtb.TBModel(lattice)
    summaries, bonds = model.nn_bonds(2)
    assert [summary["degeneracy_total"] for summary in summaries] == [6, 12]
    assert [len(shell) for shell in bonds] == [3, 6]
    model.set_shell_hops({1: -1.0})
    np.testing.assert_allclose(
        model.solve_ham([[0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0]]),
        [[-3.0, 3.0], [0.0, 0.0]],
        atol=1e-12,
    )

    parameterized = pythtb.TBModel(
        pythtb.Lattice([[1.0]], [[0.0], [0.5]], periodic_dirs=[0])
    )
    parameterized.set_hop("v", 0, 1, [0])
    parameterized.set_hop("w", 1, 0, [1])
    resolved = parameterized.with_parameters(v=1.0, w=2.0)
    assert len(parameterized.parameters) == 2
    assert resolved.parameters == []
    np.testing.assert_allclose(
        resolved.solve_ham([[0.0], [0.5]]),
        [[-3.0, 3.0], [-1.0, 1.0]],
        atol=1e-12,
    )

    reset = resolved.copy()
    reset.clear_hoppings()
    reset.clear_onsite()
    reset.add_orb([0.25])
    assert reset.nhops == 0
    assert reset.norb == 3
    np.testing.assert_allclose(reset.onsite, 0.0)


def test_finite_haldane_local_chern_marker_uses_rust_projector_core() -> None:
    require_compat_module("pythtb", ISSUE_URL)
    from pythtb import models

    finite = models.haldane(0.0, -1.0, 0.15).make_finite(
        periodic_dirs=[0, 1],
        num_cells=[5, 5],
    )
    marker, bulk = finite.local_chern_marker(
        return_bulk_avg=True,
        trim_cells=1,
    )

    assert finite.dim_k == 0
    assert finite.norb == 50
    assert finite.lattice.nsuper == [5, 5]
    assert marker.shape == (50,)
    assert bulk == pytest.approx(-0.9426725430769424, abs=1e-10)
    assert np.sum(marker) == pytest.approx(0.0, abs=1e-10)


def test_mesh_axis_and_wavefunction_diagnostics_cover_pythtb_2_api() -> None:
    pythtb = require_compat_module("pythtb", ISSUE_URL)
    from pythtb.mesh import Axis
    from pythtb import models

    axis = Axis("k", "momentum")
    axis.add_loop_component(0)
    axis.add_endpoint_component(0)
    axis.add_wind_bz_component(0)
    assert axis.is_loop and axis.has_endpoint and axis.winds_bz
    axis.remove_endpoint_component(0)
    assert not axis.has_endpoint

    mixed = pythtb.Mesh(["k", "l"], axis_names=["k", "mass"], dim_k=1)
    mixed.build_grid((3, 4), lambda_start=-1.0, lambda_stop=1.0)
    assert mixed.component_types == ("k", "l")
    assert mixed.get_k_points().shape == (3, 1)
    assert mixed.get_param_points().shape == (4, 1)
    assert "Mesh report" in mixed.info(show=False)
    np.testing.assert_allclose(
        pythtb.Mesh.gen_hyper_cube(2, 2),
        [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]],
    )

    path = pythtb.Mesh(["k"], dim_k=2)
    path.build_path([[0.0, 0.0], [1.0, 1.0]], n_interp=3)
    assert path.flat.shape == (4, 2)
    np.testing.assert_allclose(path.nodes, [[0.0, 0.0], [1.0, 1.0]])

    model = models.haldane(0.2, -1.0, 0.15)
    mesh = pythtb.Mesh(["k", "k"])
    mesh.build_grid((7, 7))
    wavefunctions = pythtb.WFArray(model.lattice, mesh)
    wavefunctions.solve_model(model)
    projector, complement = wavefunctions.projectors([0], return_Q=True)
    np.testing.assert_allclose(
        projector @ projector,
        projector,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        projector + complement,
        np.broadcast_to(
            np.eye(model.nstate),
            projector.shape,
        ),
        atol=1e-12,
    )
    assert wavefunctions.overlap_matrix().shape == (7, 7, 4, 2, 2)
    abelian = wavefunctions.berry_flux(plane=(0, 1), state_idx=[0])
    non_abelian = wavefunctions.berry_flux(
        plane=(0, 1),
        state_idx=[0],
        non_abelian=True,
    )
    full = wavefunctions.berry_flux(state_idx=[0])
    np.testing.assert_allclose(
        abelian,
        non_abelian[..., 0, 0].real,
        atol=1e-12,
    )
    np.testing.assert_allclose(full[1, 0], -full[0, 1], atol=1e-12)
    assert np.sum(abelian) / (2 * np.pi) == pytest.approx(-1.0, abs=1e-12)
