"""Clean-room smoke contracts for the Kwant-compatible entry point.

These tests are not a substitute for the pinned upstream test suite.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

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


def test_low_level_system_protocol_and_plotter_iterators() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    lattice = kwant.lattice.chain(norbs=1)
    builder = kwant.Builder()
    builder[lattice(0)] = 0.0
    builder[lattice(1)] = 0.0
    builder[lattice(0), lattice(1)] = -1.0
    finalized = builder.finalized()

    assert isinstance(finalized, kwant.system.System)
    assert isinstance(finalized, kwant.system.FiniteSystem)
    np.testing.assert_allclose(finalized.pos(1), [1.0])

    sites, lead_slices = kwant.plotter.sys_leads_sites(finalized)
    assert sites == [(0, None, 0), (1, None, 0)]
    assert lead_slices == []
    np.testing.assert_allclose(
        kwant.plotter.sys_leads_pos(finalized, sites),
        [[0.0], [1.0]],
    )

    hoppings, hopping_lead_slices = (
        kwant.plotter.sys_leads_hoppings(finalized)
    )
    assert hoppings == [((0, 1), None, 0)]
    assert hopping_lead_slices == []
    starts, ends = kwant.plotter.sys_leads_hopping_pos(
        finalized,
        hoppings,
    )
    np.testing.assert_allclose(starts, [[0.0]])
    np.testing.assert_allclose(ends, [[1.0]])

    cached = kwant.system.PrecalculatedLead(
        selfenergy=np.asarray([[-0.5j]])
    )
    np.testing.assert_allclose(cached.selfenergy(), [[-0.5j]])
    callback_lead = kwant.builder.SelfEnergyLead(
        lambda energy, args=(): np.asarray([[-0.5j]]),
        [lattice(0)],
        (),
    )
    assert kwant.system.is_selfenergy_lead(callback_lead)

    from kwant.solvers import common, default, mumps, sparse

    assert issubclass(sparse.Solver, common.SparseSolver)
    assert default.smatrix is kwant.smatrix
    assert kwant.plot is kwant.plotter.plot
    previous = mumps.options(nrhs=2, sparse_rhs=True)
    assert previous["nrhs"] == 6
    restored = mumps.reset_options()
    assert restored["nrhs"] == 2


def test_low_level_lapack_entry_points() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    matrix = np.asarray([[1.0, 2.0], [0.0, 3.0]])
    prepared = kwant.linalg.lapack.prepare_for_lapack(False, matrix)
    assert prepared.flags["F_CONTIGUOUS"]

    form = np.asfortranarray(np.diag([1.0 + 0j, 2.0 + 0j]))
    vectors = np.asfortranarray(np.eye(2, dtype=complex))
    select = np.asarray([False, True], dtype=np.int32)
    reordered, reordered_vectors, eigenvalues = (
        kwant.linalg.lapack.trsen(select, form, vectors)
    )
    np.testing.assert_allclose(eigenvalues, [2.0, 1.0])
    np.testing.assert_allclose(
        reordered_vectors @ reordered @ reordered_vectors.conj().T,
        form,
    )
    right = kwant.linalg.lapack.trevc(
        reordered,
        reordered_vectors,
        None,
    )
    np.testing.assert_allclose(
        form @ right,
        right * eigenvalues,
    )


def test_magnetic_gauge_flux_and_normalized_site_constructor() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    lattice = kwant.lattice.square(norbs=1)
    sites = [
        kwant.builder.Site(lattice, np.asarray(tag), True)
        for tag in ((0, 0), (1, 0), (1, 1), (0, 1))
    ]
    system = kwant.Builder()
    system[sites] = 0.0
    for first, second in zip(sites, [*sites[1:], sites[0]], strict=True):
        system[first, second] = -1.0

    phase = kwant.physics.magnetic_gauge(system.finalized())(0.2)
    loop_phase = np.prod(
        [
            phase(first, second)
            for first, second in zip(
                sites,
                [*sites[1:], sites[0]],
                strict=True,
            )
        ]
    )
    assert loop_phase == pytest.approx(np.exp(0.2j * np.pi))


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
    assert green.data.shape == (2, 2)
    assert green.submatrix(1, 0).shape == (1, 1)
    assert green.submatrix(1, 0)[0, 0] == pytest.approx(
        green.data[1, 0]
    )
    assert all(info.shape == (1, 1) for info in green.lead_info)
    assert green.transmission(1, 0) == pytest.approx(1.0)

    selected = kwant.greens_function(
        finalized,
        energy=0.0,
        out_leads=[1],
        in_leads=[0],
    )
    assert selected.data.shape == (1, 1)
    np.testing.assert_allclose(
        selected.data,
        green.submatrix(1, 0),
    )
    density = kwant.ldos(finalized, energy=0.0)
    assert density.shape == (3,)
    assert np.all(density > 0)


def test_steady_state_solver_uses_the_rust_open_system_core(
    monkeypatch,
) -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    lattice = kwant.lattice.chain(norbs=1)
    system = kwant.Builder()
    for index in range(4):
        system[lattice(index)] = 0.1 * index
    system[lattice.neighbors()] = -1.0
    lead = kwant.Builder(
        kwant.TranslationalSymmetry(lattice.vec((-1,)))
    )
    lead[lattice(0)] = 0.0
    lead[lattice.neighbors()] = -1.0
    system.attach_lead(lead)
    system.attach_lead(lead.reversed())
    finalized = system.finalized()

    called = []
    original = kwant.solvers._core.sparse_open_system

    def traced(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        kwant.solvers._core,
        "sparse_open_system",
        traced,
    )

    def disabled(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "steady-state transport used NumPy linear algebra"
        )

    for name in ("inv", "eigh", "eigvalsh", "pinv"):
        monkeypatch.setattr(np.linalg, name, disabled)

    scattering = kwant.smatrix(finalized, energy=0.15)
    green = kwant.greens_function(finalized, energy=0.15)
    density = kwant.ldos(finalized, energy=0.15)
    states = kwant.wave_function(finalized, energy=0.15)

    assert scattering.transmission(1, 0) > 0.0
    assert green.transmission(1, 0) == pytest.approx(
        scattering.transmission(1, 0),
        abs=1.0e-9,
    )
    assert density.shape == (4,)
    assert states(0).shape == (1, 4)
    assert len(called) == 4


def test_large_open_system_path_never_materializes_the_device_dense(
    monkeypatch,
) -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    dimension = 20_000
    lattice = kwant.lattice.chain(norbs=1)
    system = kwant.Builder()
    system[(lattice(index) for index in range(dimension))] = 0.0
    system[lattice.neighbors()] = -0.35

    def selfenergy(energy, args=()):
        del energy, args
        return np.asarray([[-0.5j]])

    system.leads.append(
        kwant.builder.SelfEnergyLead(
            selfenergy,
            [lattice(0)],
            (),
        )
    )
    system.leads.append(
        kwant.builder.SelfEnergyLead(
            selfenergy,
            [lattice(dimension - 1)],
            (),
        )
    )
    finalized = system.finalized()

    def forbidden_dense_hamiltonian(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "sparse transport materialized a dense device Hamiltonian"
        )

    monkeypatch.setattr(
        kwant.builder._core,
        "hamiltonian",
        forbidden_dense_hamiltonian,
    )
    captured = {}
    original = kwant.solvers._core.sparse_open_system

    def traced(shape, row_offsets, column_indices, values, *args):
        captured["shape"] = shape
        captured["offsets"] = len(row_offsets)
        captured["nnz"] = len(values)
        result = original(
            shape,
            row_offsets,
            column_indices,
            values,
            *args,
        )
        captured["solver_nnz"] = result.solver_nnz
        return result

    monkeypatch.setattr(
        kwant.solvers._core,
        "sparse_open_system",
        traced,
    )
    green = kwant.greens_function(finalized, energy=2.0)

    assert captured["shape"] == (dimension, dimension)
    assert captured["offsets"] == dimension + 1
    assert captured["nnz"] < 3 * dimension
    assert captured["solver_nnz"] < 6 * dimension + 4
    assert green.data.shape == (2, 2)
    assert np.all(np.isfinite(green.data))


def test_local_operators_execute_the_rust_continuity_core(monkeypatch) -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    lattice = kwant.lattice.chain(norbs=2)
    first, second = lattice(0), lattice(1)
    onsite = np.asarray([[0.4, -0.2j], [0.2j, -0.1]])
    neighbor_onsite = np.asarray([[0.3, 0.1], [0.1, -0.5]])
    hopping = np.asarray(
        [[0.5 + 0.1j, -0.2], [0.3j, 0.4 - 0.15j]]
    )
    observable = np.asarray([[1.0, 0.2j], [-0.2j, -0.4]])
    builder = kwant.Builder()
    builder[first] = onsite
    builder[second] = neighbor_onsite
    builder[first, second] = hopping
    finalized = builder.finalized()

    called = set()
    for name in (
        "local_density_operators",
        "bond_current_operators",
        "local_source_operators",
    ):
        original = getattr(kwant.operator._core, name)

        def traced(*args, _name=name, _original=original, **kwargs):
            called.add(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(kwant.operator._core, name, traced)

    density = kwant.operator.Density(
        finalized, observable, where=[first], sum=True
    )
    current = kwant.operator.Current(
        finalized, observable, where=[(first, second)], sum=True
    )
    source = kwant.operator.Source(
        finalized, observable, where=[first], sum=True
    )

    local_density = density.tocoo().toarray()
    rate = 1j * (
        finalized.hamiltonian_submatrix() @ local_density
        - local_density @ finalized.hamiltonian_submatrix()
    )
    resolved_rate = (
        current.tocoo().toarray() + source.tocoo().toarray()
    )
    np.testing.assert_allclose(resolved_rate, rate, atol=1e-13)
    assert called == {
        "local_density_operators",
        "bond_current_operators",
        "local_source_operators",
    }


def test_kpm_trace_sum_uses_the_rust_recurrence(monkeypatch) -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)

    def disabled_python_eigensolver(*args, **kwargs):
        raise AssertionError("the compatibility path used a Python eigensolver")

    monkeypatch.setattr(np.linalg, "eigh", disabled_python_eigensolver)
    monkeypatch.setattr(np.linalg, "eigvalsh", disabled_python_eigensolver)

    hamiltonian = np.asarray([[0.2, -1.0], [-1.0, -0.1]])
    density = kwant.kpm.SpectralDensity(
        hamiltonian,
        vector_factory=kwant.kpm.LocalVectors(hamiltonian),
        num_vectors=None,
        num_moments=96,
        mean=False,
    )
    integrated = density.integrate()
    np.testing.assert_allclose(integrated, [1.0, 1.0], atol=1e-10)


def test_kpm_sparse_path_does_not_materialize_dense_matrices() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)

    class NoDenseCsr(scipy.sparse.csr_matrix):
        def toarray(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("KPM attempted dense materialization")

    dimension = 20_000
    hamiltonian = NoDenseCsr(
        scipy.sparse.diags(
            (
                -np.ones(dimension - 1),
                np.zeros(dimension),
                -np.ones(dimension - 1),
            ),
            (-1, 0, 1),
            format="csr",
        )
    )
    identity = NoDenseCsr(scipy.sparse.identity(dimension, format="csr"))
    density = kwant.kpm.SpectralDensity(
        hamiltonian,
        operator=identity,
        num_vectors=1,
        num_moments=8,
        bounds=(-2.0, 2.0),
        rng=7,
    )
    assert density._rescaled_operator.nnz == 2 * (dimension - 1)
    np.testing.assert_allclose(density.integrate(), dimension, atol=1e-8)


def test_wraparound_recovers_the_cosine_chain() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    lattice = kwant.lattice.chain(norbs=1)
    periodic = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    periodic[lattice(0)] = 0.0
    periodic[lattice(0), lattice(1)] = -1.3
    wrapped = kwant.wraparound.wraparound(periodic).finalized()

    for momentum in (-2.7, -0.4, 0.0, 1.2, 3.0):
        hamiltonian = wrapped.hamiltonian_submatrix(
            params={"k_x": momentum}
        )
        np.testing.assert_allclose(
            hamiltonian,
            [[-2.6 * np.cos(momentum)]],
            atol=1e-13,
        )


def test_random_matrix_symmetries_generalize_to_larger_dimensions() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    dimension = 12
    for index, symmetry in enumerate(kwant.rmt.sym_list):
        hamiltonian = kwant.rmt.gaussian(
            dimension,
            symmetry,
            v=1.7,
            rng=700 + index,
        )
        np.testing.assert_allclose(
            hamiltonian,
            hamiltonian.conj().T,
            atol=1e-12,
        )
        if kwant.rmt.t(symmetry):
            operator = np.asarray(kwant.rmt.h_t_matrix[symmetry])
            operator = np.kron(
                np.eye(dimension // len(operator)),
                operator,
            )
            np.testing.assert_allclose(
                hamiltonian,
                operator @ hamiltonian.conj() @ operator,
                atol=1e-12,
            )
        if kwant.rmt.p(symmetry):
            operator = np.asarray(kwant.rmt.h_p_matrix[symmetry])
            operator = np.kron(
                np.eye(dimension // len(operator)),
                operator,
            )
            np.testing.assert_allclose(
                hamiltonian,
                -(operator @ hamiltonian.conj() @ operator),
                atol=1e-12,
            )


def test_circular_ensembles_honor_topological_sectors() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    for sector in (-1, 1):
        matrix = kwant.rmt.circular(10, "D", charge=sector, rng=41)
        assert np.sign(np.linalg.det(matrix).real) == sector

    aiii = kwant.rmt.circular(10, "AIII", charge=4, rng=42)
    aiii_eigenvalues = np.linalg.eigvalsh(aiii)
    assert np.count_nonzero(aiii_eigenvalues < 0) == 4

    cii = kwant.rmt.circular(12, "CII", charge=2, rng=43)
    cii_eigenvalues = np.linalg.eigvalsh(cii)
    assert np.count_nonzero(cii_eigenvalues < 0) == 4

    with pytest.raises(ValueError):
        kwant.rmt.circular(7, "AII", rng=44)
    with pytest.raises(ValueError):
        kwant.rmt.circular(10, "AIII", charge=11, rng=45)


def test_discrete_symmetry_generalizes_to_three_conservation_blocks() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    from scipy import sparse

    projectors = [
        sparse.csr_matrix(np.eye(3)[:, [column]])
        for column in range(3)
    ]
    symmetry = kwant.physics.DiscreteSymmetry(projectors=projectors)
    assert symmetry.validate(np.diag([1.0, 2.0, 3.0])) == []
    assert symmetry.validate(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
    ) == ["Conservation law"]


def test_particle_hole_basis_executes_without_python_factorizations(monkeypatch) -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    import scipy.linalg

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Python factorizations must not construct symmetry bases")

    monkeypatch.setattr(scipy.linalg, "schur", forbidden)
    monkeypatch.setattr(scipy.linalg, "qr", forbidden)

    mixed = np.asarray(
        [
            [1 / np.sqrt(2), 1j / np.sqrt(2)],
            [1j / np.sqrt(2), 1 / np.sqrt(2)],
        ],
        dtype=complex,
    )
    adapted, ordering = kwant.physics.phs_symmetrization(mixed, np.eye(2))
    np.testing.assert_allclose(adapted, adapted.conj(), atol=1e-8)
    np.testing.assert_allclose(adapted.conj().T @ adapted, np.eye(2), atol=1e-8)
    np.testing.assert_array_equal(ordering, [0, 0])

    particle_hole = np.asarray([[0, 1], [-1, 0]], dtype=complex)
    adapted, ordering = kwant.physics.phs_symmetrization(
        np.eye(2),
        particle_hole,
    )
    np.testing.assert_allclose(
        adapted[:, 1],
        particle_hole @ adapted[:, 0].conj(),
        atol=1e-8,
    )
    np.testing.assert_array_equal(ordering, [0, 1])


def test_lattice_reduction_exposes_voronoi_geometry() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    lll = kwant.linalg.lll

    basis = np.eye(2)
    full_neighbors = lll.voronoi(basis)
    assert full_neighbors.shape == (6, 2)

    reduced_neighbors = {
        tuple(vector)
        for vector in lll.voronoi(basis, reduced=True)
    }
    assert reduced_neighbors == {
        (-1, 0),
        (0, -1),
        (1, 0),
        (0, 1),
    }

    closest = lll.cvp([0.5, 0.5], basis, group_by_length=True)
    assert {tuple(vector) for vector in closest} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }


def test_compressed_graph_generalizes_to_parallel_and_dangling_edges() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    import pickle

    node_count = 257
    edges = [
        (node, (37 * node + offset) % node_count)
        for node in range(node_count)
        for offset in range(4)
    ]
    edges.extend([(17, 19), (17, 19), (41, -3), (-7, 43)])

    graph = kwant.graph.Graph(allow_negative_nodes=True)
    graph.num_nodes = node_count
    assert graph.add_edges(edges) == 0
    compressed = graph.compressed(twoway=True, edge_nr_translation=True)

    assert compressed.num_nodes == node_count
    assert compressed.num_edges == len(edges)
    assert tuple(compressed.all_edge_ids(17, 19))
    assert compressed.has_edge(41, -3)
    assert compressed.has_edge(-7, 43)
    assert compressed.tail(compressed.edge_id(len(edges) - 1)) is None
    with pytest.raises(kwant.graph.EdgeDoesNotExistError):
        compressed.head(-1)

    restored = pickle.loads(pickle.dumps(compressed))
    assert restored.__getstate__() == compressed.__getstate__()
    assert list(restored) == list(compressed)
    assert tuple(restored.in_neighbors(43)) == tuple(compressed.in_neighbors(43))


def test_dense_decompositions_generalize_to_nonnormal_matrix_pencils() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    rng = np.random.default_rng(1905)
    dimension = 12
    left = (
        rng.normal(size=(dimension, dimension))
        + 1j * rng.normal(size=(dimension, dimension))
    )
    right = (
        3.0 * np.eye(dimension)
        + 0.15 * rng.normal(size=(dimension, dimension))
        + 0.15j * rng.normal(size=(dimension, dimension))
    )

    form, vectors, eigenvalues = kwant.linalg.schur(left)
    np.testing.assert_allclose(
        vectors @ form @ vectors.conj().T,
        left,
        rtol=1e-11,
        atol=1e-11,
    )
    selection = np.array(
        [index % 3 == 1 for index in range(dimension)],
        dtype=bool,
    )
    reordered, reordered_vectors, reordered_eigenvalues = (
        kwant.linalg.order_schur(selection, form, vectors)
    )
    np.testing.assert_allclose(
        reordered_vectors @ reordered @ reordered_vectors.conj().T,
        left,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.sort_complex(reordered_eigenvalues),
        np.sort_complex(eigenvalues),
        rtol=1e-11,
        atol=1e-11,
    )
    selected_left, selected_right = kwant.linalg.evecs_from_schur(
        form,
        vectors,
        selection,
        left=True,
        right=True,
    )
    np.testing.assert_allclose(
        left @ selected_right,
        selected_right @ np.diag(eigenvalues[selection]),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        selected_left.conj().T @ left,
        np.diag(eigenvalues[selection]) @ selected_left.conj().T,
        rtol=1e-10,
        atol=1e-10,
    )

    s, t, q, z, alpha, beta = kwant.linalg.gen_schur(left, right)
    np.testing.assert_allclose(q @ s @ z.conj().T, left, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(q @ t @ z.conj().T, right, rtol=1e-11, atol=1e-11)
    generalized_left, generalized_right = kwant.linalg.evecs_from_gen_schur(
        s,
        t,
        q,
        z,
        selection,
        left=True,
        right=True,
    )
    np.testing.assert_allclose(
        left @ generalized_right @ np.diag(beta[selection]),
        right @ generalized_right @ np.diag(alpha[selection]),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.diag(beta[selection]) @ generalized_left.conj().T @ left,
        np.diag(alpha[selection]) @ generalized_left.conj().T @ right,
        rtol=1e-9,
        atol=1e-9,
    )

    form_only = kwant.linalg.schur(left, calc_q=False, calc_ev=False)
    assert len(form_only) == 1
    generalized_forms = kwant.linalg.gen_schur(
        left,
        right,
        calc_q=False,
        calc_z=False,
        calc_ev=False,
    )
    assert len(generalized_forms) == 2


def test_periodic_bands_execute_without_python_eigensolvers(monkeypatch) -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    lattice = kwant.lattice.chain(norbs=1)
    lead = kwant.Builder(kwant.TranslationalSymmetry(lattice.vec((-1,))))
    lead[lattice(0)] = 0.3
    lead[lattice(0), lattice(1)] = -1.2
    bands = kwant.physics.Bands(lead.finalized())

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Python eigensolver must not implement Bands")

    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)

    momentum = 0.7
    energy, velocity, curvature, eigenvectors = bands(
        momentum,
        derivative_order=2,
        return_eigenvectors=True,
    )
    np.testing.assert_allclose(energy, [0.3 - 2.4 * np.cos(momentum)])
    np.testing.assert_allclose(velocity, [2.4 * np.sin(momentum)])
    np.testing.assert_allclose(curvature, [2.4 * np.cos(momentum)])
    assert eigenvectors.shape == (1, 1)


def test_propagating_modes_execute_without_scipy_eigensolvers(monkeypatch) -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    import scipy.linalg

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("SciPy eigensolver must not implement lead modes")

    monkeypatch.setattr(scipy.linalg, "eig", forbidden)
    onsite = np.array([[0.3]])
    hopping = np.array([[0.7]])
    propagating, stabilized = kwant.physics.modes(onsite, hopping)

    momentum = np.arccos(-0.3 / 1.4)
    velocity = 1.4 * np.sin(momentum)
    np.testing.assert_allclose(propagating.velocities, [-velocity, velocity])
    np.testing.assert_allclose(propagating.momenta, [momentum, -momentum])
    assert stabilized.nmodes == 1

    symmetric, symmetric_stabilized = kwant.physics.modes(
        onsite,
        hopping,
        time_reversal=np.eye(1),
    )
    np.testing.assert_allclose(
        symmetric.wave_functions[:, 1],
        symmetric.wave_functions[:, 0].conj(),
    )
    assert symmetric_stabilized.nmodes == 1


def test_projected_modes_execute_in_rust_subspaces(monkeypatch) -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    import scipy.linalg
    from scipy import sparse

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("SciPy eigensolver must not implement projected modes")

    monkeypatch.setattr(scipy.linalg, "eig", forbidden)
    scale = 1 / np.sqrt(2)
    projectors = [
        sparse.csr_matrix([[scale], [1j * scale]]),
        sparse.csr_matrix([[1j * scale], [scale]]),
    ]
    propagating, stabilized = kwant.physics.modes(
        0.3 * np.eye(2),
        0.7 * np.eye(2),
        projectors=projectors,
    )

    assert propagating.block_nmodes == [1, 1]
    assert stabilized.nmodes == 2
    np.testing.assert_allclose(stabilized.vecs[1, [0, 2]], 0)
    np.testing.assert_allclose(stabilized.vecs[0, [1, 3]], 0)
    assert np.all(np.abs(stabilized.vecs[0, [0, 2]]) > 0)
    assert np.all(np.abs(stabilized.vecs[1, [1, 3]]) > 0)


def test_lead_selfenergy_executes_without_python_eigensolvers(monkeypatch) -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Python eigensolver must not implement self-energy")

    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    onsite = np.array([[0.3]])
    hopping = np.array([[0.7]])
    expected = -0.15 - 0.5j * np.sqrt(1.87)
    np.testing.assert_allclose(
        kwant.physics.selfenergy(onsite, hopping),
        [[expected]],
        atol=1e-9,
    )

    width, amplitude, energy = 5, 0.78, 1.3
    cell = (4 * amplitude - energy) * np.eye(width)
    cell += np.diag(np.full(width - 1, -amplitude), 1)
    cell += np.diag(np.full(width - 1, -amplitude), -1)
    analytic = kwant.physics.square_selfenergy(width, amplitude, energy)
    numerical = kwant.physics.selfenergy(cell, -amplitude * np.eye(width))
    np.testing.assert_allclose(analytic, numerical, atol=1e-9)


def test_digest_matches_kwant_15_byte_contract() -> None:
    kwant = require_compat_module("kwant", ISSUE_URL)
    assert kwant.digest.uniform("abc") == 0.4944136595586759
    assert kwant.digest.gauss("abc") == -3.3392115425542803
    pair = kwant.digest.uniform2(np.array([1, 2], dtype=np.int32), salt=b"x")
    assert len(pair) == 2
    assert all(0.0 <= value < 1.0 for value in pair)
