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
