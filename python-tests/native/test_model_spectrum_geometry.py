import numpy as np

import thouless
from thouless import geometry, spectrum


def chain_model() -> thouless.Model:
    lattice = thouless.Lattice([[1.0]], [0])
    builder = thouless.ModelBuilder(lattice)
    orbital = builder.add_orbital("s", [0.0])
    builder.set_onsite(orbital, 0.25)
    builder.add_hopping(orbital, orbital, [1], -1.0)
    return builder.build()


def test_rust_owned_model_preserves_complex_array_and_error_semantics():
    model = chain_model()
    assert model.state_count == 1
    np.testing.assert_allclose(model.hamiltonian([0.0]), [[-1.75]], atol=1.0e-12)
    np.testing.assert_allclose(
        model.eigensystem([0.5]).eigenvalues,
        [2.25],
        atol=1.0e-12,
    )
    values = np.array([[1.0 + 2.0j]], dtype=np.complex128)
    eigensystem = spectrum.hermitian_eigensystem(values + values.conj().T)
    assert eigensystem.eigenvectors.dtype == np.complex128

    with np.testing.assert_raises(thouless.InvalidInputError):
        model.hamiltonian([0.0, 0.5])


def test_band_structure_and_periodic_lead_bands_use_native_results():
    model = chain_model()
    path = np.linspace(0.0, 0.5, 11)[:, None]
    bands = model.band_structure(path)
    assert len(bands) == 11
    np.testing.assert_allclose(bands[0].eigenvalues, [-1.75], atol=1.0e-12)
    np.testing.assert_allclose(bands[-1].eigenvalues, [2.25], atol=1.0e-12)

    lead = spectrum.PeriodicBands([[0.0]], [[-1.0]])
    evaluation = lead.evaluate(0.25, derivative_order=2, eigenvectors=True)
    np.testing.assert_allclose(
        evaluation.energies,
        [-2.0 * np.cos(0.25)],
        atol=1.0e-12,
    )
    assert evaluation.first_derivatives is not None
    assert evaluation.second_derivatives is not None
    assert evaluation.eigenvectors is not None
    np.testing.assert_allclose(
        evaluation.first_derivatives,
        [2.0 * np.sin(0.25)],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        evaluation.second_derivatives,
        [2.0 * np.cos(0.25)],
        atol=1.0e-12,
    )


def test_finite_arbitrary_and_supercell_geometry_share_the_model_core():
    model = chain_model()
    finite = geometry.finite_cluster(model, [[0], [1], [2]])
    assert finite.periodic_dimension == 0
    assert finite.state_count == 3
    np.testing.assert_allclose(
        np.linalg.eigvalsh(finite.hamiltonian()),
        0.25 + np.array([-np.sqrt(2.0), 0.0, np.sqrt(2.0)]),
        atol=1.0e-12,
    )

    vacancy = geometry.finite_geometry(model, [([0], 0), ([2], 0)])
    np.testing.assert_allclose(vacancy.hamiltonian(), 0.25 * np.eye(2), atol=1.0e-12)

    doubled = geometry.supercell(model, [[2]])
    assert doubled.model.state_count == 2
    np.testing.assert_array_equal(doubled.translations, [[0], [1]])


def test_reciprocal_periodic_and_lattice_reduction_workflows():
    lattice = thouless.Lattice([[2.0, 0.0], [0.0, 1.0]], [0, 1])
    path = geometry.reciprocal_path(
        lattice,
        [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]],
        7,
    )
    assert path.points.shape == (7, 2)
    np.testing.assert_allclose(path.node_distances[-1], 1.5 * np.pi, atol=1.0e-12)

    folded = geometry.fold_terms(
        [(np.array([[1.0j]]), [1], True)],
        [0.25],
    )
    np.testing.assert_allclose(folded, [[-2.0 * np.sin(0.25)]], atol=1.0e-12)

    reduced, transform = geometry.lll_reduce([[1.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(transform @ np.array([[1.0, 1.0], [0.0, 1.0]]), reduced)
    closest = geometry.closest_lattice_vectors([0.49, 0.49], np.eye(2))
    assert closest.shape[1] == 2
    neighbors = geometry.voronoi_neighbors(np.eye(2))
    assert neighbors.shape[1] == 2
