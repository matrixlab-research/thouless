import numpy as np
import scipy.sparse

from thouless import continuum, graph, kpm, linalg, observables, random, symmetry, visualization


def test_kpm_reconstruction_uses_complex_binary64_arrays():
    scaled = kpm.rescale(np.diag([-1.0, 1.0]))
    vectors = kpm.chebyshev_vectors(scaled.matrix, [[1.0, 0.0]], 8)
    moments = kpm.scalar_moments([[1.0, 0.0]], vectors)
    reconstruction = kpm.reconstruct(
        moments,
        scaled.half_width,
        scaled.center,
    )
    assert reconstruction.densities.dtype == np.complex128
    assert reconstruction.energies.ndim == 1


def test_observables_preserve_local_operator_semantics():
    density = observables.densities([1, 1], [(0, [[1.0]]), (1, [[2.0]])])
    np.testing.assert_allclose(density.matrix(), np.diag([1.0, 2.0]), atol=1.0e-12)
    np.testing.assert_allclose(density.apply([3.0, 4.0]), [3.0, 8.0], atol=1.0e-12)
    np.testing.assert_allclose(
        observables.pauli_coefficients([[1.0, 0.0], [0.0, -1.0]]),
        [0.0, 0.0, 0.0, 1.0],
        atol=1.0e-12,
    )


def test_continuum_and_regular_field_workflows_are_native():
    terms = continuum.finite_difference_stencil(1, [(0, 0, 2)])
    assert terms
    coefficient = continuum.landau_ladder_coefficient([1, -1], 2, 1.5)
    assert np.isfinite(coefficient)

    field = visualization.interpolate_density(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0, 2.0],
        [([0.0, 0.0], [1.0, 0.0])],
        absolute_width=0.2,
    )
    assert field.values.size == np.prod(field.shape) * field.components


def test_discrete_symmetry_and_particle_hole_basis():
    constraint = symmetry.DiscreteSymmetry(chiral=np.diag([1.0, -1.0]))
    assert constraint.validate([[0.0, 1.0], [1.0, 0.0]]) == ()

    vectors, ordering = symmetry.particle_hole_basis(
        np.eye(2, dtype=np.complex128),
        np.eye(2, dtype=np.complex128),
    )
    np.testing.assert_allclose(vectors.conj() @ vectors.T, np.eye(2), atol=1.0e-12)
    # Square-plus-one particle-hole symmetry labels both self-conjugate
    # vectors with the same stable partner block.
    np.testing.assert_array_equal(ordering, [0, 0])


def test_random_graph_and_dense_sparse_linear_algebra():
    assert 0.0 <= random.uniform(b"model", b"salt") < 1.0
    first, second = random.uniform_pair(b"model", b"salt")
    assert 0.0 <= first < 1.0
    assert 0.0 <= second < 1.0
    assert np.isfinite(random.gaussian(b"model", b"salt"))

    gaussian = random.gaussian_matrix(
        2,
        "A",
        1.0,
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    )
    np.testing.assert_allclose(gaussian, gaussian.conj().T, atol=1.0e-12)

    builder = graph.GraphBuilder()
    builder.node_count = 3
    builder.add_edges([(0, 1), (1, 2)])
    compressed = builder.compress(reverse_index=True)
    np.testing.assert_array_equal(compressed.outgoing_neighbors(1), [2])
    np.testing.assert_array_equal(compressed.incoming_neighbors(1), [0])

    matrix = np.array([[1.0, 2.0j], [0.0, 3.0]], dtype=np.complex128)
    decomposition = linalg.schur(matrix)
    np.testing.assert_allclose(
        decomposition.vectors @ decomposition.form @ decomposition.vectors.conj().T,
        matrix,
        atol=1.0e-10,
    )

    sparse = scipy.sparse.csr_matrix(
        np.array([[2.0, 1.0j], [-1.0j, 3.0]], dtype=np.complex128)
    )
    factorization = linalg.SparseLU(sparse)
    solution = factorization.solve([[1.0], [2.0]])
    np.testing.assert_allclose(sparse @ solution, [[1.0], [2.0]], atol=1.0e-10)
    complement = linalg.sparse_schur_complement(sparse, [0])
    np.testing.assert_allclose(complement, [[2.0 - 1.0 / 3.0]], atol=1.0e-10)
