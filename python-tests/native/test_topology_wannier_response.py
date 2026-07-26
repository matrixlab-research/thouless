import numpy as np

import thouless
from thouless import response, topology, wannier


def flat_two_band_model() -> thouless.Model:
    lattice = thouless.Lattice(np.eye(2), [0, 1])
    builder = thouless.ModelBuilder(lattice)
    orbital = builder.add_orbital(
        "spinor",
        [0.0, 0.0],
        degrees_of_freedom=2,
    )
    builder.set_onsite_block(orbital, np.diag([-1.0, 1.0]))
    return builder.build()


def test_wilson_flux_chern_and_quantum_geometry_are_gauge_covariant():
    frames = np.ones((4, 1, 1), dtype=np.complex128)
    assert abs(topology.wilson_phase(frames)) <= 1.0e-12
    np.testing.assert_allclose(topology.wilson_eigenphases(frames), [0.0], atol=1.0e-12)
    assert abs(topology.berry_flux(frames)) <= 1.0e-12

    lattice = thouless.Lattice(np.eye(2), [0, 1])
    builder = thouless.ModelBuilder(lattice)
    orbital = builder.add_orbital("s", [0.0, 0.0])
    builder.set_onsite(orbital, -1.0)
    model = builder.build()
    chern = topology.chern_numbers(model, [3, 3], (0, 1), [0])
    np.testing.assert_allclose(chern.values, [0.0], atol=1.0e-12)

    hamiltonians = np.array([[[-1.0, 0.0], [0.0, 1.0]]], dtype=np.complex128)
    derivatives = np.zeros((1, 2, 2, 2), dtype=np.complex128)
    tensor = topology.quantum_geometric_tensor(
        hamiltonians,
        derivatives,
        [0],
    )
    np.testing.assert_allclose(tensor, 0.0, atol=1.0e-12)


def test_local_marker_and_wannier_projection_use_owned_numpy_results():
    marker = topology.local_chern_marker(
        np.diag([-1.0, 1.0]),
        [[0.0, 0.0], [1.0, 0.0]],
        [0],
        1.0,
    )
    np.testing.assert_allclose(marker, 0.0, atol=1.0e-12)

    frames = np.ones((2, 1, 1), dtype=np.complex128)
    projected = wannier.project_trials(frames, [[1.0]])
    frames[:] = 0.0
    np.testing.assert_allclose(projected, 1.0, atol=1.0e-12)
    overlaps = wannier.periodic_overlaps([2], projected, [[1]], [[1.0]])
    spread = wannier.spread_decomposition(overlaps, [[1.0]], [1.0])
    np.testing.assert_allclose(spread.centers, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(spread.spreads, 0.0, atol=1.0e-12)
    transformed = wannier.inverse_bloch_transform([2], projected)
    assert transformed.shape == projected.shape


def test_intrinsic_response_is_evaluated_and_integrated_in_rust():
    model = flat_two_band_model()
    point = response.band_response(
        model,
        [0.2, 0.3],
        chemical_potential=0.0,
        temperature=0.1,
    )
    np.testing.assert_allclose(point.energies, [-1.0, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(point.group_velocities, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(point.berry_curvatures, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        response.intrinsic_curvature(
            model,
            [0.2, 0.3],
            chemical_potential=0.0,
            temperature=0.1,
        ),
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        response.integrated_intrinsic_curvature(
            model,
            [3, 4],
            [0.0, 0.5],
            chemical_potential=0.0,
            temperature=0.1,
        ),
        0.0,
        atol=1.0e-12,
    )
    assert response.occupation_weighted_curvature([point], [1.0], 0, 1) == 0.0
    assert response.berry_curvature_dipole([point], [1.0], 0, 0, 1) == 0.0
