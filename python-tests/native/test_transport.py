import numpy as np

from thouless import transport


def test_lead_modes_self_energy_and_ballistic_scattering():
    cell = np.array([[0.0]], dtype=np.complex128)
    hopping = np.array([[-1.0]], dtype=np.complex128)
    self_energy = transport.lead_self_energy(
        cell,
        hopping,
        0.0,
    )
    np.testing.assert_allclose(self_energy, [[-1.0j]], atol=1.0e-5)

    modes = transport.propagating_modes(cell, hopping)
    assert modes.incoming_count == 1
    np.testing.assert_allclose(np.sort(modes.velocities), [-2.0, 2.0], atol=1.0e-10)

    lead = transport.Lead(cell, hopping, [[-1.0]])
    result = transport.solve(
        [[0.0]],
        [lead, lead],
        0.0,
    )
    np.testing.assert_allclose(result.transmissions[0, 1], 1.0, atol=1.0e-6)
    np.testing.assert_allclose(result.transmissions[1, 0], 1.0, atol=1.0e-6)


def test_partition_noise_is_zero_for_perfect_reflection_channels():
    assert abs(transport.partition_shot_noise(np.eye(2))) <= 1.0e-12
