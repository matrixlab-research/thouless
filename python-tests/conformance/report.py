"""Emit public cross-language scientific conformance metrics."""

from __future__ import annotations

import numpy as np

import thouless
from thouless import geometry, observables, topology, transport


def ssh_model() -> thouless.Model:
    builder = thouless.ModelBuilder(thouless.Lattice([[1.0]], [0]))
    first = builder.add_orbital("a", [0.0])
    second = builder.add_orbital("b", [0.5])
    builder.add_hopping(first, second, [0], 0.6)
    builder.add_hopping(first, second, [1], 1.0)
    return builder.build()


def qwz_model() -> thouless.Model:
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    builder = thouless.ModelBuilder(thouless.Lattice(np.eye(2), [0, 1]))
    orbital = builder.add_orbital("spinor", [0.0, 0.0], degrees_of_freedom=2)
    builder.set_onsite_block(orbital, -sigma_z)
    builder.add_hopping_block(orbital, orbital, [1, 0], 0.5 * sigma_z - 0.5j * sigma_x)
    builder.add_hopping_block(orbital, orbital, [0, 1], 0.5 * sigma_z - 0.5j * sigma_y)
    return builder.build()


def main() -> None:
    ssh = ssh_model()
    zone_edge = ssh.eigensystem([0.5]).eigenvalues
    frames = [
        ssh.eigensystem([sample / 65.0]).eigenvectors[:, 0][None, :]
        for sample in range(65)
    ]
    frames.append(frames[0] * np.array([[1.0, -1.0]]))
    polarization = (topology.wilson_phase(np.asarray(frames)) / (2.0 * np.pi)) % 1.0

    chern = abs(topology.chern_numbers(qwz_model(), [31, 31], (0, 1), [0]).values[0])

    vacancy = geometry.finite_geometry(ssh, [([0], 0), ([2], 1)])
    projected = observables.project_diagonal(np.eye(2), [1.0, 2.0])

    cell = np.array([[0.0]], dtype=np.complex128)
    hopping = np.array([[-1.0]], dtype=np.complex128)
    lead = transport.Lead(cell, hopping, hopping)
    scattering = transport.solve(cell, [lead, lead], 0.0)

    inverse_sqrt_two = 1.0 / np.sqrt(2.0)
    wilson_frames = np.array(
        [
            [[inverse_sqrt_two, inverse_sqrt_two]],
            [[inverse_sqrt_two, 1j * inverse_sqrt_two]],
            [[inverse_sqrt_two, inverse_sqrt_two]],
        ],
        dtype=np.complex128,
    )
    phase = topology.wilson_phase(wilson_frames)
    transformed = wilson_frames.copy()
    transformed[1] *= np.exp(0.37j)
    gauge_delta = abs(topology.wilson_phase(transformed) - phase)

    invalid_shape = 0.0
    try:
        ssh.hamiltonian([0.0, 0.5])
    except thouless.ThoulessError:
        invalid_shape = 1.0

    metrics = {
        "ssh_gap": float(zone_edge[1] - zone_edge[0]),
        "ssh_polarization": float(polarization),
        "chern_absolute": float(chern),
        "vacancy_states": float(vacancy.state_count),
        "vacancy_observable_trace": float(np.trace(projected).real),
        "ballistic_transmission": float(scattering.transmissions[1, 0]),
        "wilson_gauge_delta": float(gauge_delta),
        "invalid_shape_error": invalid_shape,
    }
    for name in sorted(metrics):
        print(f"{name}={metrics[name]:.17e}")


if __name__ == "__main__":
    main()
