"""Lead dispersion and propagating modes for periodic tight-binding systems."""

from __future__ import annotations

import numpy as np
import scipy.linalg


class PropagatingModes:
    def __init__(self, wave_functions, velocities, momenta):
        self.wave_functions = np.asarray(wave_functions, dtype=complex)
        self.velocities = np.asarray(velocities, dtype=float)
        self.momenta = np.asarray(momenta, dtype=float)
        self.block_nmodes = [len(self.momenta) // 2]


class StabilizedModes:
    def __init__(self, vecs, vecslmbdainv, nmodes, sqrt_hop=None):
        self.vecs = np.asarray(vecs, dtype=complex)
        self.vecslmbdainv = np.asarray(vecslmbdainv, dtype=complex)
        self.nmodes = int(nmodes)
        self.sqrt_hop = sqrt_hop


def modes(h_cell, h_hop, tol=1e6, stabilization=None, *, discrete_symmetry=None):
    """Solve Bloch modes of a nearest-cell periodic lead."""
    h_cell = np.asarray(h_cell, dtype=complex)
    h_hop = np.asarray(h_hop, dtype=complex)
    if h_cell.ndim != 2 or h_cell.shape[0] != h_cell.shape[1]:
        raise ValueError("Cell Hamiltonian must be square")
    size = h_cell.shape[0]
    if h_hop.shape != (size, size):
        raise ValueError("Inter-cell hopping must be square")

    identity = np.eye(size, dtype=complex)
    zero = np.zeros_like(identity)
    first = np.block([[-h_cell, -h_hop], [identity, zero]])
    second = np.block([[h_hop.conj().T, zero], [zero, identity]])
    eigenvalues, eigenvectors = scipy.linalg.eig(first, second)

    candidates = []
    for eigenvalue, vector in zip(eigenvalues, eigenvectors.T):
        if not np.isfinite(eigenvalue) or abs(abs(eigenvalue) - 1) > 1e-7:
            continue
        wave = vector[:size]
        norm = np.linalg.norm(wave)
        if norm == 0:
            continue
        wave = wave / norm
        velocity = float(
            np.real(
                1j
                * np.vdot(
                    wave,
                    (
                        h_hop.conj().T * eigenvalue
                        - h_hop / eigenvalue
                    )
                    @ wave,
                )
            )
        )
        if abs(velocity) <= 1e-10:
            continue
        wave /= np.sqrt(abs(velocity))
        candidates.append((velocity, float(np.angle(eigenvalue)), wave))

    candidates.sort(key=lambda item: (item[0] > 0, item[1]))
    wave_functions = (
        np.column_stack([item[2] for item in candidates])
        if candidates
        else np.empty((size, 0), dtype=complex)
    )
    velocities = [item[0] for item in candidates]
    momenta = [item[1] for item in candidates]
    propagating = PropagatingModes(wave_functions, velocities, momenta)
    stabilized = StabilizedModes(
        wave_functions,
        wave_functions,
        sum(velocity < 0 for velocity in velocities),
    )
    return propagating, stabilized


__all__ = ["PropagatingModes", "StabilizedModes", "modes"]
