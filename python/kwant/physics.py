"""Lead dispersion and propagating modes for periodic tight-binding systems."""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg
from scipy import sparse as scipy_sparse


class DiscreteSymmetry:
    """Sparse projectors and operators describing discrete symmetries."""

    def __init__(
        self,
        projectors=None,
        time_reversal=None,
        particle_hole=None,
        chiral=None,
    ):
        self.projectors = (
            None
            if projectors is None
            else tuple(
                scipy_sparse.csr_matrix(projector)
                for projector in projectors
            )
        )
        self.time_reversal = (
            None
            if time_reversal is None
            else scipy_sparse.csr_matrix(time_reversal)
        )
        self.particle_hole = (
            None
            if particle_hole is None
            else scipy_sparse.csr_matrix(particle_hole)
        )
        self.chiral = (
            None if chiral is None else scipy_sparse.csr_matrix(chiral)
        )

    def __getitem__(self, item):
        return (
            self.projectors,
            self.time_reversal,
            self.particle_hole,
            self.chiral,
        )[item]


class Bands:
    """Bloch energies and momentum derivatives of a periodic lead."""

    _crossover_size = 8

    def __init__(self, system, args=(), *, params=None):
        self.ham = np.asarray(
            system.cell_hamiltonian(args=args, params=params),
            dtype=complex,
        )
        if (
            self.ham.ndim != 2
            or self.ham.shape[0] != self.ham.shape[1]
            or not np.allclose(self.ham, self.ham.conj().T)
        ):
            raise ValueError("The cell Hamiltonian is not Hermitian.")
        inter_cell = np.asarray(
            system.inter_cell_hopping(args=args, params=params),
            dtype=complex,
        )
        self.hop = np.zeros_like(self.ham, dtype=complex)
        self.hop[:, : inter_cell.shape[1]] = inter_cell

    def __call__(
        self,
        momentum,
        derivative_order=0,
        return_eigenvectors=False,
    ):
        if derivative_order > 2:
            raise NotImplementedError(
                "Band derivatives are implemented only through second order"
            )
        phase = complex(
            math.cos(float(momentum)), -math.sin(float(momentum))
        )
        phased_hopping = self.hop * phase
        hamiltonian = (
            self.ham
            + phased_hopping
            + phased_hopping.conj().T
        )
        need_vectors = return_eigenvectors or derivative_order > 0
        if need_vectors:
            energies, eigenvectors = np.linalg.eigh(hamiltonian)
        else:
            energies = np.linalg.eigvalsh(hamiltonian)
            eigenvectors = None
        output = [energies.real]
        if derivative_order:
            first_derivative = 1j * (
                -phased_hopping + phased_hopping.conj().T
            )
            transformed_first = (
                eigenvectors.conj().T
                @ first_derivative
                @ eigenvectors
            )
            output.append(np.diag(transformed_first).real)
        if derivative_order == 2:
            second_derivative = -(
                phased_hopping + phased_hopping.conj().T
            )
            transformed_second = (
                eigenvectors.conj().T
                @ second_derivative
                @ eigenvectors
            )
            energy_difference = energies[:, None] - energies[None, :]
            inverse_difference = np.zeros_like(energy_difference)
            np.divide(
                1.0,
                energy_difference,
                out=inverse_difference,
                where=energy_difference != 0,
            )
            output.append(
                (
                    np.diag(transformed_second)
                    - 2
                    * np.sum(
                        inverse_difference
                        * np.abs(transformed_first) ** 2,
                        axis=0,
                    )
                ).real
            )
        if return_eigenvectors:
            output.append(eigenvectors)
        return output[0] if len(output) == 1 else tuple(output)


class PropagatingModes:
    def __init__(self, wave_functions, velocities, momenta):
        self.wave_functions = np.asarray(wave_functions, dtype=complex)
        self.velocities = np.asarray(velocities, dtype=float)
        self.momenta = np.asarray(momenta, dtype=float)
        self.block_nmodes = [len(self.momenta) // 2]


class StabilizedModes:
    def __init__(
        self,
        vecs,
        vecslmbdainv,
        nmodes,
        sqrt_hop=None,
        selfenergy=None,
    ):
        self.vecs = np.asarray(vecs, dtype=complex)
        self.vecslmbdainv = np.asarray(vecslmbdainv, dtype=complex)
        self.nmodes = int(nmodes)
        self.sqrt_hop = sqrt_hop
        self._selfenergy = selfenergy

    def selfenergy(self):
        if self._selfenergy is None:
            raise ValueError(
                "Self-energy is unavailable for these stabilized modes"
            )
        return np.asarray(self._selfenergy, dtype=complex)


def two_terminal_shotnoise(scattering_matrix):
    """Return zero-temperature shot noise for a two-lead conductor."""
    from .solvers import SMatrix

    if not isinstance(scattering_matrix, SMatrix):
        raise NotImplementedError(
            "Green-function shot-noise evaluation is not implemented"
        )
    if len(scattering_matrix.lead_info) != 2:
        raise ValueError("Shot noise requires exactly two leads")
    block = scattering_matrix.submatrix(
        scattering_matrix.out_leads[0],
        scattering_matrix.in_leads[0],
    )
    probabilities = block @ block.conj().T
    return float(
        np.trace(probabilities - probabilities @ probabilities).real
    )


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


__all__ = [
    "Bands",
    "DiscreteSymmetry",
    "PropagatingModes",
    "StabilizedModes",
    "modes",
    "two_terminal_shotnoise",
]
