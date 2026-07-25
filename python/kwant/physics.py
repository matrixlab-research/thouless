"""Lead dispersion and propagating modes for periodic tight-binding systems."""

from __future__ import annotations

import numpy as np
from scipy import sparse as scipy_sparse
from thouless import _core


class DiscreteSymmetry:
    """Sparse projectors and operators describing discrete symmetries."""

    def __init__(
        self,
        projectors=None,
        time_reversal=None,
        particle_hole=None,
        chiral=None,
    ):
        if projectors is not None:
            try:
                projectors = [projector.tocsr() for projector in projectors]
            except AttributeError as error:
                raise TypeError(
                    "projectors must be a sequence of sparse matrices."
                ) from error

        symmetries = [time_reversal, particle_hole, chiral]
        try:
            symmetries = [
                None if symmetry is None else symmetry.tocsr()
                for symmetry in symmetries
            ]
        except AttributeError as error:
            raise TypeError("Symmetries must be sparse matrices.") from error
        normalized = _core.discrete_symmetry_normalize(
            self._projector_rows(projectors),
            *(self._matrix_rows(symmetry) for symmetry in symmetries),
        )
        normalized_projectors, *normalized_symmetries = normalized
        self.projectors = (
            None
            if normalized_projectors is None
            else [
                scipy_sparse.csr_matrix(projector)
                for projector in normalized_projectors
            ]
        )
        (
            self.time_reversal,
            self.particle_hole,
            self.chiral,
        ) = (
            None
            if symmetry is None
            else scipy_sparse.csr_matrix(symmetry)
            for symmetry in normalized_symmetries
        )

    @staticmethod
    def _matrix_rows(matrix):
        return None if matrix is None else matrix.toarray().tolist()

    @classmethod
    def _projector_rows(cls, projectors):
        return (
            None
            if projectors is None
            else [cls._matrix_rows(projector) for projector in projectors]
        )

    def validate(self, matrix):
        """Return the declared conservation laws and symmetries broken by a matrix."""
        dense = (
            matrix.toarray()
            if scipy_sparse.issparse(matrix)
            else np.asarray(matrix)
        )
        return _core.discrete_symmetry_validate(
            self._projector_rows(self.projectors),
            self._matrix_rows(self.time_reversal),
            self._matrix_rows(self.particle_hole),
            self._matrix_rows(self.chiral),
            dense.tolist(),
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
        if self.ham.ndim != 2 or self.ham.shape[0] != self.ham.shape[1]:
            raise ValueError("The cell Hamiltonian is not square.")
        inter_cell = np.asarray(
            system.inter_cell_hopping(args=args, params=params),
            dtype=complex,
        )
        self.hop = np.zeros_like(self.ham, dtype=complex)
        self.hop[:, : inter_cell.shape[1]] = inter_cell
        try:
            _core.validate_periodic_bands(
                self.ham.tolist(),
                self.hop.tolist(),
            )
        except ValueError as error:
            raise ValueError("The cell Hamiltonian is not Hermitian.") from error

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
        energies, first, second, eigenvectors = _core.lead_band_evaluation(
            self.ham.tolist(),
            self.hop.tolist(),
            float(momentum),
            int(derivative_order),
            bool(return_eigenvectors),
        )
        output = [np.asarray(energies, dtype=float)]
        if derivative_order:
            output.append(np.asarray(first, dtype=float))
        if derivative_order == 2:
            output.append(np.asarray(second, dtype=float))
        if return_eigenvectors:
            output.append(np.asarray(eigenvectors, dtype=complex))
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
    return _core.reflection_shot_noise(np.asarray(block, dtype=complex).tolist())


def phs_symmetrization(wave_functions, particle_hole):
    """Return a particle-hole-adapted orthonormal basis at a TRIM."""
    wave_functions = np.asarray(wave_functions, dtype=complex)
    particle_hole = (
        particle_hole.toarray()
        if scipy_sparse.issparse(particle_hole)
        else np.asarray(particle_hole, dtype=complex)
    )
    adapted, ordering = _core.particle_hole_basis(
        wave_functions.tolist(),
        particle_hole.tolist(),
    )
    return (
        np.asarray(adapted, dtype=complex),
        np.asarray(ordering, dtype=int),
    )


def modes(
    h_cell,
    h_hop,
    tol=1e6,
    stabilization=None,
    *,
    discrete_symmetry=None,
    projectors=None,
    time_reversal=None,
    particle_hole=None,
    chiral=None,
):
    """Solve Bloch modes of a nearest-cell periodic lead."""
    del tol, stabilization
    if discrete_symmetry is not None:
        projectors, time_reversal, particle_hole, chiral = discrete_symmetry
    if any(
        symmetry is not None
        for symmetry in (time_reversal, particle_hole, chiral)
    ):
        raise NotImplementedError(
            "Symmetry-related lead modes are tracked in "
            "https://github.com/matrixlab-research/thouless/issues/5"
        )
    h_cell = np.asarray(h_cell, dtype=complex)
    h_hop = np.asarray(h_hop, dtype=complex)
    if h_cell.ndim != 2 or h_cell.shape[0] != h_cell.shape[1]:
        raise ValueError("Cell Hamiltonian must be square")
    size = h_cell.shape[0]
    if h_hop.ndim != 2 or h_hop.shape[0] != size or h_hop.shape[1] > size:
        raise ValueError("Inter-cell hopping has an incompatible shape")
    square_hopping = np.zeros_like(h_cell)
    square_hopping[:, : h_hop.shape[1]] = h_hop

    if projectors is None:
        (
            wave_functions,
            velocities,
            momenta,
            incoming_count,
            stabilized_vectors,
            stabilized_vectors_lambda_inverse,
            square_root_hopping,
        ) = _core.lead_propagating_modes(
            h_cell.tolist(),
            square_hopping.tolist(),
        )
        block_nmodes = [incoming_count]
        projected = False
    else:
        (
            wave_functions,
            velocities,
            momenta,
            incoming_count,
            stabilized_vectors,
            stabilized_vectors_lambda_inverse,
            square_root_hopping,
            block_nmodes,
        ) = _core.lead_projected_modes(
            h_cell.tolist(),
            square_hopping.tolist(),
            [
                (
                    projector.toarray()
                    if scipy_sparse.issparse(projector)
                    else np.asarray(projector, dtype=complex)
                ).tolist()
                for projector in projectors
            ],
        )
        projected = True
    wave_functions = np.asarray(wave_functions, dtype=complex).reshape(
        size,
        len(velocities),
    )
    propagating = PropagatingModes(
        wave_functions,
        velocities,
        momenta,
    )
    propagating.block_nmodes = list(block_nmodes)
    interface_size = h_hop.shape[1]
    stabilized_vectors = np.asarray(stabilized_vectors, dtype=complex)
    stabilized_vectors_lambda_inverse = np.asarray(
        stabilized_vectors_lambda_inverse,
        dtype=complex,
    )
    if not projected:
        stabilized_vectors = stabilized_vectors[:interface_size]
        stabilized_vectors_lambda_inverse = stabilized_vectors_lambda_inverse[
            :interface_size
        ]
    square_root_hopping = np.asarray(square_root_hopping, dtype=complex)[
        :interface_size
    ]
    stabilized = StabilizedModes(
        stabilized_vectors,
        stabilized_vectors_lambda_inverse,
        incoming_count,
        sqrt_hop=square_root_hopping,
        selfenergy=_core.lead_retarded_self_energy(
            h_cell.tolist(),
            h_hop.tolist(),
            maximum_rank=incoming_count,
        ),
    )
    return propagating, stabilized


def selfenergy(h_cell, h_hop, tol=1e6):
    """Return the retarded self-energy of a semi-infinite periodic lead."""
    del tol
    h_cell = np.asarray(h_cell, dtype=complex)
    h_hop = np.asarray(h_hop, dtype=complex)
    if h_cell.ndim != 2 or h_cell.shape[0] != h_cell.shape[1]:
        raise ValueError("Cell Hamiltonian must be square")
    if (
        h_hop.ndim != 2
        or h_hop.shape[0] != h_cell.shape[0]
        or h_hop.shape[1] > h_cell.shape[0]
    ):
        raise ValueError("Inter-cell hopping has an incompatible shape")
    return np.asarray(
        _core.lead_retarded_self_energy(
            h_cell.tolist(),
            h_hop.tolist(),
        ),
        dtype=complex,
    )


def square_selfenergy(width, hopping, fermi_energy):
    """Return the analytic self-energy of a hard-wall square-lattice strip."""
    return np.asarray(
        _core.square_strip_self_energy(
            int(width),
            float(hopping),
            float(fermi_energy),
        ),
        dtype=complex,
    )


__all__ = [
    "Bands",
    "DiscreteSymmetry",
    "PropagatingModes",
    "StabilizedModes",
    "modes",
    "selfenergy",
    "square_selfenergy",
    "two_terminal_shotnoise",
]
