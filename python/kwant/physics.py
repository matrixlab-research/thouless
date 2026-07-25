"""Lead dispersion and propagating modes for periodic tight-binding systems."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy import sparse as scipy_sparse
from thouless import _core

Linsys = namedtuple("Linsys", ["eigenproblem", "v", "extract"])


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


def setup_linsys(h_cell, h_hop, tol=1e6, stabilization=None):
    """Construct Kwant's stabilized translation eigenproblem."""
    h_cell = np.asarray(h_cell)
    h_hop = np.asarray(h_hop)
    if h_cell.ndim != 2 or h_cell.shape[0] != h_cell.shape[1]:
        raise ValueError("Cell Hamiltonian must be square")
    if h_hop.shape != h_cell.shape:
        raise ValueError("Inter-cell hopping must have the cell shape")
    if stabilization is None:
        singular_basis = False
        force_imaginary = False
        regularize_pencil = None
    else:
        if len(stabilization) != 2:
            raise ValueError("stabilization must contain two booleans")
        singular_basis = True
        force_imaginary = bool(stabilization[0])
        regularize_pencil = True if bool(stabilization[1]) else None
    native = _core.lead_setup_linear_system(
        np.asarray(h_cell, dtype=complex).tolist(),
        np.asarray(h_hop, dtype=complex).tolist(),
        float(tol),
        singular_basis,
        force_imaginary,
        regularize_pencil,
    )
    def native_matrix(rows):
        matrix = np.asarray(rows, dtype=complex)
        return matrix.real if native.uses_real_arithmetic else matrix

    left = native_matrix(native.left)
    right = None if native.right is None else native_matrix(native.right)
    square_root_hopping = native_matrix(native.square_root_hopping)

    def extract(projected, inverse_bloch_factor):
        projected = np.asarray(projected, dtype=complex)
        inverse_bloch_factor = np.asarray(inverse_bloch_factor, dtype=complex)
        if projected.ndim == 1:
            factor = complex(inverse_bloch_factor.reshape(()))
            return np.asarray(native.extract(projected.tolist(), factor), dtype=complex)
        if projected.ndim != 2:
            raise ValueError("projected eigenvectors must be one- or two-dimensional")
        factors = np.broadcast_to(
            inverse_bloch_factor,
            (projected.shape[1],),
        )
        columns = [
            native.extract(projected[:, column].tolist(), complex(factors[column]))
            for column in range(projected.shape[1])
        ]
        if not columns:
            return np.zeros((h_cell.shape[0], 0), dtype=complex)
        return np.asarray(columns, dtype=complex).T

    return Linsys((left, right), square_root_hopping, extract)


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
    if discrete_symmetry is not None:
        projectors, time_reversal, particle_hole, chiral = discrete_symmetry
    has_symmetries = any(
        symmetry is not None
        for symmetry in (time_reversal, particle_hole, chiral)
    )

    def symmetry_rows(symmetry):
        if symmetry is None:
            return None
        return (
            symmetry.toarray()
            if scipy_sparse.issparse(symmetry)
            else np.asarray(symmetry, dtype=complex)
        ).tolist()

    h_cell = np.asarray(h_cell, dtype=complex)
    h_hop = np.asarray(h_hop, dtype=complex)
    if h_cell.ndim != 2 or h_cell.shape[0] != h_cell.shape[1]:
        raise ValueError("Cell Hamiltonian must be square")
    size = h_cell.shape[0]
    if h_hop.ndim != 2 or h_hop.shape[0] != size or h_hop.shape[1] > size:
        raise ValueError("Inter-cell hopping has an incompatible shape")
    square_hopping = np.zeros_like(h_cell)
    square_hopping[:, : h_hop.shape[1]] = h_hop

    if projectors is None and stabilization is not None:
        (
            wave_functions,
            velocities,
            momenta,
            incoming_count,
            stabilized_vectors,
            stabilized_vectors_lambda_inverse,
            square_root_hopping,
        ) = _core.lead_singular_basis_modes(
            h_cell.tolist(),
            square_hopping.tolist(),
            np.finfo(float).eps * float(tol),
            symmetry_rows(time_reversal),
            symmetry_rows(particle_hole),
            symmetry_rows(chiral),
        )
        block_nmodes = [incoming_count]
        projected = False
    elif projectors is None and not has_symmetries:
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
    elif projectors is None:
        (
            wave_functions,
            velocities,
            momenta,
            incoming_count,
            stabilized_vectors,
            stabilized_vectors_lambda_inverse,
            square_root_hopping,
        ) = _core.lead_symmetric_modes(
            h_cell.tolist(),
            square_hopping.tolist(),
            symmetry_rows(time_reversal),
            symmetry_rows(particle_hole),
            symmetry_rows(chiral),
        )
        block_nmodes = [incoming_count]
        projected = False
    else:
        projector_rows = [
            (
                projector.toarray()
                if scipy_sparse.issparse(projector)
                else np.asarray(projector, dtype=complex)
            ).tolist()
            for projector in projectors
        ]
        if has_symmetries:
            result = _core.lead_symmetric_projected_modes(
                h_cell.tolist(),
                square_hopping.tolist(),
                projector_rows,
                symmetry_rows(time_reversal),
                symmetry_rows(particle_hole),
                symmetry_rows(chiral),
            )
        else:
            result = _core.lead_projected_modes(
                h_cell.tolist(),
                square_hopping.tolist(),
                projector_rows,
            )
        (
            wave_functions,
            velocities,
            momenta,
            incoming_count,
            stabilized_vectors,
            stabilized_vectors_lambda_inverse,
            square_root_hopping,
            block_nmodes,
        ) = result
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
    "setup_linsys",
    "modes",
    "selfenergy",
    "square_selfenergy",
    "two_terminal_shotnoise",
]
