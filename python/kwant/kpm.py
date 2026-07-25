"""Kernel polynomial methods backed by the Thouless Rust core."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from thouless import _core

from ._common import ensure_rng
from .operator import _LocalOperator


SAMPLING = 2


def _is_system(value):
    return hasattr(value, "hamiltonian_submatrix") and hasattr(value, "sites")


def _dense_matrix(value, params=None):
    if _is_system(value):
        value = value.hamiltonian_submatrix(params=params, sparse=False)
    elif hasattr(value, "toarray"):
        value = value.toarray()
    try:
        matrix = np.asarray(value, dtype=complex)
    except Exception as error:
        raise ValueError(
            "'hamiltonian' is neither a matrix nor a Kwant system."
        ) from error
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("'hamiltonian' must be a square matrix")
    return matrix


def _real_if_close(value):
    return np.real_if_close(np.asarray(value))


def _squeeze_spectral_tensor(value, mean, output_shape=()):
    array = np.asarray(value, dtype=complex)
    if mean:
        array = array[:, 0]
    if output_shape == ():
        array = array[..., 0]
    else:
        array = array.reshape(array.shape[:-1] + tuple(output_shape))
    return _real_if_close(array)


def _squeeze_integral(value, mean, output_shape=()):
    array = np.asarray(value, dtype=complex)
    if mean:
        array = array[0]
    if output_shape == ():
        array = array[..., 0]
    else:
        array = array.reshape(array.shape[:-1] + tuple(output_shape))
    return _real_if_close(array)


class _RescaledOperator:
    """Small compatibility wrapper around a Rust-owned rescaled matrix."""

    def __init__(self, matrix):
        self._matrix = np.asarray(matrix, dtype=complex)
        self.shape = self._matrix.shape
        self.dtype = self._matrix.dtype

    def matvec(self, vector):
        values = _core.kpm_chebyshev_vectors(
            self._matrix,
            [np.asarray(vector, dtype=complex)],
            2,
        )
        return np.asarray(values[0][1], dtype=complex)

    def dot(self, vector):
        return self.matvec(vector)


class _VectorFactory:
    """Finite view of an iterable of Hilbert-space vectors."""

    def __init__(self, vectors=None, num_vectors=None, accumulate=True):
        if not isinstance(vectors, Iterable):
            raise AssertionError("vectors must be iterable")
        try:
            available = len(vectors)
            if num_vectors is None:
                num_vectors = available
        except TypeError:
            available = math.inf
            if num_vectors is None:
                raise ValueError(
                    "'num_vectors' must be specified when 'vectors' has no len() method."
                )
        self._max_vectors = available
        self._iterator = iter(vectors)
        self.accumulate = bool(accumulate)
        self.saved_vectors = []
        self.num_vectors = 0
        self._last_idx = -math.inf
        self._last_vector = None
        self.add_vectors(num_vectors=num_vectors)

    def _fill_in_saved_vectors(self, index):
        if index < self._last_idx and not self.accumulate:
            raise ValueError("Cannot get previous values if 'accumulate' is False")
        if index >= self.num_vectors:
            raise IndexError("Requested more vectors than available")
        self._last_idx = index
        if self.accumulate:
            if self.saved_vectors[index] is None:
                self.saved_vectors[index] = next(self._iterator)
        else:
            self._last_vector = next(self._iterator)

    def __getitem__(self, index):
        self._fill_in_saved_vectors(index)
        if self.accumulate:
            return self.saved_vectors[index]
        return self._last_vector

    def __iter__(self):
        for index in range(self.num_vectors):
            yield self[index]

    def add_vectors(self, num_vectors=None):
        if (
            num_vectors is None
            or num_vectors <= 0
            or num_vectors != int(num_vectors)
        ):
            raise ValueError("'num_vectors' must be a positive integer")
        num_vectors = int(num_vectors)
        if self.num_vectors + num_vectors > self._max_vectors:
            raise ValueError("'num_vectors' is larger than available vectors")
        self.num_vectors += num_vectors
        if self.accumulate:
            self.saved_vectors.extend([None] * num_vectors)


def _orbital_indices(syst, where):
    if not _is_system(syst):
        matrix = _dense_matrix(syst)
        if where is None:
            return matrix.shape[0], list(range(matrix.shape[0]))
        return matrix.shape[0], [int(index) for index in where]

    sites = list(syst.sites)
    offsets = [0]
    for site in sites:
        norbs = site.family.norbs
        if norbs is None:
            raise ValueError("KPM requires defined orbital counts")
        offsets.append(offsets[-1] + int(norbs))
    if where is None:
        selected = list(range(len(sites)))
    elif callable(where):
        selected = [index for index, site in enumerate(sites) if where(site)]
    else:
        selected = []
        for site in where:
            if isinstance(site, (int, np.integer)):
                selected.append(int(site))
            else:
                try:
                    selected.append(syst.id_by_site[site])
                except (KeyError, TypeError) as error:
                    raise ValueError(f"Unknown site {site!r}") from error
    orbitals = [
        orbital
        for site in selected
        for orbital in range(offsets[site], offsets[site + 1])
    ]
    return offsets[-1], orbitals


def RandomVectors(syst, where=None, rng=None):
    """Yield random-phase vectors supported on selected orbitals."""
    rng = ensure_rng(rng)
    dimension, orbitals = _orbital_indices(syst, where)
    while True:
        vector = np.zeros(dimension, dtype=complex)
        vector[orbitals] = np.exp(
            2j * np.pi * rng.random_sample(len(orbitals))
        )
        yield vector


class LocalVectors:
    """Yield one normalized orbital basis vector at a time."""

    def __init__(self, syst, where=None, *args):
        del args
        self.tot_norbs, self.orbs = _orbital_indices(syst, where)
        self._idx = 0

    def __len__(self):
        return len(self.orbs)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self):
            raise StopIteration("Too many vectors requested from this generator")
        vector = np.zeros(self.tot_norbs)
        vector[self.orbs[self._idx]] = 1
        self._idx += 1
        return vector


def _kernel_spec(kernel):
    if kernel is None or kernel is jackson_kernel:
        return "jackson", None
    if kernel is lorentz_kernel:
        return "lorentz", 4.0
    return None, None


class SpectralDensity:
    """Kernel-polynomial spectral density of a Hamiltonian and operator."""

    def __init__(
        self,
        hamiltonian,
        params=None,
        operator=None,
        num_vectors=10,
        num_moments=None,
        energy_resolution=None,
        vector_factory=None,
        bounds=None,
        eps=0.05,
        rng=None,
        kernel=None,
        mean=True,
        accumulate_vectors=True,
    ):
        if num_moments and energy_resolution:
            raise TypeError(
                "either 'num_moments' or 'energy_resolution' must be provided."
            )
        if eps <= 0:
            raise ValueError("'eps' must be positive")
        self.eps = eps
        self.mean = bool(mean)
        self._source_hamiltonian = hamiltonian
        dense_hamiltonian = _dense_matrix(hamiltonian, params=params)
        self._dense_hamiltonian = dense_hamiltonian

        if operator is None:
            self.operator = None
            self._operator_kind = "identity"
            self._operator_matrix = None
            self._operator_output_shape = ()
        elif isinstance(operator, _LocalOperator):
            self.operator = operator.bind(params=params)
            self._operator_kind = "callback"
            self._operator_matrix = None
            self._operator_output_shape = None
        elif callable(operator):
            self.operator = operator
            self._operator_kind = "callback"
            self._operator_matrix = None
            self._operator_output_shape = None
        elif hasattr(operator, "dot"):
            self._operator_matrix = _dense_matrix(operator)
            self.operator = operator
            self._operator_kind = "matrix"
            self._operator_output_shape = ()
        else:
            raise ValueError(
                "Parameter 'operator' has no '.dot' attribute and is not callable."
            )

        rng = ensure_rng(rng)
        self._v0 = np.exp(
            2j * np.pi * rng.random_sample(dense_hamiltonian.shape[0])
        )
        rescaled, self._a, self._b = _core.kpm_rescale_hamiltonian(
            dense_hamiltonian,
            eps,
            bounds,
        )
        self._rescaled_matrix = np.asarray(rescaled, dtype=complex)
        self.hamiltonian = _RescaledOperator(self._rescaled_matrix)
        self.bounds = (self._b - self._a, self._b + self._a)

        if energy_resolution:
            num_moments = math.ceil((1.6 * self._a) / energy_resolution)
        elif num_moments is None:
            num_moments = 100
        if num_moments <= 0 or num_moments != int(num_moments):
            raise ValueError("'num_moments' must be a positive integer")
        self.num_moments = int(num_moments)

        if vector_factory is None:
            vector_factory = RandomVectors(dense_hamiltonian, rng=rng)
        elif not isinstance(vector_factory, Iterable):
            raise TypeError("vector_factory must be iterable")
        else:
            try:
                len(vector_factory)
            except TypeError:
                if num_vectors is None:
                    raise ValueError(
                        "num_vectors must be provided if vector_factory has no length."
                    )
        self._vector_factory = _VectorFactory(
            vector_factory,
            num_vectors=num_vectors,
            accumulate=accumulate_vectors,
        )
        self._initial_vectors = [
            np.asarray(self._vector_factory[index], dtype=complex)
            for index in range(self._vector_factory.num_vectors)
        ]
        self.kernel = kernel if kernel is not None else jackson_kernel
        self._last_two_alphas = []
        self._moments_list = []
        self._recalculate()

    @property
    def num_vectors(self):
        return len(self._initial_vectors)

    @property
    def energies(self):
        return self._energies

    def _raw_moments(self, chebyshev):
        if self._operator_kind == "identity":
            return np.asarray(
                _core.kpm_scalar_moments(
                    self._initial_vectors,
                    chebyshev,
                    None,
                ),
                dtype=complex,
            )
        if self._operator_kind == "matrix":
            return np.asarray(
                _core.kpm_scalar_moments(
                    self._initial_vectors,
                    chebyshev,
                    self._operator_matrix,
                ),
                dtype=complex,
            )

        rows = []
        expected_shape = None
        for vector, initial in enumerate(self._initial_vectors):
            moments = []
            for ket in chebyshev[vector]:
                value = np.asarray(self.operator(initial, ket), dtype=complex)
                shape = value.shape
                if expected_shape is None:
                    expected_shape = shape
                elif shape != expected_shape:
                    raise ValueError("operator returned values with inconsistent shape")
                moments.append(value.reshape(-1))
            rows.append(moments)
        self._operator_output_shape = expected_shape
        return np.asarray(rows, dtype=complex)

    def _recalculate(self):
        chebyshev = np.asarray(
            _core.kpm_chebyshev_vectors(
                self._rescaled_matrix,
                self._initial_vectors,
                self.num_moments,
            ),
            dtype=complex,
        )
        self._chebyshev = chebyshev
        raw = self._raw_moments(chebyshev)
        self._raw_moments_array = raw
        if self._operator_output_shape == ():
            public_raw = raw[..., 0]
        else:
            public_raw = raw.reshape(
                raw.shape[:-1] + tuple(self._operator_output_shape)
            )
        self._moments_list = [np.asarray(row) for row in public_raw]
        self._last_two_alphas = [
            (
                np.asarray(row[-2 if len(row) > 1 else 0]),
                np.asarray(row[-1]),
            )
            for row in chebyshev
        ]

        kernel_name, kernel_strength = _kernel_spec(self.kernel)
        if kernel_name is None:
            base = _core.kpm_reconstruct(
                raw,
                self._a,
                self._b,
                "none",
                None,
                self.mean,
            )
            moments = np.asarray(base[3], dtype=complex)
            public_moments = _squeeze_spectral_tensor(
                np.swapaxes(moments, 0, 0),
                self.mean,
                self._operator_output_shape,
            )
            custom = np.asarray(self.kernel(public_moments), dtype=complex)
            if self.mean:
                custom = custom.reshape((self.num_moments, 1, -1))
            else:
                custom = custom.reshape(
                    (self.num_moments, self.num_vectors, -1)
                )
            energies, densities, gammas, moments = (
                _core.kpm_reconstruct_stabilized(
                    custom,
                    self._a,
                    self._b,
                )
            )
            self._energies = np.asarray(energies)
            self._density_tensor = np.asarray(densities, dtype=complex)
            self._gamma_tensor = np.asarray(gammas, dtype=complex)
            self._stabilized_moments = np.asarray(moments, dtype=complex)
            self.densities = _squeeze_spectral_tensor(
                self._density_tensor,
                self.mean,
                self._operator_output_shape,
            )
            self._gammas = _squeeze_spectral_tensor(
                self._gamma_tensor,
                self.mean,
                self._operator_output_shape,
            )
            return

        energies, densities, gammas, moments = _core.kpm_reconstruct(
            raw,
            self._a,
            self._b,
            kernel_name,
            kernel_strength,
            self.mean,
        )
        self._energies = np.asarray(energies)
        self._density_tensor = np.asarray(densities, dtype=complex)
        self._gamma_tensor = np.asarray(gammas, dtype=complex)
        self._stabilized_moments = np.asarray(moments, dtype=complex)
        self.densities = _squeeze_spectral_tensor(
            self._density_tensor,
            self.mean,
            self._operator_output_shape,
        )
        self._gammas = _squeeze_spectral_tensor(
            self._gamma_tensor,
            self.mean,
            self._operator_output_shape,
        )

    def _moments(self):
        return _squeeze_spectral_tensor(
            self._stabilized_moments,
            self.mean,
            self._operator_output_shape,
        )

    def __call__(self, energy=None):
        if energy is None:
            return self.energies, self.densities
        input_array = np.asarray(energy)
        flat = np.atleast_1d(input_array).astype(float).reshape(-1)
        values = np.asarray(
            _core.kpm_evaluate(
                self._stabilized_moments,
                self._a,
                self._b,
                flat,
            ),
            dtype=complex,
        )
        public = _squeeze_spectral_tensor(
            values,
            self.mean,
            self._operator_output_shape,
        )
        if input_array.ndim == 0:
            return _real_if_close(public[0])
        return public.reshape(input_array.shape + public.shape[1:])

    def integrate(self, distribution_function=None):
        if distribution_function is None:
            distribution = np.ones(len(self.energies))
        else:
            distribution = np.asarray(
                distribution_function(self.energies), dtype=float
            )
            if distribution.shape != self.energies.shape:
                distribution = np.broadcast_to(
                    distribution, self.energies.shape
                )
        result = _core.kpm_integrate(
            self._gamma_tensor,
            distribution,
            self._a,
            self._b,
        )
        return _squeeze_integral(
            result,
            self.mean,
            self._operator_output_shape,
        )

    def add_moments(self, num_moments=None, *, energy_resolution=None):
        if not ((num_moments is None) ^ (energy_resolution is None)):
            raise TypeError(
                "either 'num_moments' or 'energy_resolution' must be provided."
            )
        if energy_resolution is not None:
            if energy_resolution <= 0:
                raise ValueError("'energy_resolution' must be positive")
            present = self._a * 1.6 / self.num_moments
            if present < energy_resolution:
                raise ValueError(
                    "Energy resolution is already smaller than the requested resolution"
                )
            target = math.ceil((1.6 * self._a) / energy_resolution)
            num_moments = target
        if (
            num_moments is None
            or num_moments <= 0
            or num_moments != int(num_moments)
        ):
            raise ValueError("'num_moments' must be a positive integer")
        if not self._vector_factory.accumulate:
            raise ValueError(
                "Cannot increase the number of moments if 'accumulate_vectors' is 'False'."
            )
        self.num_moments += int(num_moments)
        self._recalculate()

    def add_vectors(self, num_vectors=None):
        old_count = self._vector_factory.num_vectors
        self._vector_factory.add_vectors(num_vectors)
        for index in range(old_count, self._vector_factory.num_vectors):
            self._initial_vectors.append(
                np.asarray(self._vector_factory[index], dtype=complex)
            )
        self._recalculate()


class _MatrixOperator:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=complex)


def _normalize_operator(operator, params):
    if operator is None:
        return None
    if isinstance(operator, _LocalOperator):
        return operator.bind(params=params)
    if callable(operator):
        return operator
    if hasattr(operator, "dot"):
        return _MatrixOperator(_dense_matrix(operator))
    raise TypeError(
        "The operators must have a '.dot' attribute or must be callable."
    )


def _apply_to_vectors(operator, vectors):
    values = np.asarray(vectors, dtype=complex)
    if operator is None:
        return values
    if isinstance(operator, _MatrixOperator):
        return np.asarray(
            _core.kpm_apply_operator(operator.matrix, values),
            dtype=complex,
        )
    transformed = [
        [
            np.asarray(operator(vector), dtype=complex)
            for vector in moment_vectors
        ]
        for moment_vectors in values
    ]
    return np.asarray(transformed, dtype=complex)


class Correlator:
    """Two-operator Kubo-Bastin response from Rust KPM moments."""

    def __init__(self, hamiltonian, operator1=None, operator2=None, **kwargs):
        params = kwargs.get("params")
        self.mean = bool(kwargs.get("mean", True))
        self.operator1 = _normalize_operator(operator1, params)
        self.operator2 = _normalize_operator(operator2, params)
        kwargs.pop("operator", None)
        kwargs["accumulate_vectors"] = True
        self._spectrum_R = SpectralDensity(
            hamiltonian, operator=lambda bra, ket: ket, **kwargs
        )
        self._a = self._spectrum_R._a
        self._b = self._spectrum_R._b
        self.num_vectors = self._spectrum_R.num_vectors
        self.num_moments = self._spectrum_R.num_moments
        self._recalculate()

    @property
    def energies(self):
        return self._spectrum_R.energies

    def _initial_after_operator(self):
        initial = np.asarray(self._spectrum_R._initial_vectors, dtype=complex)
        shaped = initial[:, None, :]
        return _apply_to_vectors(self.operator1, shaped)[:, 0, :]

    def _recalculate(self):
        right_chebyshev = self._spectrum_R._chebyshev
        psi = _apply_to_vectors(self.operator2, right_chebyshev)
        left_initial = self._initial_after_operator()
        omega = np.asarray(
            _core.kpm_chebyshev_vectors(
                self._spectrum_R._rescaled_matrix,
                left_initial,
                self.num_moments,
            ),
            dtype=complex,
        )
        self._psi = np.swapaxes(psi, 1, 2)
        self._omega = omega
        self.moments_matrix = np.asarray(
            _core.kpm_correlation_moments(
                omega,
                psi,
                self.mean,
            ),
            dtype=complex,
        )
        self._integral_factor = np.asarray(
            _core.kpm_correlation_integral_factor(
                self.moments_matrix,
                self.num_moments,
                "jackson",
                None,
            ),
            dtype=complex,
        )

    def __call__(self, mu=0, temperature=0):
        value = np.asarray(
            _core.kpm_correlation_response(
                self._integral_factor,
                self._a,
                self._b,
                mu,
                temperature,
            ),
            dtype=complex,
        )
        if self.mean:
            value = value[0]
        return _real_if_close(value)

    def add_moments(self, num_moments=None, *, energy_resolution=None):
        self._spectrum_R.add_moments(
            num_moments=num_moments,
            energy_resolution=energy_resolution,
        )
        self.num_moments = self._spectrum_R.num_moments
        self._recalculate()

    def add_vectors(self, num_vectors=None):
        self._spectrum_R.add_vectors(num_vectors)
        self.num_vectors = self._spectrum_R.num_vectors
        self._recalculate()


def _orbital_positions(system):
    positions = []
    for site in system.sites:
        norbs = site.family.norbs
        if norbs is None:
            raise ValueError("KPM requires defined orbital counts")
        positions.extend([site.pos] * int(norbs))
    return np.asarray(positions, dtype=float)


def _velocity(hamiltonian, params, operator, positions):
    if isinstance(operator, _LocalOperator) or not isinstance(operator, str):
        return operator
    directions = {"x": 0, "y": 1, "z": 2}
    try:
        direction = directions[operator]
    except KeyError as error:
        raise ValueError(f"{operator} is not an allowed direction.") from error
    dense = _dense_matrix(hamiltonian, params=params)
    if _is_system(hamiltonian):
        positions = _orbital_positions(hamiltonian)
    if positions is None:
        raise ValueError("positions are required to construct a velocity")
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or direction >= positions.shape[1]:
        raise ValueError(f"{operator} is not an allowed direction.")
    return np.asarray(
        _core.kpm_velocity_operator(dense, positions, direction),
        dtype=complex,
    )


def conductivity(hamiltonian, alpha="x", beta="x", positions=None, **kwargs):
    """Return a Kubo-Bastin conductivity correlator."""
    if positions is None and not _is_system(hamiltonian):
        raise ValueError("If 'hamiltonian' is a matrix, positions must be provided")
    params = kwargs.get("params")
    alpha = _velocity(hamiltonian, params, alpha, positions)
    beta = _velocity(hamiltonian, params, beta, positions)
    return Correlator(
        hamiltonian,
        operator1=alpha,
        operator2=beta,
        **kwargs,
    )


def fermi_distribution(energy, mu, temperature):
    """Evaluate a Fermi-Dirac distribution."""
    input_array = np.asarray(energy, dtype=float)
    values = np.asarray(
        _core.kpm_fermi_distribution(
            np.atleast_1d(input_array).reshape(-1),
            mu,
            temperature,
        )
    ).reshape(np.atleast_1d(input_array).shape)
    if input_array.ndim == 0:
        return values[0]
    return values.reshape(input_array.shape)


def jackson_kernel(moments):
    """Apply Jackson damping along the first moment axis."""
    array = np.asarray(moments)
    flattened = array.reshape((len(array), -1))
    damped = np.asarray(
        _core.kpm_apply_kernel(flattened, "jackson", None)
    )
    return _real_if_close(damped.reshape(array.shape))


def lorentz_kernel(moments, l=4):
    """Apply Lorentz damping along the first moment axis."""
    array = np.asarray(moments)
    flattened = array.reshape((len(array), -1))
    damped = np.asarray(
        _core.kpm_apply_kernel(flattened, "lorentz", l)
    )
    return _real_if_close(damped.reshape(array.shape))


def _rescale(hamiltonian, eps, v0, bounds):
    """Return the Rust-rescaled Hamiltonian and affine energy scale."""
    del v0
    matrix, half_width, center = _core.kpm_rescale_hamiltonian(
        _dense_matrix(hamiltonian),
        eps,
        bounds,
    )
    return _RescaledOperator(matrix), (half_width, center)


def _chebyshev_nodes(n_sampling):
    return np.asarray(_core.kpm_chebyshev_nodes(n_sampling))


def _calc_fft_moments(moments):
    array = np.asarray(moments, dtype=complex)
    if array.ndim == 1:
        raw = array[:, None, None]
    else:
        raw = array.reshape((array.shape[0], 1, -1))
    _, densities, gammas, _ = _core.kpm_reconstruct_stabilized(
        raw, 1.0, 0.0
    )
    densities = np.asarray(densities, dtype=complex)[:, 0]
    gammas = np.asarray(gammas, dtype=complex)[:, 0]
    sample_count = SAMPLING * len(array)
    extra_shape = array.shape[1:]
    densities = densities.reshape((sample_count,) + extra_shape)
    gammas = gammas.reshape((sample_count,) + extra_shape)
    return _real_if_close(densities), _real_if_close(gammas)


__all__ = [
    "SpectralDensity",
    "Correlator",
    "conductivity",
    "RandomVectors",
    "LocalVectors",
    "jackson_kernel",
    "lorentz_kernel",
    "fermi_distribution",
]
