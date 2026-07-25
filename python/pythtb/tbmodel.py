"""PythTB model compatibility implemented over the Thouless Rust core."""

from __future__ import annotations

import copy
import inspect
import itertools
import warnings

import numpy as np

from thouless import _core

from .lattice import Lattice


def _provider_names(provider):
    if isinstance(provider, str):
        return [provider]
    return list(inspect.signature(provider).parameters)


class TBModel:
    """Build and solve a tight-binding Hamiltonian."""

    def __init__(self, lattice: Lattice, spinful: bool = False):
        if not isinstance(lattice, Lattice):
            raise TypeError("lattice must be a pythtb.Lattice")
        self._lattice = copy.copy(lattice)
        self._spinful = bool(spinful)
        self._nspin = 2 if self._spinful else 1
        shape = (self.norb, 2, 2) if self._spinful else (self.norb,)
        self._site_energies = np.zeros(
            shape, dtype=complex if self._spinful else float
        )
        self._hoppings: list[dict] = []
        self._onsite_providers: dict[int, object] = {}
        self._hopping_providers: dict[tuple[int, int, tuple[int, ...]], object] = {}
        self._from_w90 = False
        self._assume_position_operator_diagonal = True

    @property
    def lattice(self):
        return copy.copy(self._lattice)

    @property
    def dim_r(self):
        return self._lattice.dim_r

    @property
    def dim_k(self):
        return self._lattice.dim_k

    @property
    def nspin(self):
        return self._nspin

    @property
    def spinful(self):
        return self._spinful

    @property
    def periodic_dirs(self):
        return self._lattice.periodic_dirs

    @property
    def norb(self):
        return self._lattice.norb

    @property
    def nstate(self):
        return self.norb * self.nspin

    @property
    def orb_vecs(self):
        return self._lattice.orb_vecs

    @property
    def lat_vecs(self):
        return self._lattice.lat_vecs

    @property
    def recip_lat_vecs(self):
        return self._lattice.recip_lat_vecs

    @property
    def recip_volume(self):
        return self._lattice.recip_volume

    @property
    def cell_volume(self):
        return self._lattice.cell_volume

    @property
    def onsite(self):
        return self._site_energies.copy()

    @property
    def hoppings(self):
        result = []
        for hopping in self._hoppings:
            entry = {
                "amplitude": hopping["amplitude"].copy()
                if self.spinful
                else complex(hopping["amplitude"][0, 0]),
                "from_orbital": hopping["target"],
                "to_orbital": hopping["source"],
            }
            if any(hopping["offset"]):
                entry["lattice_vector"] = list(hopping["offset"])
            result.append(entry)
        return result

    @property
    def nhops(self):
        return len(self._hoppings)

    @property
    def parameters(self):
        result = []
        for index, provider in self._onsite_providers.items():
            result.append(
                {
                    "kind": "onsite",
                    "orbitals": index,
                    "names": _provider_names(provider),
                }
            )
        for (target, source, offset), provider in self._hopping_providers.items():
            result.append(
                {
                    "kind": "hopping",
                    "orbitals": (target, source),
                    "R": offset,
                    "names": _provider_names(provider),
                }
            )
        return result

    @property
    def from_w90(self):
        return self._from_w90

    @property
    def assume_position_operator_diagonal(self):
        return self._assume_position_operator_diagonal

    @assume_position_operator_diagonal.setter
    def assume_position_operator_diagonal(self, value):
        if not isinstance(value, bool):
            raise ValueError("assume_position_operator_diagonal must be a boolean.")
        self._assume_position_operator_diagonal = value

    def display(self):
        """Deprecated report printer retained for source compatibility."""
        warnings.warn(
            "display() is deprecated; use info(show=True)",
            FutureWarning,
            stacklevel=2,
        )
        return self.info(show=True)

    def info(self, show=True, short=False):
        """Return or print a compact, deterministic model report."""
        lines = [
            "----------------------------------------",
            "       Tight-binding model report",
            "----------------------------------------",
            f"real-space dimension       = {self.dim_r}",
            f"reciprocal-space dimension = {self.dim_k}",
            f"periodic directions        = {self.periodic_dirs}",
            f"number of orbitals         = {self.norb}",
            f"spinful                     = {self.spinful}",
            f"number of electronic states = {self.nstate}",
        ]
        if not short:
            lines.append("Site energies:")
            for index, energy in enumerate(self._site_energies):
                lines.append(f"  <{index}| H |{index}> = {energy}")
            lines.append("Hoppings:")
            for hopping in self.hoppings:
                lines.append(
                    "  "
                    f"<{hopping['from_orbital']}| H |"
                    f"{hopping['to_orbital']}> = {hopping['amplitude']}"
                )
        report = "\n".join(lines)
        if show:
            print(report)
            return None
        return report

    def clear_hoppings(self):
        """Remove all numeric and parameter-dependent hopping terms."""
        self._hoppings.clear()
        self._hopping_providers.clear()

    def clear_onsite(self):
        """Reset all onsite terms to zero."""
        self._site_energies.fill(0)
        self._onsite_providers.clear()

    def get_num_orbitals(self):
        """Deprecated alias for :attr:`norb`."""
        warnings.warn(
            "get_num_orbitals() is deprecated; use norb",
            FutureWarning,
            stacklevel=2,
        )
        return self.norb

    def get_orb(self):
        """Deprecated alias for :meth:`get_orb_vecs`."""
        warnings.warn(
            "get_orb() is deprecated; use get_orb_vecs()",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_orb_vecs()

    def nn_bonds(self, n_shells, report=False):
        """Return real-space nearest-neighbor shell summaries and bonds."""
        return self._lattice.nn_bonds(n_shells, report=report)

    def get_lat(self):
        """Deprecated alias for :meth:`get_lat_vecs`."""
        warnings.warn(
            "get_lat() is deprecated; use get_lat_vecs()",
            FutureWarning,
            stacklevel=2,
        )
        return self.get_lat_vecs()

    def get_lat_vecs(self):
        return self.lat_vecs

    def get_orb_vecs(self, cartesian=False):
        return self._lattice.get_orb_vecs(cartesian)

    def copy(self):
        return copy.deepcopy(self)

    def _value_to_block(self, value, *, onsite):
        if self.spinful:
            array = np.asarray(value, dtype=complex)
            if array.ndim == 0:
                block = complex(array) * np.eye(2, dtype=complex)
            elif array.shape == (4,):
                a, b, c, d = array
                block = np.array([[a + d, b - 1j * c], [b + 1j * c, a - d]])
            elif array.shape == (2, 2):
                block = array.copy()
            else:
                raise ValueError("spinful values must be a scalar, 4-vector, or 2x2 matrix")
            if onsite and not np.allclose(block, block.conj().T):
                raise ValueError("Onsite terms should be Hermitian for spinful models.")
            return block
        array = np.asarray(value)
        if array.ndim != 0:
            raise ValueError("spinless values must be scalars")
        scalar = complex(array)
        if onsite and abs(scalar.imag) > 1e-12:
            raise ValueError("Onsite terms should be real for spinless models.")
        return np.array([[scalar]], dtype=complex)

    @staticmethod
    def _evaluate(provider, values):
        if isinstance(provider, str):
            if provider not in values:
                raise ValueError(f"Missing value for parameter {provider!r}")
            return values[provider]
        names = _provider_names(provider)
        missing = [name for name in names if name not in values]
        if missing:
            raise ValueError(f"Missing values for parameters: {missing}")
        return provider(**{name: values[name] for name in names})

    def set_onsite(self, onsite_en, ind_i=None, mode="set"):
        mode = mode.lower()
        if mode == "reset":
            mode = "set"
        if mode not in ("set", "add"):
            raise ValueError("Mode should be either 'set' or 'add'.")
        if ind_i is None:
            if not isinstance(onsite_en, (list, np.ndarray)):
                raise TypeError(
                    "When ind_i is not specified, onsite_en must be a list or array."
                )
            if len(onsite_en) != self.norb:
                raise ValueError(
                    "List of onsite energies must include a value for every orbital."
                )
            pairs = list(enumerate(onsite_en))
        else:
            if not 0 <= ind_i < self.norb:
                raise ValueError("Index ind_i is not within the range of orbitals.")
            pairs = [(int(ind_i), onsite_en)]

        for index, value in pairs:
            if callable(value) or isinstance(value, str):
                self._onsite_providers[index] = value
                self._site_energies[index] = 0
                continue
            block = self._value_to_block(value, onsite=True)
            canonical = block if self.spinful else block[0, 0].real
            if mode == "set":
                self._site_energies[index] = canonical
            else:
                self._site_energies[index] += canonical
            self._onsite_providers.pop(index, None)

    def _validate_hopping_key(self, target, source, offset):
        if not 0 <= target < self.norb or not 0 <= source < self.norb:
            raise ValueError("Hopping orbital index is out of range.")
        if len(offset) != self.dim_r:
            raise ValueError("Lattice vector has wrong dimension.")
        for axis, component in enumerate(offset):
            if axis not in self.periodic_dirs and component != 0:
                raise ValueError("Hopping has a nonzero component in an open direction.")
        if target == source and all(component == 0 for component in offset):
            raise ValueError("Onsite hopping must be set with set_onsite.")

    def _matching_hopping(self, target, source, offset):
        opposite = tuple(-component for component in offset)
        for index, hopping in enumerate(self._hoppings):
            if (
                hopping["target"] == target
                and hopping["source"] == source
                and hopping["offset"] == offset
            ):
                return index, False
            if (
                hopping["target"] == source
                and hopping["source"] == target
                and hopping["offset"] == opposite
            ):
                return index, True
        return None, False

    def _store_hopping(
        self, block, target, source, offset, mode="set", allow_conjugate_pair=False
    ):
        match, conjugate = self._matching_hopping(target, source, offset)
        if conjugate and not allow_conjugate_pair:
            raise ValueError("Hopping is already specified by its conjugate pair.")
        if match is None:
            self._hoppings.append(
                {
                    "amplitude": block.copy(),
                    "target": target,
                    "source": source,
                    "offset": offset,
                }
            )
            return
        canonical = block.conj().T if conjugate else block
        if mode == "set" and not conjugate:
            self._hoppings[match]["amplitude"] = canonical
        else:
            self._hoppings[match]["amplitude"] += canonical

    def set_hop(
        self,
        hop_amp,
        ind_i,
        ind_j,
        ind_R=None,
        mode="set",
        allow_conjugate_pair=False,
    ):
        mode = mode.lower()
        if mode == "reset":
            mode = "set"
        if mode not in ("set", "add"):
            raise ValueError("Mode should be either 'set' or 'add'.")
        offset = tuple(int(value) for value in ([0] * self.dim_r if ind_R is None else ind_R))
        target, source = int(ind_i), int(ind_j)
        self._validate_hopping_key(target, source, offset)
        key = (target, source, offset)
        if callable(hop_amp) or isinstance(hop_amp, str):
            self._hopping_providers[key] = hop_amp
            return
        block = self._value_to_block(hop_amp, onsite=False)
        self._store_hopping(
            block, target, source, offset, mode, bool(allow_conjugate_pair)
        )
        self._hopping_providers.pop(key, None)

    def set_shell_hops(self, shell_hops, mode="set"):
        """Assign one hopping value to every bond in selected radial shells."""
        if not isinstance(shell_hops, dict):
            raise TypeError(
                "shell_hops must be a dictionary mapping shell index to hopping amplitude."
            )
        if not shell_hops:
            raise ValueError("shell_hops must have at least one element.")
        if any(
            not isinstance(shell, (int, np.integer)) or int(shell) < 1
            for shell in shell_hops
        ):
            raise ValueError("Each shell index must be a positive integer.")
        _, bonds_by_shell = self.nn_bonds(max(int(shell) for shell in shell_hops))
        for shell_index, bonds in enumerate(bonds_by_shell, start=1):
            if shell_index not in shell_hops:
                continue
            for source, target, offset in bonds:
                self.set_hop(
                    shell_hops[shell_index],
                    source,
                    target,
                    offset,
                    mode=mode,
                    allow_conjugate_pair=True,
                )

    def set_parameters(self, params=None, /, **kwargs):
        values = {} if params is None else dict(params)
        values.update(kwargs)
        for index, provider in list(self._onsite_providers.items()):
            names = _provider_names(provider)
            if all(name in values for name in names):
                self.set_onsite(self._evaluate(provider, values), index)
        for key, provider in list(self._hopping_providers.items()):
            names = _provider_names(provider)
            if all(name in values for name in names):
                target, source, offset = key
                block = self._value_to_block(self._evaluate(provider, values), onsite=False)
                self._store_hopping(block, target, source, offset)
                del self._hopping_providers[key]

    def with_parameters(self, params=None, /, **kwargs):
        """Return an independent model with supplied parameters resolved."""
        values = {} if params is None else dict(params)
        values.update(kwargs)
        result = self.copy()
        if values:
            result.set_parameters(values)
        return result

    def _resolved_terms(self, parameters):
        onsite = self._site_energies.astype(complex, copy=True)
        for index, provider in self._onsite_providers.items():
            block = self._value_to_block(
                self._evaluate(provider, parameters), onsite=True
            )
            onsite[index] = block if self.spinful else block[0, 0]
        hoppings = [dict(term, amplitude=term["amplitude"].copy()) for term in self._hoppings]
        for (target, source, offset), provider in self._hopping_providers.items():
            block = self._value_to_block(
                self._evaluate(provider, parameters), onsite=False
            )
            match = next(
                (
                    term
                    for term in hoppings
                    if term["target"] == target
                    and term["source"] == source
                    and term["offset"] == offset
                ),
                None,
            )
            if match is None:
                hoppings.append(
                    {
                        "amplitude": block,
                        "target": target,
                        "source": source,
                        "offset": offset,
                    }
                )
            else:
                match["amplitude"] = block
        return onsite, hoppings

    def _backend_args(self, parameters):
        onsite, hoppings = self._resolved_terms(parameters)
        if self.spinful:
            onsite_blocks = onsite.tolist()
        else:
            onsite_blocks = [[[complex(value)]] for value in onsite]
        hopping_data = [
            (
                term["target"],
                term["source"],
                list(term["offset"]),
                term["amplitude"].tolist(),
            )
            for term in hoppings
        ]
        return (
            self.lat_vecs.tolist(),
            self.periodic_dirs,
            self.orb_vecs.tolist(),
            [self.nspin] * self.norb,
            onsite_blocks,
            hopping_data,
        )

    def _normalize_kpoints(self, k_pts):
        if self.dim_k == 0:
            if k_pts is not None and np.asarray(k_pts).size:
                raise ValueError("k_pts must not be supplied for finite systems.")
            return [np.empty(0)]
        if k_pts is None:
            raise ValueError("k_pts must be specified for periodic systems.")
        array = np.asarray(k_pts, dtype=float)
        if array.ndim == 0:
            if self.dim_k != 1:
                raise ValueError("k-point has wrong dimension.")
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(-1, 1) if self.dim_k == 1 else array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.dim_k:
            raise ValueError("k-points must have shape (Nk, dim_k).")
        return list(array)

    def _parameter_axes(self, parameters):
        scalar = {}
        names = []
        axes = []
        for name, value in parameters.items():
            array = np.asarray(value)
            if array.ndim == 0:
                scalar[name] = array.item()
            elif array.ndim == 1:
                names.append(name)
                axes.append(list(array))
            else:
                raise ValueError("parameter values must be scalar or one-dimensional")
        combinations = [
            dict(scalar, **dict(zip(names, values, strict=True)))
            for values in (itertools.product(*axes) if axes else [()])
        ]
        return combinations, tuple(len(axis) for axis in axes)

    @staticmethod
    def _normalize_parameter_axis(values, *, name, period=None):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or values.size < 2:
            raise ValueError(f"Parameter axis {name!r} must contain at least two values.")
        differences = np.diff(values)
        step = float(differences[0])
        if not np.allclose(differences, step):
            raise ValueError(f"Parameter axis {name!r} must be uniformly spaced.")
        periodic = period is not None
        trimmed = False
        if period is not None and np.isclose(values[-1] - values[0], period):
            values = values[:-1]
            trimmed = True
        elif np.isclose(values[-1], values[0]):
            values = values[:-1]
            periodic = True
            trimmed = True
        return values, step, periodic, trimmed

    def hamiltonian(self, k_pts=None, flatten_spin_axis=False, **params):
        momenta = self._normalize_kpoints(k_pts)
        parameter_sets, parameter_shape = self._parameter_axes(params)
        matrices = []
        for momentum in momenta:
            per_k = []
            for parameters in parameter_sets:
                matrix = _core.hamiltonian(
                    *self._backend_args(parameters), list(momentum)
                )
                per_k.append(np.asarray(matrix, dtype=complex))
            matrices.append(np.asarray(per_k))
        result = np.asarray(matrices)
        result = result.reshape((len(momenta),) + parameter_shape + (self.nstate, self.nstate))
        if self.dim_k == 0:
            result = result[0]
        if self.spinful and not flatten_spin_axis:
            result = result.reshape(result.shape[:-2] + (self.norb, 2, self.norb, 2))
        return result

    def velocity(
        self,
        k_pts,
        cartesian=False,
        flatten_spin_axis=False,
        *,
        param_periods=None,
        diff_scheme="central",
        diff_order=2,
        **params,
    ):
        if diff_order < 1:
            raise ValueError("diff_order must be positive")
        if diff_scheme not in ("central", "forward"):
            raise ValueError("diff_scheme must be 'central' or 'forward'")
        momenta = self._normalize_kpoints(k_pts)
        parameter_sets, parameter_shape = self._parameter_axes(params)

        k_derivatives = []
        hamiltonians = []
        for momentum in momenta:
            per_k_derivatives = []
            per_k_hamiltonians = []
            for parameters in parameter_sets:
                arguments = self._backend_args(parameters)
                per_k_derivatives.append(
                    np.asarray(
                        _core.momentum_derivatives(
                            *arguments, list(momentum), bool(cartesian)
                        ),
                        dtype=complex,
                    )
                )
                per_k_hamiltonians.append(
                    np.asarray(
                        _core.hamiltonian(*arguments, list(momentum)), dtype=complex
                    )
                )
            k_derivatives.append(per_k_derivatives)
            hamiltonians.append(per_k_hamiltonians)

        direction_count = self.dim_r if cartesian else self.dim_k
        k_derivatives = np.asarray(k_derivatives).reshape(
            (len(momenta),) + parameter_shape + (direction_count, self.nstate, self.nstate)
        )
        k_derivatives = np.moveaxis(k_derivatives, -3, 0)
        hamiltonians = np.asarray(hamiltonians).reshape(
            (len(momenta),) + parameter_shape + (self.nstate, self.nstate)
        )
        components = [k_derivatives]

        sweep_names = []
        sweep_values = []
        for name, value in params.items():
            array = np.asarray(value)
            if array.ndim == 1:
                sweep_names.append(name)
                sweep_values.append(array.astype(float))
        periods = {} if param_periods is None else dict(param_periods)
        for sweep_index, (name, values) in enumerate(
            zip(sweep_names, sweep_values, strict=True)
        ):
            normalized, step, periodic, trimmed = self._normalize_parameter_axis(
                values, name=name, period=periods.get(name)
            )
            derivative = np.empty_like(hamiltonians)
            sample_axis = 1 + sweep_index
            fixed_shape = hamiltonians.shape[:sample_axis] + hamiltonians.shape[
                sample_axis + 1 : -2
            ]
            for fixed_index in np.ndindex(fixed_shape):
                before = fixed_index[:sample_axis]
                after = fixed_index[sample_axis:]
                selector = before + (slice(None),) + after + (slice(None), slice(None))
                samples = hamiltonians[selector]
                samples_for_difference = samples[:-1] if trimmed else samples
                if len(samples_for_difference) != len(normalized):
                    raise ValueError(f"Parameter axis {name!r} normalization mismatch")
                line = np.asarray(
                    _core.finite_difference(
                        samples_for_difference.tolist(),
                        step,
                        periodic,
                        diff_scheme,
                    ),
                    dtype=complex,
                )
                if trimmed and periodic:
                    line = np.concatenate([line, line[:1]], axis=0)
                derivative[selector] = line
            components.append(derivative[np.newaxis, ...])

        result = np.concatenate(components, axis=0)
        if self.spinful and not flatten_spin_axis:
            result = result.reshape(
                result.shape[:-2] + (self.norb, 2, self.norb, 2)
            )
        return result

    def solve_ham(
        self,
        k_pts=None,
        return_eigvecs=False,
        flatten_spin_axis=True,
        use_tensorflow=False,
        **params,
    ):
        if use_tensorflow:
            warnings.warn("Thouless uses its Rust eigensolver; use_tensorflow is ignored.")
        momenta = self._normalize_kpoints(k_pts)
        parameter_sets, parameter_shape = self._parameter_axes(params)
        values = []
        vectors = []
        for momentum in momenta:
            per_k_values = []
            per_k_vectors = []
            for parameters in parameter_sets:
                eigenvalues, eigenvectors = _core.eigensystem(
                    *self._backend_args(parameters), list(momentum)
                )
                per_k_values.append(eigenvalues)
                per_k_vectors.append(np.asarray(eigenvectors, dtype=complex).T)
            values.append(per_k_values)
            vectors.append(per_k_vectors)
        value_array = np.asarray(values).reshape(
            (len(momenta),) + parameter_shape + (self.nstate,)
        )
        vector_array = np.asarray(vectors).reshape(
            (len(momenta),) + parameter_shape + (self.nstate, self.nstate)
        )
        if self.dim_k == 0 or len(momenta) == 1:
            value_array = value_array[0]
            vector_array = vector_array[0]
        if self.spinful and not flatten_spin_axis:
            vector_array = vector_array.reshape(
                vector_array.shape[:-1] + (self.norb, self.nspin)
            )
        return (value_array, vector_array) if return_eigvecs else value_array

    def solve_one(self, k_list=None, eig_vectors=False):
        return self.solve_ham(
            k_list, return_eigvecs=eig_vectors, flatten_spin_axis=False
        )

    def solve_all(self, k_list=None, eig_vectors=False):
        return self.solve_ham(
            k_list, return_eigvecs=eig_vectors, flatten_spin_axis=False
        )

    def position_matrix(self, evecs, pos_dir):
        if pos_dir in self.periodic_dirs:
            raise ValueError(
                "Can not compute position matrix elements along periodic direction!"
            )
        if not 0 <= pos_dir < self.dim_r:
            raise ValueError("Direction out of range!")
        if not isinstance(evecs, np.ndarray):
            raise TypeError("evec must be a numpy array.")
        expected_dimension = 3 if self.spinful else 2
        if evecs.ndim != expected_dimension:
            raise ValueError(
                "evec has the wrong rank for the model's spin configuration"
            )
        states = evecs.reshape(evecs.shape[0], -1)
        positions = np.repeat(self.orb_vecs[:, pos_dir], self.nspin)
        return np.asarray(
            _core.diagonal_observable_matrix(
                states.tolist(),
                positions.tolist(),
            ),
            dtype=complex,
        )

    def position_expectation(self, evecs, pos_dir):
        matrix = self.position_matrix(evecs, pos_dir)
        return np.asarray(np.real(np.diag(matrix)), dtype=float)

    def position_hwf(
        self,
        evecs,
        pos_dir,
        hwf_evec=False,
        basis="orbital",
    ):
        position = self.position_matrix(evecs, pos_dir)
        centers, eigenvectors = _core.matrix_eigensystem(position.tolist())
        centers = np.asarray(centers, dtype=float)
        if not hwf_evec:
            return centers
        hybrid = np.asarray(eigenvectors, dtype=complex).T
        normalized_basis = basis.lower().strip()
        if normalized_basis in ("wavefunction", "bloch"):
            return centers, hybrid
        if normalized_basis != "orbital":
            raise ValueError(
                "Basis must be either 'wavefunction', 'bloch', or 'orbital'"
            )
        orbital_hybrid = hybrid @ evecs.reshape(evecs.shape[0], -1)
        if self.spinful:
            orbital_hybrid = orbital_hybrid.reshape(
                hybrid.shape[0], self.norb, self.nspin
            )
        return centers, orbital_hybrid

    def quantum_geometric_tensor(
        self,
        k_pts,
        occ_idxs=None,
        plane=None,
        *,
        cartesian=False,
        non_abelian=False,
        param_periods=None,
        diff_scheme="central",
        diff_order=2,
        use_tensorflow=False,
        **params,
    ):
        """Evaluate the occupied-subspace Kubo quantum geometric tensor."""
        if use_tensorflow:
            warnings.warn(
                "Thouless evaluates quantum geometry in its Rust core; "
                "use_tensorflow is ignored."
            )
        hamiltonians = np.asarray(
            self.hamiltonian(
                k_pts,
                flatten_spin_axis=True,
                **params,
            ),
            dtype=complex,
        )
        derivatives = np.asarray(
            self.velocity(
                k_pts,
                cartesian=cartesian,
                flatten_spin_axis=True,
                param_periods=param_periods,
                diff_scheme=diff_scheme,
                diff_order=diff_order,
                **params,
            ),
            dtype=complex,
        )
        if self.dim_k == 0 and derivatives.shape[1] == 1:
            derivatives = derivatives[:, 0]
        direction_count = derivatives.shape[0]
        if direction_count < 2:
            raise ValueError(
                "Quantum geometric tensor requires at least two independent "
                "coordinates (crystal momenta and/or varying parameters)."
            )
        sample_shape = hamiltonians.shape[:-2]
        if derivatives.shape[1:-2] != sample_shape:
            raise ValueError(
                "Hamiltonian and derivative sampling axes are inconsistent"
            )
        occupied = (
            np.arange(self.nstate // 2, dtype=int)
            if occ_idxs is None
            else np.atleast_1d(occ_idxs).astype(int)
        )
        if (
            len(occupied) == 0
            or len(np.unique(occupied)) != len(occupied)
            or np.any(occupied < 0)
            or np.any(occupied >= self.nstate)
            or len(occupied) == self.nstate
        ):
            raise ValueError(
                "occ_idxs must contain unique valid occupied states and "
                "leave at least one empty state"
            )

        derivative_groups = np.moveaxis(derivatives, 0, -3)
        tensor = np.asarray(
            _core.quantum_geometric_tensor_kubo(
                hamiltonians.reshape(
                    -1,
                    self.nstate,
                    self.nstate,
                ).tolist(),
                derivative_groups.reshape(
                    -1,
                    direction_count,
                    self.nstate,
                    self.nstate,
                ).tolist(),
                occupied.tolist(),
            ),
            dtype=complex,
        )
        tensor = tensor.reshape(
            sample_shape
            + (
                direction_count,
                direction_count,
                len(occupied),
                len(occupied),
            )
        )
        tensor = np.moveaxis(tensor, (-4, -3), (0, 1))
        if not non_abelian:
            tensor = np.trace(tensor, axis1=-2, axis2=-1)
        if plane is None:
            return tensor
        if not isinstance(plane, tuple) or len(plane) != 2:
            raise ValueError("plane must be a tuple of length 2.")
        return tensor[plane]

    def berry_curvature(
        self,
        k_pts,
        occ_idxs=None,
        plane=None,
        *,
        cartesian=False,
        non_abelian=False,
        param_periods=None,
        diff_scheme="central",
        diff_order=2,
        use_tensorflow=False,
        **params,
    ):
        """Evaluate Kubo Berry curvature for an occupied subspace."""
        tensor = self.quantum_geometric_tensor(
            k_pts,
            occ_idxs=occ_idxs,
            cartesian=cartesian,
            non_abelian=non_abelian,
            param_periods=param_periods,
            diff_scheme=diff_scheme,
            diff_order=diff_order,
            use_tensorflow=use_tensorflow,
            **params,
        )
        if non_abelian:
            curvature = 1j * (
                tensor - np.swapaxes(tensor, -1, -2).conj()
            )
        else:
            curvature = -2 * tensor.imag
        if plane is None:
            return curvature
        if not isinstance(plane, tuple) or len(plane) != 2:
            raise ValueError("plane must be a tuple of length 2.")
        return curvature[plane]

    def quantum_metric(
        self,
        k_pts,
        occ_idxs=None,
        plane=None,
        *,
        cartesian=False,
        non_abelian=False,
        param_periods=None,
        diff_scheme="central",
        diff_order=2,
        use_tensorflow=False,
        **params,
    ):
        """Evaluate the Kubo quantum metric for an occupied subspace."""
        tensor = self.quantum_geometric_tensor(
            k_pts,
            occ_idxs=occ_idxs,
            cartesian=cartesian,
            non_abelian=non_abelian,
            param_periods=param_periods,
            diff_scheme=diff_scheme,
            diff_order=diff_order,
            use_tensorflow=use_tensorflow,
            **params,
        )
        if non_abelian:
            metric = 0.5 * (
                tensor + np.swapaxes(tensor, -1, -2)
            )
        else:
            metric = tensor.real
        if plane is None:
            return metric
        if not isinstance(plane, tuple) or len(plane) != 2:
            raise ValueError("plane must be a tuple of length 2.")
        return metric[plane]

    def axion_angle(
        self,
        nks=(20, 20, 20),
        occ_idxs=None,
        return_second_chern=False,
        *,
        param_periods=None,
        diff_scheme="central",
        diff_order=4,
        use_tensorflow=False,
        **params,
    ):
        """Evaluate an axion sweep from the non-Abelian Kubo curvature."""

        if self.dim_k != 3:
            raise ValueError(
                "axion_angle requires a three-dimensional periodic model"
            )
        sizes = tuple(int(size) for size in nks)
        if len(sizes) != 3 or min(sizes) < 3:
            raise ValueError(
                "nks must contain three grid sizes of at least three"
            )
        if diff_scheme not in ("central", "forward"):
            raise ValueError("diff_scheme must be 'central' or 'forward'")
        if diff_order < 1:
            raise ValueError("diff_order must be positive")
        if use_tensorflow:
            warnings.warn(
                "Thouless evaluates second Chern response in its Rust core; "
                "use_tensorflow is ignored."
            )

        scalar_params = {}
        sweep_params = []
        for name, value in params.items():
            array = np.asarray(value)
            if array.ndim == 0:
                scalar_params[name] = array.item()
            elif array.ndim == 1:
                sweep_params.append((name, array.astype(float)))
            else:
                raise ValueError(
                    "axion_angle parameters must be scalar or one-dimensional"
                )
        if len(sweep_params) != 1:
            raise ValueError(
                "axion_angle requires exactly one swept parameter"
            )
        sweep_name, raw_values = sweep_params[0]
        periods = {} if param_periods is None else dict(param_periods)
        sweep_values, sweep_step, periodic, trimmed = (
            self._normalize_parameter_axis(
                raw_values,
                name=sweep_name,
                period=periods.get(sweep_name),
            )
        )
        if len(sweep_values) < 3:
            raise ValueError(
                "axion_angle requires at least three parameter samples"
            )

        occupied = (
            np.arange(self.nstate // 2, dtype=int)
            if occ_idxs is None
            else np.atleast_1d(occ_idxs).astype(int)
        )
        if (
            len(occupied) == 0
            or len(set(occupied.tolist())) != len(occupied)
            or np.any(occupied < 0)
            or np.any(occupied >= self.nstate)
        ):
            raise ValueError("occ_idxs must contain unique valid state indices")

        axes = [
            np.arange(size, dtype=float) / size for size in sizes
        ]
        momenta = np.stack(
            np.meshgrid(*axes, indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        solve_params = dict(scalar_params)
        solve_params[sweep_name] = sweep_values
        hamiltonians = self.hamiltonian(
            momenta,
            flatten_spin_axis=True,
            **solve_params,
        )
        derivatives = self.velocity(
            momenta,
            flatten_spin_axis=True,
            param_periods=periods,
            diff_scheme=diff_scheme,
            diff_order=diff_order,
            **solve_params,
        )
        derivative_groups = np.moveaxis(
            np.asarray(derivatives, dtype=complex),
            0,
            -3,
        )
        slice_density, second_chern = _core.second_chern_kubo(
            np.asarray(hamiltonians, dtype=complex).reshape(
                -1,
                self.nstate,
                self.nstate,
            ).tolist(),
            derivative_groups.reshape(
                -1,
                4,
                self.nstate,
                self.nstate,
            ).tolist(),
            [*sizes, len(sweep_values)],
            [
                1.0 / sizes[0],
                1.0 / sizes[1],
                1.0 / sizes[2],
                float(sweep_step),
            ],
            bool(periodic),
            occupied.tolist(),
        )
        slice_density = np.asarray(slice_density, dtype=float)
        theta = np.zeros(len(sweep_values), dtype=float)
        if len(theta) > 1:
            theta[1:] = (
                2.0
                * np.pi
                * sweep_step
                * np.cumsum(
                    0.5 * (slice_density[:-1] + slice_density[1:])
                )
            )
        output_values = np.asarray(sweep_values, dtype=float)
        if periodic and trimmed:
            closing_theta = theta[-1] + (
                2.0
                * np.pi
                * sweep_step
                * 0.5
                * (slice_density[-1] + slice_density[0])
            )
            output_values = np.append(
                output_values,
                output_values[0] + sweep_step * len(sweep_values),
            )
            theta = np.append(theta, closing_theta)
        theta = np.unwrap(theta, period=2.0 * np.pi)
        result = (output_values, theta)
        if return_second_chern:
            result += (float(second_chern),)
        return result

    def chern_number(
        self,
        plane,
        nks,
        occ_idxs=None,
        *,
        param_periods=None,
        diff_scheme="central",
        diff_order=4,
        use_tensorflow=False,
        **params,
    ):
        """Compute a first Chern number from Rust-evaluated occupied subspaces."""
        sizes = tuple(int(size) for size in nks)
        if len(sizes) != self.dim_k:
            raise ValueError("nks must provide one entry per periodic direction.")
        if not isinstance(plane, tuple) or len(plane) != 2:
            raise ValueError("plane must be a tuple of two axis indices.")
        first, second = (int(plane[0]), int(plane[1]))
        if first == second:
            raise ValueError("Chern number plane indices must be different.")
        occupied = (
            list(range(self.nstate // 2))
            if occ_idxs is None
            else np.atleast_1d(occ_idxs).astype(int).tolist()
        )
        if use_tensorflow:
            warnings.warn(
                "Thouless evaluates Chern numbers in its Rust core; "
                "use_tensorflow is ignored."
            )
        if diff_scheme not in ("central", "forward"):
            raise ValueError("diff_scheme must be 'central' or 'forward'")
        if diff_order < 1:
            raise ValueError("diff_order must be positive")

        has_sweep = any(np.asarray(value).ndim != 0 for value in params.values())
        if not has_sweep:
            values, spectator_shape = _core.uniform_grid_chern(
                *self._backend_args(params),
                list(sizes),
                [first, second],
                occupied,
            )
            result = np.asarray(values, dtype=float)
            if spectator_shape:
                return result.reshape(tuple(spectator_shape))
            return float(result[0])

        return self._chern_number_with_parameter_sweeps(
            (first, second),
            sizes,
            occupied,
            {} if param_periods is None else dict(param_periods),
            params,
        )

    def local_chern_marker(
        self,
        occ_idxs=None,
        return_bulk_avg=False,
        trim_cells=4,
        **params,
    ):
        """Evaluate the Bianco-Resta marker for a finite two-dimensional model."""
        if self.dim_k != 0:
            raise ValueError(
                "Local Chern marker is only defined for real-space models (dim_k=0)."
            )
        if self.dim_r != 2:
            raise NotImplementedError(
                "Local Chern marker is only defined for 2D models (dim_r=2)."
            )
        occupied = (
            np.arange(self.nstate // 2, dtype=int)
            if occ_idxs is None
            else np.atleast_1d(occ_idxs).astype(int)
        )
        if (
            len(occupied) == 0
            or len(np.unique(occupied)) != len(occupied)
            or np.any(occupied < 0)
            or np.any(occupied >= self.nstate)
        ):
            raise ValueError("occ_idxs must contain unique valid state indices")
        hamiltonian = np.asarray(
            self.hamiltonian(flatten_spin_axis=True, **params),
            dtype=complex,
        )
        orbital_positions = self.get_orb_vecs(cartesian=True)
        state_positions = np.repeat(
            orbital_positions,
            self.nspin,
            axis=0,
        )
        state_marker = np.asarray(
            _core.local_chern_marker_kubo(
                hamiltonian.tolist(),
                state_positions.tolist(),
                occupied.tolist(),
                float(self.cell_volume),
            ),
            dtype=float,
        )
        local_marker = state_marker.reshape(self.norb, self.nspin).sum(axis=1)
        if not return_bulk_avg:
            return local_marker

        if isinstance(trim_cells, (int, np.integer)):
            trim = (int(trim_cells), int(trim_cells))
        else:
            trim = tuple(int(value) for value in trim_cells)
            if len(trim) != 2:
                raise ValueError("trim_cells must be an integer or a pair")
        if any(value < 0 for value in trim):
            raise ValueError("trim_cells must be nonnegative")
        reduced = self.get_orb_vecs(cartesian=False)
        cell_indices = np.floor(reduced + 1.0e-9).astype(int)
        per_cell = {}
        for cell, marker in zip(
            map(tuple, cell_indices),
            local_marker,
            strict=True,
        ):
            per_cell[cell] = per_cell.get(cell, 0.0) + float(marker)
        coordinates = np.asarray(list(per_cell), dtype=int)
        lower = coordinates.min(axis=0) + np.asarray(trim)
        upper = coordinates.max(axis=0) - np.asarray(trim)
        if np.any(lower > upper):
            raise ValueError(
                f"trim_cells={trim_cells} is too large for the finite sample"
            )
        bulk_values = [
            marker
            for cell, marker in per_cell.items()
            if all(
                lower[axis] <= cell[axis] <= upper[axis]
                for axis in range(2)
            )
        ]
        if not bulk_values:
            raise ValueError("No bulk cells remain after trimming")
        return local_marker, float(np.mean(bulk_values))

    def _chern_number_with_parameter_sweeps(
        self,
        plane,
        sizes,
        occupied,
        param_periods,
        params,
    ):
        from .mesh import Mesh
        from .wfarray import WFArray

        scalar_params = {}
        sweep_names = []
        sweep_values = []
        for name, value in params.items():
            array = np.asarray(value)
            if array.ndim == 0:
                scalar_params[name] = array.item()
            elif array.ndim == 1 and array.size >= 2:
                sweep_names.append(name)
                sweep_values.append(array.astype(float))
            else:
                raise ValueError(
                    f"Swept parameter {name!r} must be a 1D array "
                    "with at least two samples."
                )
        if scalar_params:
            model = self.copy().set_parameters(scalar_params)
        else:
            model = self

        coordinate_axes = [
            np.arange(size, dtype=float) / size for size in sizes
        ] + sweep_values
        grids = np.meshgrid(*coordinate_axes, indexing="ij")
        points = np.stack(grids, axis=-1)
        mesh = Mesh(
            ["k"] * self.dim_k + ["l"] * len(sweep_names),
            axis_names=[f"k_{axis}" for axis in range(self.dim_k)] + sweep_names,
            dim_k=self.dim_k,
        )
        mesh.build_custom(points)
        for axis in range(self.dim_k):
            mesh.loop(axis, axis, winds_bz=True)
        for parameter_index, (name, values) in enumerate(
            zip(sweep_names, sweep_values, strict=True)
        ):
            if name not in param_periods:
                continue
            axis = self.dim_k + parameter_index
            component = self.dim_k + parameter_index
            period = float(param_periods[name])
            duplicate_endpoint = np.isclose(values[-1] - values[0], period) or np.isclose(
                values[-1], values[0]
            )
            mesh.loop(axis, component, closed=bool(duplicate_endpoint))

        wavefunctions = WFArray(
            model.lattice,
            mesh,
            spinful=model.spinful,
        )
        wavefunctions.solve_model(model)
        return wavefunctions.chern_number(plane=plane, state_idx=occupied)

    def k_uniform_mesh(
        self,
        mesh_size,
        *,
        gamma_centered=False,
        include_endpoints=True,
    ):
        sizes = tuple(int(size) for size in mesh_size)
        if len(sizes) != self.dim_k:
            raise ValueError("mesh_size must have one entry per periodic direction")
        return self._lattice.k_uniform_mesh(
            sizes,
            gamma_centered=gamma_centered,
            include_endpoints=include_endpoints,
        )

    def k_path(self, k_nodes, nk, report=False):
        return self._lattice.k_path(k_nodes, nk, report)

    def _replace_from_backend(self, data):
        (
            primitive_vectors,
            periodic_axes,
            orbital_positions,
            degrees_of_freedom,
            onsite_blocks,
            hoppings,
        ) = data
        if any(int(degrees) != self.nspin for degrees in degrees_of_freedom):
            raise ValueError(
                "Transformed model has internal degrees of freedom "
                "incompatible with the PythTB spin convention."
            )
        self._lattice = Lattice(
            primitive_vectors,
            orbital_positions,
            periodic_axes,
        )
        blocks = np.asarray(onsite_blocks, dtype=complex)
        self._site_energies = (
            blocks
            if self.spinful
            else np.asarray(blocks[:, 0, 0].real, dtype=float)
        )
        self._hoppings = [
            {
                "target": int(target),
                "source": int(source),
                "offset": tuple(int(value) for value in offset),
                "amplitude": np.asarray(amplitude, dtype=complex),
            }
            for target, source, offset, amplitude in hoppings
        ]

    def add_orb(self, orb_pos):
        """Append an empty orbital to the model in place."""
        self._lattice.add_orb(orb_pos)
        if self.spinful:
            self._site_energies = np.concatenate(
                (
                    self._site_energies,
                    np.zeros((1, 2, 2), dtype=complex),
                ),
                axis=0,
            )
        else:
            self._site_energies = np.append(self._site_energies, 0.0)

    def remove_orb(self, to_remove):
        """Remove orbital subspaces and compact all remaining hopping indices."""
        if isinstance(to_remove, int):
            indices = [to_remove]
        elif isinstance(to_remove, (list, np.ndarray)):
            indices = list(to_remove)
        else:
            raise TypeError("to_remove must be an integer or a list of integers.")
        for index in indices:
            if not isinstance(index, int):
                raise TypeError("All indices in to_remove must be integers.")
            if index < 0 or index >= self.norb:
                raise ValueError("Index out of bounds.")
        if len(indices) != len(set(indices)):
            raise ValueError("All indices in to_remove must be unique.")
        if self._onsite_providers or self._hopping_providers:
            raise ValueError(
                "Resolve parameter-dependent terms with set_parameters before "
                "removing orbitals."
            )
        transformed = _core.remove_model_orbitals(
            *self._backend_args({}),
            indices,
        )
        self._replace_from_backend(transformed)

    def change_nonperiodic_vector(
        self,
        fin_dir,
        new_latt_vec=None,
        to_home=True,
    ):
        """Change one open real-space basis vector without moving the geometry."""
        if not isinstance(fin_dir, int):
            raise TypeError("Argument fin_dir must be an integer")
        if self._onsite_providers or self._hopping_providers:
            raise ValueError(
                "Resolve parameter-dependent terms with set_parameters before "
                "changing lattice vectors."
            )
        replacement = (
            None
            if new_latt_vec is None
            else np.asarray(new_latt_vec, dtype=float).tolist()
        )
        transformed = _core.change_model_nonperiodic_vector(
            *self._backend_args({}),
            fin_dir,
            bool(to_home),
            replacement,
        )
        self._replace_from_backend(transformed)

    def make_supercell(
        self,
        sc_red_lat,
        return_sc_vectors=False,
        to_home=True,
    ):
        """Return a commensurate model generated by a Rust core transformation."""
        integer_basis = np.asarray(sc_red_lat)
        if integer_basis.shape != (self.dim_r, self.dim_r):
            raise ValueError("Dimension of sc_red_lat array must be dim_r*dim_r")
        if not np.issubdtype(integer_basis.dtype, np.integer):
            raise TypeError("sc_red_lat array elements must be integers")
        if self._onsite_providers or self._hopping_providers:
            raise ValueError(
                "Resolve parameter-dependent terms with set_parameters before "
                "constructing a supercell."
            )
        transformed, translations = _core.make_model_supercell(
            *self._backend_args({}),
            integer_basis.astype(int).tolist(),
            bool(to_home),
        )
        result = self.copy()
        result._replace_from_backend(transformed)
        translations = np.asarray(translations, dtype=int)
        return (result, translations) if return_sc_vectors else result

    def cut_piece(
        self,
        num_cells,
        periodic_dir,
        glue_edges=False,
        *,
        glue_edgs=None,
    ):
        if glue_edgs is not None:
            glue_edges = glue_edgs
        new_lattice = self._lattice.cut_piece(num_cells, periodic_dir)
        result = TBModel(new_lattice, self.spinful)
        result._from_w90 = self._from_w90
        result.assume_position_operator_diagonal = (
            self.assume_position_operator_diagonal
        )
        for cell in range(num_cells):
            for orbital in range(self.norb):
                result.set_onsite(
                    self._site_energies[orbital],
                    cell * self.norb + orbital,
                )
        for term in self._hoppings:
            shift = term["offset"][periodic_dir]
            for cell in range(num_cells):
                source_cell = cell + shift
                if glue_edges:
                    source_cell %= num_cells
                elif not 0 <= source_cell < num_cells:
                    continue
                offset = list(term["offset"])
                offset[periodic_dir] = 0
                result.set_hop(
                    term["amplitude"]
                    if self.spinful
                    else term["amplitude"][0, 0],
                    cell * self.norb + term["target"],
                    source_cell * self.norb + term["source"],
                    offset,
                    mode="add",
                    allow_conjugate_pair=True,
                )
        for cell in range(num_cells):
            for orbital, provider in self._onsite_providers.items():
                result.set_onsite(provider, cell * self.norb + orbital)
        for (target, source, original_offset), provider in (
            self._hopping_providers.items()
        ):
            shift = original_offset[periodic_dir]
            for cell in range(num_cells):
                source_cell = cell + shift
                if glue_edges:
                    source_cell %= num_cells
                elif not 0 <= source_cell < num_cells:
                    continue
                offset = list(original_offset)
                offset[periodic_dir] = 0
                result.set_hop(
                    provider,
                    cell * self.norb + target,
                    source_cell * self.norb + source,
                    offset,
                    allow_conjugate_pair=True,
                )
        return result

    def make_finite(self, periodic_dirs, num_cells, glue_edges=None):
        """Cut one or more periodic directions into a finite sample."""
        directions = list(periodic_dirs)
        cell_counts = list(num_cells)
        if self.dim_k == 0:
            raise ValueError("Model is already finite!")
        if len(directions) != len(cell_counts):
            raise ValueError(
                "Length of periodic_dirs must match length of num_cells."
            )
        if len(set(directions)) != len(directions):
            raise ValueError("All directions in periodic_dirs must be unique.")
        if any(direction not in self.periodic_dirs for direction in directions):
            raise ValueError("All directions in periodic_dirs must be periodic.")
        if any(
            not isinstance(count, (int, np.integer)) or int(count) < 1
            for count in cell_counts
        ):
            raise ValueError("Each num_cells entry must be a positive integer.")
        if glue_edges is None:
            glue = [False] * len(directions)
        else:
            glue = list(glue_edges)
            if len(glue) != len(directions):
                raise ValueError(
                    "Length of glue_edges must match number of periodic directions."
                )
        result = self
        for direction, count, glue_direction in zip(
            directions,
            cell_counts,
            glue,
            strict=True,
        ):
            result = result.cut_piece(
                int(count),
                int(direction),
                glue_edges=bool(glue_direction),
            )
        return result

    def reduce_dim(self):
        """Deprecated PythTB 2.0 placeholder retained verbatim."""
        warnings.warn(
            "reduce_dim() is deprecated; use make_finite() or cut_piece()",
            FutureWarning,
            stacklevel=2,
        )
        return None

    def visualize(
        self,
        proj_plane=None,
        eig_dr=None,
        draw_hoppings=True,
        annotate_onsite=False,
        ph_color="black",
    ):
        """Draw orbitals, lattice vectors, hoppings, and an optional state."""
        from .visualization import plot_tbmodel

        return plot_tbmodel(
            self,
            proj_plane=proj_plane,
            eig_dr=eig_dr,
            draw_hoppings=draw_hoppings,
            annotate_onsite=annotate_onsite,
            ph_color=ph_color,
        )

    def visualize_3d(
        self,
        draw_hoppings=True,
        show_model_info=True,
        site_colors=None,
        site_names=None,
        show=True,
    ):
        """Build an interactive three-dimensional model figure."""
        from .visualization import plot_tbmodel_3d

        return plot_tbmodel_3d(
            self,
            draw_hoppings=draw_hoppings,
            show_model_info=show_model_info,
            site_colors=site_colors,
            site_names=site_names,
            show=show,
        )

    def plot_bands(
        self,
        k_nodes,
        k_node_labels=None,
        nk=101,
        fig=None,
        ax=None,
        proj_orb_idx=None,
        proj_spin=False,
        bands_label=None,
        scat_size=3,
        lw=2,
        lc="b",
        ls="solid",
        cmap="plasma",
        cbar=True,
    ):
        """Plot bands along a reciprocal-space path."""
        from .visualization import plot_bands

        return plot_bands(
            self,
            k_nodes,
            nk=nk,
            ktick_labels=k_node_labels,
            bands_label=bands_label,
            proj_orb_idx=proj_orb_idx,
            proj_spin=proj_spin,
            fig=fig,
            ax=ax,
            scat_size=scat_size,
            lw=lw,
            lc=lc,
            ls=ls,
            cmap=cmap,
            cbar=cbar,
        )


class tb_model(TBModel):
    """Deprecated PythTB 1.x constructor retained by PythTB 2.0."""

    def __init__(self, dim_k, dim_r, lat=None, orb=None, per=None, nspin=1):
        if lat is None or (isinstance(lat, str) and lat == "unit"):
            lat = np.eye(dim_r)
        if orb is None or (isinstance(orb, str) and orb == "bravais"):
            orb = np.zeros((1, dim_r))
        elif isinstance(orb, (int, np.integer)):
            orb = np.zeros((int(orb), dim_r))
        if per is None:
            per = list(range(dim_k))
        if len(per) != dim_k:
            raise ValueError("len(per) must equal dim_k")
        super().__init__(Lattice(lat, orb, per), spinful=nspin == 2)


__all__ = ["TBModel", "tb_model"]
