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

    def set_parameters(self, parameters=None, **kwargs):
        values = {} if parameters is None else dict(parameters)
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
        return self

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

    def k_uniform_mesh(self, mesh_size):
        sizes = tuple(int(size) for size in mesh_size)
        if len(sizes) != self.dim_k:
            raise ValueError("mesh_size must have one entry per periodic direction")
        axes = [np.arange(size) / size for size in sizes]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, self.dim_k)

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
        return result


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
