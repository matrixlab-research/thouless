"""Wavefunction-mesh compatibility for PythTB."""

from __future__ import annotations

import numpy as np

from thouless import _core

from .lattice import Lattice
from .mesh import Mesh
from .tbmodel import TBModel


class WFArray:
    """Store Rust-computed eigenstates on a PythTB sampling mesh."""

    def __init__(self, lattice, mesh, nstates=None, spinful=False):
        if not isinstance(lattice, Lattice):
            raise TypeError("lattice must be of type pythtb.Lattice")
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be of type pythtb.Mesh")
        if lattice.dim_k != mesh.dim_k:
            raise ValueError(
                f"Lattice dim_k ({lattice.dim_k}) does not match Mesh dim_k ({mesh.dim_k})"
            )
        if not mesh.filled:
            raise ValueError("Mesh points are not initialized. Did you call build_grid?")
        short_loops = [
            index
            for index, axis in enumerate(mesh.axes)
            if axis.is_loop and axis.size < 2
        ]
        if short_loops:
            raise ValueError("Looping mesh axes must have at least two samples")
        self._lattice = lattice
        self._mesh = mesh
        self._spinful = bool(spinful)
        self._nspin = 2 if spinful else 1
        self._nstates = lattice.norb * self._nspin if nstates is None else int(nstates)
        self._wfs = np.empty(self.shape, dtype=complex)
        self._energies = None

    @property
    def lattice(self):
        return self._lattice

    @property
    def mesh(self):
        return self._mesh

    @property
    def spinful(self):
        return self._spinful

    @property
    def nspin(self):
        return self._nspin

    @property
    def norb(self):
        return self.lattice.norb

    @property
    def nstates(self):
        return self._nstates

    @property
    def naxes(self):
        return self.mesh.naxes

    @property
    def dim_k(self):
        return self.mesh.dim_k

    @property
    def shape_mesh(self):
        return self.mesh.shape_axes

    @property
    def shape(self):
        tail = (
            (self.nstates, self.norb, 2)
            if self.spinful
            else (self.nstates, self.norb)
        )
        return self.shape_mesh + tail

    @property
    def wfs(self):
        return self._wfs

    @property
    def energies(self):
        return self._energies

    def __getitem__(self, index):
        return self._wfs[index]

    def __setitem__(self, index, value):
        self._wfs[index] = np.asarray(value, dtype=complex)

    def _basis_phase(self, mesh_axis):
        phase = np.ones(self.norb, dtype=complex)
        axis = self.mesh.axes[mesh_axis]
        for component in axis.winds_bz_components:
            real_direction = self.lattice.periodic_dirs[component]
            phase *= np.exp(-2j * np.pi * self.lattice.orb_vecs[:, real_direction])
        return np.repeat(phase, self.nspin)

    def _enforce_closed_boundaries(self):
        flat = self._wfs.reshape(self.shape_mesh + (self.nstates, -1))
        for mesh_axis, axis in enumerate(self.mesh.axes):
            if not axis.has_endpoint:
                continue
            first = [slice(None)] * flat.ndim
            last = [slice(None)] * flat.ndim
            first[mesh_axis] = 0
            last[mesh_axis] = -1
            value = flat[tuple(first)]
            if axis.winds_bz_components:
                value = value * self._basis_phase(mesh_axis)
            flat[tuple(last)] = value

    def set_states(self, wfs, is_cell_periodic=True, is_spin_axis_flat=False):
        wfs = np.asarray(wfs, dtype=complex)
        expected = (
            self.shape_mesh + (self.nstates, self.norb * self.nspin)
            if is_spin_axis_flat and self.spinful
            else self.shape
        )
        if wfs.shape != expected:
            raise ValueError(
                f"wfs shape {wfs.shape} does not match expected shape {expected}"
            )
        self._wfs = wfs.reshape(self.shape)
        self._enforce_closed_boundaries()

    def states(self, state_idx=None, flatten_spin_axis=False, return_psi=False):
        if return_psi:
            raise NotImplementedError("full Bloch-state storage is not implemented yet")
        indices = (
            np.arange(self.nstates)
            if state_idx is None
            else np.atleast_1d(state_idx).astype(int)
        )
        if np.any(indices < 0) or np.any(indices >= self.nstates):
            raise IndexError("state index is outside the WFArray")
        selected = np.take(self._wfs, indices, axis=self.naxes)
        if self.spinful and flatten_spin_axis:
            selected = selected.reshape(
                self.shape_mesh + (len(indices), self.norb * self.nspin)
            )
        return selected

    def solve_model(self, model: TBModel, use_tensorflow=False):
        if not isinstance(model, TBModel):
            raise TypeError("model must be a pythtb.TBModel")
        if model.lattice != self.lattice or model.spinful != self.spinful:
            raise ValueError("model geometry and spin must match the WFArray")
        energies = np.empty(self.shape_mesh + (self.nstates,), dtype=float)
        for index in np.ndindex(self.shape_mesh):
            point = self.mesh.points[index]
            momentum = point[: self.mesh.dim_k]
            parameters = {
                axis.name: point[self.mesh.dim_k + parameter_index]
                for parameter_index, axis in enumerate(
                    axis for axis in self.mesh.axes if axis.type == "l"
                )
            }
            values, vectors = model.solve_ham(
                momentum,
                return_eigvecs=True,
                flatten_spin_axis=not self.spinful,
                use_tensorflow=use_tensorflow,
                **parameters,
            )
            energies[index] = values[: self.nstates]
            self._wfs[index] = vectors[: self.nstates]
        self._energies = energies
        self._model = model
        self._enforce_closed_boundaries()
        return energies

    def solve_on_grid(self, start_k=None):
        if not hasattr(self, "_model"):
            raise ValueError("legacy solve_on_grid requires a wf_array constructed from a model")
        return self.solve_model(self._model)

    def solve_on_one_point(self, kpt, mesh_indices):
        values, vectors = self._model.solve_ham(kpt, return_eigvecs=True)
        self._wfs[tuple(mesh_indices)] = vectors
        if self._energies is None:
            self._energies = np.empty(self.shape_mesh + (self.nstates,), dtype=float)
        self._energies[tuple(mesh_indices)] = values

    def berry_phase(
        self, axis_idx, state_idx=None, berry_evals=False, contin=True
    ):
        if berry_evals:
            raise NotImplementedError(
                "Wilson-loop eigenphases are tracked in "
                "https://github.com/matrixlab-research/thouless/issues/2"
            )
        if not 0 <= axis_idx < self.naxes:
            raise ValueError("axis_idx is outside the mesh")
        frames = self.states(state_idx, flatten_spin_axis=True)
        frames = np.moveaxis(frames, axis_idx, 0)
        transverse_shape = frames.shape[1:-2]
        output = np.empty(transverse_shape, dtype=float)
        for transverse in np.ndindex(transverse_shape):
            line = frames[(slice(None),) + transverse]
            axis = self.mesh.axes[axis_idx]
            if axis.is_loop and not axis.has_endpoint:
                closure = line[0] * self._basis_phase(axis_idx)
                line = np.concatenate([line, closure[np.newaxis]], axis=0)
            output[transverse] = _core.wilson_phase(line.tolist())
        if contin and output.ndim:
            for axis in range(output.ndim):
                output = np.unwrap(output, axis=axis)
        return output.item() if output.ndim == 0 else output

    def _frame_at(self, frames, index, crossed_axes):
        frame = frames[index].copy()
        for axis in crossed_axes:
            frame *= self._basis_phase(axis)
        return frame

    def berry_flux(
        self,
        plane=None,
        state_idx=None,
        non_abelian=False,
        *,
        use_tensorflow=False,
    ):
        if use_tensorflow:
            raise ValueError("Thouless topology always executes in the Rust core")
        if non_abelian:
            raise NotImplementedError(
                "non-Abelian flux matrices are tracked in "
                "https://github.com/matrixlab-research/thouless/issues/2"
            )
        if self.naxes < 2:
            raise ValueError("Berry flux requires at least two mesh axes")
        if plane is not None and (
            not isinstance(plane, (tuple, list, np.ndarray)) or len(plane) != 2
        ):
            if state_idx is not None:
                raise ValueError("ambiguous legacy berry_flux arguments")
            state_idx = plane
            plane = (0, 1)
        if plane is None:
            plane = (0, 1)
        first_axis, second_axis = (int(plane[0]), int(plane[1]))
        if (
            first_axis == second_axis
            or not 0 <= first_axis < self.naxes
            or not 0 <= second_axis < self.naxes
        ):
            raise ValueError("plane must contain two distinct mesh axes")

        frames = self.states(state_idx, flatten_spin_axis=True)
        output_shape = list(self.shape_mesh)
        for axis_index in (first_axis, second_axis):
            axis = self.mesh.axes[axis_index]
            if axis.has_endpoint or not axis.is_loop:
                output_shape[axis_index] -= 1
        output = np.empty(tuple(output_shape), dtype=float)

        for base in np.ndindex(*output_shape):
            corners = []
            for step_first, step_second in ((0, 0), (1, 0), (1, 1), (0, 1)):
                index = list(base)
                crossed = []
                for axis_index, step in (
                    (first_axis, step_first),
                    (second_axis, step_second),
                ):
                    if not step:
                        continue
                    index[axis_index] += 1
                    if index[axis_index] == self.shape_mesh[axis_index]:
                        index[axis_index] = 0
                        crossed.append(axis_index)
                corners.append(self._frame_at(frames, tuple(index), crossed).tolist())
            output[base] = _core.berry_flux(corners)
        return output

    def chern_number(self, plane=(0, 1), state_idx=None):
        flux = self.berry_flux(plane=plane, state_idx=state_idx)
        return np.sum(flux, axis=tuple(sorted(int(axis) for axis in plane))) / (
            2 * np.pi
        )


class wf_array(WFArray):
    """Deprecated PythTB 1.x wavefunction-array constructor."""

    def __init__(self, model, mesh_size, nsta_arr=None):
        mesh = Mesh(["k"] * model.dim_k, dim_k=model.dim_k)
        mesh.build_grid(mesh_size)
        super().__init__(
            model.lattice,
            mesh,
            nstates=nsta_arr,
            spinful=model.spinful,
        )
        self._model = model


__all__ = ["WFArray", "wf_array"]
