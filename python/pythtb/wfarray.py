"""Wavefunction-mesh compatibility for PythTB."""

from __future__ import annotations

import numpy as np

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

    def berry_phase(self, *args, **kwargs):
        raise NotImplementedError(
            "Berry-phase compatibility awaits the Rust topology core: "
            "https://github.com/matrixlab-research/thouless/issues/2"
        )

    def berry_flux(self, *args, **kwargs):
        raise NotImplementedError(
            "Berry-flux compatibility awaits the Rust topology core: "
            "https://github.com/matrixlab-research/thouless/issues/2"
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
