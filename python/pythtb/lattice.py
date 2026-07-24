"""PythTB lattice compatibility backed by Thouless geometry conventions."""

from __future__ import annotations

import copy

import numpy as np

from thouless import _core


class Lattice:
    """Real-space primitive vectors, orbitals, and selected periodic axes."""

    def __init__(self, lat_vecs, orb_vecs, periodic_dirs=[]):
        lat = np.asarray(lat_vecs, dtype=float)
        if lat.size == 0:
            lat = np.empty((0, 0), dtype=float)
        if lat.ndim != 2 or lat.shape[0] != lat.shape[1]:
            raise ValueError("Wrong lat array dimensions. Must have shape (dim_r, dim_r).")
        if lat.shape[0] > 3:
            raise ValueError("Argument dim_r must be from 0 to 3.")
        if lat.shape[0]:
            determinant = float(np.linalg.det(lat))
            if determinant < 0:
                raise ValueError("Lattice vectors need to form right handed system.")
            if determinant < 1e-10:
                raise ValueError("Volume of unit cell is zero.")
        self._lat_vectors = lat
        self._periodic_dirs: list[int] = []
        self.periodic_dirs = periodic_dirs
        self.orb_vecs = orb_vecs
        self._nsuper = [1] * self.dim_r

    def __copy__(self):
        result = type(self)(self.lat_vecs, self.orb_vecs, self.periodic_dirs)
        result._nsuper = self._nsuper.copy()
        return result

    def __eq__(self, other):
        return (
            isinstance(other, Lattice)
            and np.allclose(self.lat_vecs, other.lat_vecs)
            and np.allclose(self.orb_vecs, other.orb_vecs)
            and self.periodic_dirs == other.periodic_dirs
        )

    @property
    def dim_r(self):
        return self._lat_vectors.shape[0]

    @property
    def dim_k(self):
        return len(self._periodic_dirs)

    @property
    def norb(self):
        return self._orb_vectors.shape[0]

    @property
    def nsuper(self):
        return self._nsuper.copy()

    @property
    def lat_vecs(self):
        return self._lat_vectors.copy()

    @lat_vecs.setter
    def lat_vecs(self, value):
        replacement = type(self)(value, self.orb_vecs, self.periodic_dirs)
        self.__dict__.update(replacement.__dict__)

    @property
    def orb_vecs(self):
        return self._orb_vectors.copy()

    @orb_vecs.setter
    def orb_vecs(self, value):
        if isinstance(value, (int, np.integer)):
            if value < 0:
                raise ValueError("Number of orbitals must be positive.")
            array = np.zeros((int(value), self.dim_r), dtype=float)
        else:
            array = np.asarray(value, dtype=float)
            if self.dim_r == 0 and array.size == 0:
                array = np.empty((0, 0), dtype=float)
            if array.ndim != 2 or array.shape[1] != self.dim_r:
                raise ValueError(
                    "Wrong orb array dimensions. Must have shape (norb, dim_r)."
                )
        self._orb_vectors = array
        self._orb_vecs_cart = array @ self._lat_vectors

    @property
    def periodic_dirs(self):
        return self._periodic_dirs.copy()

    @periodic_dirs.setter
    def periodic_dirs(self, value):
        if value is ... or value == "all":
            value = list(range(self.dim_r))
        elif value is None:
            value = []
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise TypeError("periodic_dirs must be a list of integers.")
        result = []
        for index in value:
            if not isinstance(index, (int, np.integer)):
                raise TypeError("periodic_dirs entries must be integers.")
            index = int(index)
            if index < 0 or index >= self.dim_r:
                raise ValueError(
                    f"Periodic direction {index} is out of bounds for lattice dimension {self.dim_r}."
                )
            if index in result:
                raise ValueError("periodic_dirs entries must be unique.")
            result.append(index)
        self._periodic_dirs = result

    @property
    def cell_volume(self):
        return 0.0 if self.dim_r == 0 else abs(float(np.linalg.det(self.lat_vecs)))

    @property
    def recip_lat_vecs(self):
        if self.dim_k == 0:
            raise ValueError(
                "Reciprocal lattice vectors are not defined for zero-dimensional k-space."
            )
        periodic = self.lat_vecs[np.asarray(self.periodic_dirs)]
        gram = periodic @ periodic.T
        return 2 * np.pi * np.linalg.solve(gram, periodic)

    @property
    def recip_volume(self):
        reciprocal = self.recip_lat_vecs
        return float(np.sqrt(np.linalg.det(reciprocal @ reciprocal.T)))

    def get_lat_vecs(self):
        return self.lat_vecs

    def get_orb_vecs(self, cartesian=False):
        return self._orb_vecs_cart.copy() if cartesian else self.orb_vecs

    def get_recip_lat_vecs(self):
        return self.recip_lat_vecs

    def copy(self):
        return copy.copy(self)

    def k_path(self, k_nodes, nk, report=False):
        if isinstance(k_nodes, str):
            if self.dim_k != 1:
                raise ValueError("named k-paths are only defined in one dimension")
            presets = {
                "full": [[0.0], [0.5], [1.0]],
                "fullc": [[-0.5], [0.0], [0.5]],
                "half": [[0.0], [0.5]],
            }
            if k_nodes not in presets:
                raise ValueError(f"unknown one-dimensional k-path {k_nodes!r}")
            nodes = np.asarray(presets[k_nodes], dtype=float)
        else:
            nodes = np.asarray(k_nodes, dtype=float)
            if nodes.ndim == 1 and self.dim_k == 1:
                nodes = nodes[:, np.newaxis]
        if nodes.ndim != 2 or nodes.shape[1] != self.dim_k:
            raise ValueError(
                f"Dimension mismatch: kpts shape {nodes.shape}, model dim {self.dim_k}"
            )

        points, distances, node_distances = _core.reciprocal_path(
            self.lat_vecs.tolist(),
            self.periodic_dirs,
            nodes.tolist(),
            int(nk),
        )
        points = np.asarray(points, dtype=float)
        distances = np.asarray(distances, dtype=float)
        node_distances = np.asarray(node_distances, dtype=float)
        if report:
            print("----- k_path report -----")
            print("Real-space lattice vectors:\n", self.lat_vecs[self.periodic_dirs])
            print("Nodes (reduced coords):\n", nodes)
            print("Node distances (cumulative):", node_distances)
            print("-------------------------")
        return points, distances, node_distances

    def cut_piece(self, num_cells, periodic_dir):
        if not isinstance(num_cells, int):
            raise TypeError("Parameter `num_cells` is not an integer")
        if num_cells < 1:
            raise ValueError("Argument num_cells must be positive!")
        if periodic_dir not in self.periodic_dirs:
            raise Exception("Can not make model finite along this direction!")
        periodic = self.periodic_dirs
        periodic.remove(periodic_dir)
        orbitals = []
        for cell in range(num_cells):
            for orbital in self.orb_vecs:
                shifted = orbital.copy()
                shifted[periodic_dir] += cell
                orbitals.append(shifted)
        result = type(self)(self.lat_vecs, orbitals, periodic)
        result._nsuper = self._nsuper.copy()
        result._nsuper[periodic_dir] = num_cells
        return result


__all__ = ["Lattice"]
