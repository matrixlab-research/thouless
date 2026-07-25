"""PythTB lattice compatibility backed by Thouless geometry conventions."""

from __future__ import annotations

import copy
import itertools

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

    def info(self, show=True):
        """Return or print a compact report of the lattice geometry."""
        lines = [
            "----------------------------------------",
            "          Lattice report",
            "----------------------------------------",
            f"real-space dimension       = {self.dim_r}",
            f"reciprocal-space dimension = {self.dim_k}",
            f"periodic directions        = {self.periodic_dirs}",
            f"number of orbitals         = {self.norb}",
            "Lattice vectors (Cartesian):",
        ]
        lines.extend(
            f"  # {index} ===> {np.array2string(vector, precision=8)}"
            for index, vector in enumerate(self.lat_vecs)
        )
        lines.append("Orbital vectors (fractional):")
        lines.extend(
            f"  # {index} ===> {np.array2string(vector, precision=8)}"
            for index, vector in enumerate(self.orb_vecs)
        )
        lines.append("----------------------------------------")
        report = "\n".join(lines)
        if show:
            print(report)
            return None
        return report

    def copy(self):
        return copy.copy(self)

    def add_orb(self, orb_pos):
        """Append one orbital in reduced coordinates."""
        position = np.asarray(
            [orb_pos] if np.isscalar(orb_pos) else orb_pos,
            dtype=float,
        )
        if position.shape != (self.dim_r,):
            raise ValueError(f"Orbital position must be of length {self.dim_r}.")
        if not np.all(np.isfinite(position)):
            raise ValueError("Orbital position must be finite.")
        self.orb_vecs = np.vstack((self.orb_vecs, position))

    def remove_orb(self, to_remove):
        """Remove one or more orbitals in place."""
        if isinstance(to_remove, (int, np.integer)):
            indices = [int(to_remove)]
        elif isinstance(to_remove, (list, tuple, np.ndarray)):
            indices = list(to_remove)
        else:
            raise TypeError("to_remove must be an integer or a list of integers.")
        if any(not isinstance(index, (int, np.integer)) for index in indices):
            raise TypeError("All indices in to_remove must be integers.")
        indices = [int(index) for index in indices]
        if len(indices) != len(set(indices)):
            raise ValueError("All indices in to_remove must be unique.")
        if any(index < 0 or index >= self.norb for index in indices):
            raise ValueError("Index out of bounds.")
        self.orb_vecs = np.delete(self.orb_vecs, indices, axis=0)

    def change_nonperiodic_vector(self, fin_dir, new_lat_vec=None):
        """Change an open real-space basis vector without moving orbitals."""
        if not isinstance(fin_dir, (int, np.integer)):
            raise TypeError("Argument fin_dir must be an integer")
        fin_dir = int(fin_dir)
        if fin_dir < 0 or fin_dir >= self.dim_r or fin_dir in self.periodic_dirs:
            raise ValueError(f"Selected direction {fin_dir} is not nonperiodic")
        cartesian_orbitals = self.get_orb_vecs(cartesian=True)
        lattice = self.lat_vecs
        if new_lat_vec is None:
            candidate = lattice[fin_dir].copy()
            if self.periodic_dirs:
                periodic = lattice[np.asarray(self.periodic_dirs)]
                coefficients = np.linalg.lstsq(
                    periodic.T,
                    candidate,
                    rcond=None,
                )[0]
                candidate -= coefficients @ periodic
            norm = np.linalg.norm(candidate)
            if norm < 1e-10:
                raise ValueError("New nonperiodic vector has zero length.")
            candidate *= np.linalg.norm(lattice[fin_dir]) / norm
        else:
            candidate = np.asarray(new_lat_vec, dtype=float)
            if candidate.shape != (self.dim_r,):
                raise ValueError("Non-periodic vector has wrong shape.")
            if not np.all(np.isfinite(candidate)) or np.linalg.norm(candidate) < 1e-10:
                raise ValueError("New non-periodic vector has zero length.")
        lattice[fin_dir] = candidate
        determinant = float(np.linalg.det(lattice))
        if determinant < 1e-10:
            raise ValueError(
                "New lattice vectors must remain linearly independent and right handed."
            )
        reduced_orbitals = np.linalg.solve(lattice.T, cartesian_orbitals.T).T
        self._lat_vectors = lattice
        self.orb_vecs = reduced_orbitals

    def nn_bonds(self, n_shell, report=False):
        """Enumerate unique bonds in the shortest radial neighbor shells."""
        if not isinstance(n_shell, (int, np.integer)) or int(n_shell) < 1:
            raise ValueError("n_shell must be a positive integer.")
        if self.norb == 0 or self.dim_r == 0:
            raise ValueError("Nearest-neighbor shells require a nonempty lattice.")
        n_shell = int(n_shell)
        periodic = self.periodic_dirs
        periodic_vectors = self.lat_vecs[np.asarray(periodic)] if periodic else None
        singular_floor = (
            float(np.linalg.svd(periodic_vectors, compute_uv=False)[-1])
            if periodic
            else np.inf
        )
        cartesian_orbitals = self.get_orb_vecs(cartesian=True)
        orbital_diameter = max(
            (
                np.linalg.norm(left - right)
                for left in cartesian_orbitals
                for right in cartesian_orbitals
            ),
            default=0.0,
        )

        bound = max(1, n_shell)
        candidates = None
        unique_distances = None
        while bound <= 256:
            ranges = [
                range(-bound, bound + 1) if axis in periodic else (0,)
                for axis in range(self.dim_r)
            ]
            candidates = []
            for shift in itertools.product(*ranges):
                shift = np.asarray(shift, dtype=int)
                translation = shift @ self.lat_vecs
                for source in range(self.norb):
                    for target in range(self.norb):
                        if source == target and not np.any(shift):
                            continue
                        displacement = (
                            cartesian_orbitals[target]
                            + translation
                            - cartesian_orbitals[source]
                        )
                        distance_sq = round(
                            float(np.dot(displacement, displacement)),
                            12,
                        )
                        candidates.append(
                            (
                                distance_sq,
                                source,
                                target,
                                tuple(int(value) for value in shift),
                                displacement,
                            )
                        )
            unique_distances = sorted({entry[0] for entry in candidates})
            if len(unique_distances) < n_shell:
                bound *= 2
                continue
            shell_radius = np.sqrt(unique_distances[n_shell - 1])
            unseen_lower_bound = singular_floor * (bound + 1) - orbital_diameter
            if not periodic or unseen_lower_bound > shell_radius + 1e-10:
                break
            bound *= 2
        else:
            raise RuntimeError("Could not certify nearest-neighbor shell enumeration.")

        summaries = []
        bonds_by_shell = []
        for shell_index, distance_sq in enumerate(
            unique_distances[:n_shell],
            start=1,
        ):
            entries = [
                entry for entry in candidates if entry[0] == distance_sq
            ]
            seen = set()
            bonds = []
            orbital_data = {}
            for _, source, target, shift, displacement in entries:
                orbital_data.setdefault(
                    source,
                    {
                        "degeneracy": 0,
                        "neighbors": [],
                        "shifts": [],
                        "displacements": [],
                    },
                )
                data = orbital_data[source]
                data["degeneracy"] += 1
                data["neighbors"].append(target)
                data["shifts"].append(list(shift))
                data["displacements"].append(displacement.tolist())
                conjugate = (
                    target,
                    source,
                    tuple(-value for value in shift),
                )
                key = (source, target, shift)
                if conjugate not in seen and key not in seen:
                    seen.add(key)
                    bonds.append(key)
            summaries.append(
                {
                    "shell": shell_index,
                    "radius": float(np.sqrt(distance_sq)),
                    "distance_sq": float(distance_sq),
                    "degeneracy_total": len(entries),
                    "orbitals": orbital_data,
                }
            )
            bonds_by_shell.append(bonds)
        if report:
            for summary in summaries:
                print(
                    f"shell {summary['shell']}: "
                    f"radius={summary['radius']:.8g}, "
                    f"degeneracy={summary['degeneracy_total']}"
                )
        return summaries, bonds_by_shell

    def k_uniform_mesh(self, mesh_size):
        """Return a flattened uniform mesh over all periodic directions."""
        sizes = tuple(int(size) for size in mesh_size)
        if len(sizes) != self.dim_k or any(size < 1 for size in sizes):
            raise ValueError("mesh_size must contain one positive size per periodic direction")
        axes = [np.arange(size, dtype=float) / size for size in sizes]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
            -1,
            self.dim_k,
        )

    def get_kpath_distance(self, k_points):
        """Return cumulative Cartesian reciprocal-space path distance."""
        points = np.asarray(k_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.dim_k:
            raise ValueError("k_points must have shape (Nk, dim_k)")
        cartesian = points @ self.recip_lat_vecs
        distance = np.zeros(len(points), dtype=float)
        if len(points) > 1:
            distance[1:] = np.cumsum(
                np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
            )
        return distance

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
