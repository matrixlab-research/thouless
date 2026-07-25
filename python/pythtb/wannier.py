"""Wannier projection, localization, and interpolation compatibility."""

from __future__ import annotations

import numpy as np

from .utils import get_trial_wfs
from .wfarray import WFArray


def _polar_factor(matrix):
    left, _, right = np.linalg.svd(matrix, full_matrices=False)
    return left @ right


class Wannier:
    """Construct localized orbitals from a toroidal Bloch-state mesh."""

    def __init__(self, bloch_states):
        if not isinstance(bloch_states, WFArray):
            raise TypeError("bloch_states must be a pythtb.WFArray")
        if not bloch_states.mesh.is_k_torus:
            raise ValueError("Wannierization requires a toroidal k-space mesh.")
        if any(axis.has_endpoint for axis in bloch_states.mesh.k_axes):
            raise ValueError(
                "Wannierization requires non-duplicated Brillouin-zone endpoints."
            )
        self._wfa = bloch_states
        self._trial_wfs = None
        self._tilde_states = None
        self._wannier = None
        self._spread = None
        self._centers = None
        self._omega_i = None
        self._omega_d = None
        self._omega_od = None
        self._A = None
        cell_axes = [
            np.arange(-size // 2, size // 2)
            for size in self.nks
        ]
        self.supercell = np.stack(
            np.meshgrid(*cell_axes, indexing="ij"),
            axis=-1,
        ).reshape(-1, len(self.nks))

    @property
    def mesh(self):
        return self.bloch_states.mesh

    @property
    def lattice(self):
        return self.bloch_states.lattice

    @property
    def bloch_states(self):
        return self._wfa

    @property
    def tilde_states(self):
        if self._tilde_states is None:
            raise ValueError(
                "Bloch-like states are not set; call project() or set_tilde_states()."
            )
        return self._tilde_states

    @property
    def nks(self):
        return self.mesh.shape_k

    @property
    def wannier(self):
        if self._wannier is None:
            raise ValueError("Wannier functions are not initialized.")
        return self._wannier

    @property
    def spread(self):
        if self._spread is None:
            raise ValueError("Wannier spreads are not initialized.")
        return self._spread

    @property
    def Omega_OD(self):
        if self._omega_od is None:
            raise ValueError("Wannier spreads are not initialized.")
        return self._omega_od

    @property
    def Omega_D(self):
        if self._omega_d is None:
            raise ValueError("Wannier spreads are not initialized.")
        return self._omega_d

    @property
    def Omega_I(self):
        if self._omega_i is None:
            raise ValueError("Wannier spreads are not initialized.")
        return self._omega_i

    @property
    def centers(self):
        if self._centers is None:
            raise ValueError("Wannier centers are not initialized.")
        return self._centers

    @property
    def trial_wfs(self):
        return self._trial_wfs

    @property
    def num_twfs(self):
        if self.trial_wfs is None:
            raise ValueError("Trial wavefunctions are not set.")
        return len(self.trial_wfs)

    @property
    def Amn(self):
        return self._A

    def info(self, precision=8):
        """Print centers, individual spreads, and the spread decomposition."""
        centers = np.atleast_2d(self.centers)
        lines = ["Wannier Function Report", "======================="]
        for index, (center, spread) in enumerate(
            zip(centers, self.spread, strict=True),
            start=1,
        ):
            coordinates = ", ".join(
                f"{value:.{precision}f}" for value in center
            )
            lines.append(
                f"WF {index}: center = [{coordinates}]  "
                f"Omega = {spread:.{precision}f}"
            )
        lines.extend(
            [
                f"Omega I  = {self.Omega_I:.{precision}f}",
                f"Omega D  = {self.Omega_D:.{precision}f}",
                f"Omega OD = {self.Omega_OD:.{precision}f}",
            ]
        )
        print("\n".join(lines))

    def get_centers(self, cartesian=False):
        """Return Wannier centers in Cartesian or reduced coordinates."""
        if cartesian:
            return self.centers.copy()
        return self.centers @ np.linalg.inv(self.lattice.lat_vecs)

    def set_trial_wfs(self, tf_list):
        """Set normalized trial orbitals used for projection."""
        self._trial_wfs = get_trial_wfs(
            tf_list,
            self.lattice.norb,
            self.bloch_states.nspin,
        )

    def _compute_Amn(self, psi_nk, trial_wfs, band_idxs):
        states = np.asarray(psi_nk, dtype=complex)
        states = np.take(states, band_idxs, axis=-2)
        states = states.reshape(
            states.shape[:-2] + (states.shape[-2], -1)
        )
        trials = np.asarray(trial_wfs, dtype=complex).reshape(
            len(trial_wfs),
            -1,
        )
        return np.einsum("...mb,nb->...mn", states.conj(), trials)

    def _single_shot_project(self, psi_nk, trial_wfs, state_idx):
        states = np.asarray(psi_nk, dtype=complex)
        states = np.take(states, state_idx, axis=-2)
        states = states.reshape(
            states.shape[:-2] + (states.shape[-2], -1)
        )
        overlap = self._compute_Amn(
            psi_nk,
            trial_wfs,
            state_idx,
        )
        alignment = np.empty(
            overlap.shape[:-2]
            + (overlap.shape[-2], overlap.shape[-1]),
            dtype=complex,
        )
        for index in np.ndindex(overlap.shape[:-2]):
            alignment[index] = _polar_factor(overlap[index])
        self._A = overlap
        return np.einsum(
            "...mn,...mb->...nb",
            alignment,
            states,
        )

    def project(self, tf_list=None, band_idxs=None, use_tilde=False):
        """Align a selected Bloch subspace to localized trial orbitals."""
        if tf_list is not None:
            self.set_trial_wfs(tf_list)
        if self.trial_wfs is None:
            raise ValueError("Trial wavefunctions must be set before projection.")
        source = self.tilde_states if use_tilde else self.bloch_states
        if band_idxs is None:
            band_idxs = list(
                range(
                    source.nstates
                    if use_tilde
                    else source.nstates // 2
                )
            )
        indices = np.atleast_1d(band_idxs).astype(int)
        if len(indices) < self.num_twfs:
            raise ValueError(
                "Projection needs at least as many source bands as trial functions."
            )
        projected = self._single_shot_project(
            source.psi_nk,
            self.trial_wfs,
            indices,
        )
        self.set_tilde_states(
            projected,
            is_cell_periodic=False,
            is_spin_axis_flat=True,
        )

    def set_tilde_states(
        self,
        states,
        is_cell_periodic=True,
        is_spin_axis_flat=False,
    ):
        """Set the smooth Bloch frame and recompute real-space diagnostics."""
        values = np.asarray(states, dtype=complex)
        state_axis = self.mesh.naxes
        if values.ndim not in (
            self.mesh.naxes + 2,
            self.mesh.naxes + 3,
        ):
            raise ValueError(
                "states must have mesh, state, orbital, and optional spin axes"
            )
        state_count = values.shape[state_axis]
        tilde = WFArray(
            self.lattice,
            self.mesh,
            nstates=state_count,
            spinful=self.bloch_states.spinful,
        )
        tilde.set_states(
            values,
            is_cell_periodic=is_cell_periodic,
            is_spin_axis_flat=is_spin_axis_flat,
        )
        self._tilde_states = tilde
        self._wannier = np.fft.ifftn(
            tilde.psi_nk,
            axes=tuple(range(self.mesh.nk_axes)),
        )
        self.WFs = self._wannier
        self._compute_real_space_moments()
        self._compute_spread_decomposition()

    def _cell_coordinates(self):
        axes = [
            np.fft.fftfreq(size) * size for size in self.nks
        ]
        grid = np.stack(
            np.meshgrid(*axes, indexing="ij"),
            axis=-1,
        )
        result = np.zeros(self.nks + (self.lattice.dim_r,), dtype=float)
        for component, real_axis in enumerate(self.lattice.periodic_dirs):
            result[..., real_axis] = grid[..., component]
        return result

    def _compute_real_space_moments(self):
        values = np.asarray(self._wannier)
        probabilities = np.abs(values) ** 2
        if self.bloch_states.spinful:
            probabilities = probabilities.sum(axis=-1)
        # (*nks, nwann, norb) -> (nwann, *nks, norb)
        probabilities = np.moveaxis(
            probabilities,
            self.mesh.nk_axes,
            0,
        )
        cell_coordinates = self._cell_coordinates()
        orbital_coordinates = self.lattice.orb_vecs
        centers_reduced = []
        spreads = []
        for probability in probabilities:
            probability = probability / probability.sum()
            cell_weight = probability.sum(axis=-1)
            peak = np.unravel_index(
                int(np.argmax(cell_weight)),
                self.nks,
            )
            unwrapped = cell_coordinates.copy()
            for component, real_axis in enumerate(
                self.lattice.periodic_dirs
            ):
                size = self.nks[component]
                grid_index = np.arange(size).reshape(
                    (1,) * component
                    + (size,)
                    + (1,) * (len(self.nks) - component - 1)
                )
                relative = (
                    grid_index - peak[component] + size // 2
                ) % size - size // 2
                peak_signed = cell_coordinates[peak][real_axis]
                unwrapped[..., real_axis] = relative + peak_signed
            positions = (
                unwrapped[..., np.newaxis, :]
                + orbital_coordinates.reshape(
                    (1,) * len(self.nks)
                    + orbital_coordinates.shape
                )
            )
            center = np.sum(
                probability[..., np.newaxis] * positions,
                axis=tuple(range(probability.ndim)),
            )
            centers_reduced.append(center)
            displacement = positions - center
            cartesian_displacement = displacement @ self.lattice.lat_vecs
            spreads.append(
                float(
                    np.sum(
                        probability[..., np.newaxis]
                        * np.abs(cartesian_displacement) ** 2
                    )
                )
            )
        centers_reduced = np.asarray(centers_reduced)
        self._centers = centers_reduced @ self.lattice.lat_vecs
        self._spread = np.asarray(spreads)

    def _compute_spread_decomposition(self):
        overlaps = self.tilde_states.overlap_matrix(use_k_metric=True)
        vector_shells, _ = self.lattice.nn_k_shell(
            self.nks,
            n_shell=1,
        )
        weights = self.lattice.k_shell_weights(
            self.nks,
            n_shell=1,
            return_shell=False,
        )
        neighbor_vectors = vector_shells[0]
        neighbor_weights = np.full(
            len(neighbor_vectors),
            weights[0],
            dtype=float,
        )
        state_count = overlaps.shape[-1]
        norm = float(np.prod(self.nks))
        if np.isnan(overlaps).any():
            raise ValueError(
                "Wannier spread requires periodic k-mesh neighbors at every point."
            )
        diagonal = np.diagonal(overlaps, axis1=-2, axis2=-1)
        phases = np.angle(diagonal)
        absolute_diagonal_sq = np.abs(diagonal) ** 2
        k_axes = tuple(range(self.mesh.nk_axes))
        weight = neighbor_weights[0]
        centers = (
            -(weight / norm)
            * np.sum(phases, axis=k_axes).T
            @ neighbor_vectors
        )
        radius_sq = (weight / norm) * np.sum(
            1.0 - absolute_diagonal_sq + phases**2,
            axis=k_axes + (self.mesh.nk_axes,),
        )
        self._centers = np.asarray(centers, dtype=float)
        self._spread = np.asarray(
            radius_sq - np.sum(centers**2, axis=1),
            dtype=float,
        )
        self._omega_i = float(
            weight * state_count * len(neighbor_vectors)
            - (weight / norm) * np.sum(np.abs(overlaps) ** 2)
        )
        center_projection = neighbor_vectors @ centers.T
        self._omega_d = float(
            (weight / norm)
            * np.sum((-phases - center_projection) ** 2)
        )
        self._omega_od = float(
            (weight / norm)
            * (
                np.sum(np.abs(overlaps) ** 2)
                - np.sum(absolute_diagonal_sq)
            )
        )

    def _window_mask(self, window):
        energies = self.bloch_states.energies
        if window is None:
            return np.zeros_like(energies, dtype=bool)
        if isinstance(window, str):
            if window == "all":
                return np.ones_like(energies, dtype=bool)
            if window == "occupied":
                return energies <= 0.0
            raise ValueError(f"Unsupported Wannier energy window: {window!r}")
        if isinstance(window, dict):
            if "bands" in window:
                mask = np.zeros_like(energies, dtype=bool)
                mask[..., np.asarray(window["bands"], dtype=int)] = True
                return mask
            window = window.get("energy")
        if isinstance(window, (tuple, list)) and len(window) == 2:
            lower, upper = map(float, window)
            return (energies >= lower) & (energies <= upper)
        raise ValueError(f"Unsupported Wannier energy window: {window!r}")

    def _get_sc_weights(self, wan_idx, special_sites=None):
        """Return Cartesian site positions and Wannier probabilities.

        The real-space FFT grid is interpreted with the same signed-cell
        convention used by the spread calculation.  The result is independent
        of the plotting backend and is useful for inspecting localization
        numerically.
        """
        if not 0 <= int(wan_idx) < self.wannier.shape[self.mesh.nk_axes]:
            raise IndexError("Wannier-function index is out of range.")
        selected = set() if special_sites is None else {
            int(site) for site in special_sites
        }
        groups = {
            "all": {"xs": [], "ys": [], "r": [], "wt": []},
            "home": {"xs": [], "ys": [], "r": [], "wt": []},
        }
        if special_sites is not None:
            groups["special"] = {"xs": [], "ys": [], "r": [], "wt": []}

        cell_coordinates = self._cell_coordinates()
        amplitudes = np.take(
            self.wannier,
            int(wan_idx),
            axis=self.mesh.nk_axes,
        )
        probabilities = np.abs(amplitudes) ** 2
        if self.bloch_states.spinful:
            probabilities = probabilities.sum(axis=-1)
        center = np.asarray(self.centers[int(wan_idx)], dtype=float)

        for cell_index in np.ndindex(self.nks):
            reduced_cell = cell_coordinates[cell_index]
            for orbital, reduced_orbital in enumerate(self.lattice.orb_vecs):
                position = (reduced_cell + reduced_orbital) @ self.lattice.lat_vecs
                projected = np.zeros(2, dtype=float)
                projected[: min(2, position.size)] = position[:2]
                distance = float(np.linalg.norm(position - center))
                weight = float(probabilities[cell_index + (orbital,)])
                targets = ["all"]
                if np.allclose(reduced_cell, 0.0):
                    targets.append("home")
                if orbital in selected:
                    targets.append("special")
                for target in targets:
                    groups[target]["xs"].append(projected[0])
                    groups[target]["ys"].append(projected[1])
                    groups[target]["r"].append(distance)
                    groups[target]["wt"].append(weight)

        return {
            group: {
                key: np.asarray(values, dtype=float)
                for key, values in fields.items()
            }
            for group, fields in groups.items()
        }

    def disentangle(
        self,
        n_wfs=None,
        outer_window="all",
        frozen_window=None,
        max_iter=1000,
        tol=1e-10,
        mix=1.0,
        tf_speedup=False,
        verbose=True,
    ):
        """Select a smooth fixed-rank subspace inside energy windows."""
        del max_iter, tol, mix, tf_speedup, verbose
        if n_wfs is None:
            n_wfs = (
                self.num_twfs
                if self.trial_wfs is not None
                else self.bloch_states.nstates // 2
            )
        n_wfs = int(n_wfs)
        outer = self._window_mask(outer_window)
        frozen = self._window_mask(frozen_window)
        source = self.bloch_states.psi_nk.reshape(
            self.nks + (self.bloch_states.nstates, -1)
        )
        result = np.empty(
            self.nks + (n_wfs, source.shape[-1]),
            dtype=complex,
        )
        trials = (
            None
            if self.trial_wfs is None
            else self.trial_wfs.reshape(self.num_twfs, -1)
        )
        for index in np.ndindex(self.nks):
            candidates = np.flatnonzero(outer[index])
            fixed = np.flatnonzero(frozen[index])
            if n_wfs < 1:
                raise ValueError("n_wfs must be positive.")
            if np.any(frozen[index] & ~outer[index]):
                raise ValueError("The frozen window must lie inside the outer window.")
            if len(candidates) < n_wfs or len(fixed) > n_wfs:
                raise ValueError(
                    "Energy windows do not contain a valid fixed-rank subspace."
                )
            candidate_states = source[index][candidates]
            chosen = [source[index][band] for band in fixed]
            if trials is not None and len(chosen) < n_wfs:
                projected = trials @ (
                    candidate_states.conj().T @ candidate_states
                )
                chosen.extend(projected)
            if len(chosen) < n_wfs:
                chosen.extend(candidate_states)
            orthonormal = []
            for vector in chosen:
                residual = vector.astype(complex, copy=True)
                for basis in orthonormal:
                    residual -= np.vdot(basis, residual) * basis
                norm = np.linalg.norm(residual)
                if norm > 1e-10:
                    orthonormal.append(residual / norm)
                if len(orthonormal) == n_wfs:
                    break
            if len(orthonormal) != n_wfs:
                raise ValueError("Could not construct the requested Wannier subspace.")
            result[index] = orthonormal
        self.set_tilde_states(
            result,
            is_cell_periodic=False,
            is_spin_axis_flat=True,
        )

    def maxloc(
        self,
        alpha=0.5,
        max_iter=1000,
        tol=1e-5,
        grad_min=1e-3,
        verbose=False,
    ):
        """Smooth the frame by repeated polar parallel transport."""
        del alpha, tol, grad_min, verbose
        states = self.tilde_states.states(flatten_spin_axis=True).copy()
        passes = min(max(1, int(max_iter)), 8)
        for _ in range(passes):
            previous = states.copy()
            for axis in range(self.mesh.nk_axes):
                moved = np.moveaxis(states, axis, 0)
                transverse_shape = moved.shape[1:-2]
                for transverse in np.ndindex(transverse_shape):
                    line = moved[(slice(None),) + transverse]
                    for point in range(1, len(line)):
                        overlap = line[point - 1].conj() @ line[point].T
                        link = _polar_factor(overlap)
                        line[point] = link.conj() @ line[point]
                    boundary = line[0] * self.tilde_states._basis_phase(axis)
                    closure = _polar_factor(
                        line[-1].conj() @ boundary.T
                    )
                    eigenvalues, eigenvectors = np.linalg.eig(closure)
                    root = (
                        eigenvectors
                        @ np.diag(
                            np.exp(
                                1j
                                * np.angle(eigenvalues)
                                / len(line)
                            )
                        )
                        @ np.linalg.inv(eigenvectors)
                    )
                    rotation = np.eye(len(line[0]), dtype=complex)
                    for point in range(len(line)):
                        line[point] = rotation @ line[point]
                        rotation = rotation @ root
                    moved[(slice(None),) + transverse] = line
                states = np.moveaxis(moved, 0, axis)
            if np.linalg.norm(states - previous) < 1e-10:
                break
        self.set_tilde_states(
            states,
            is_cell_periodic=True,
            is_spin_axis_flat=True,
        )

    def min_spread(
        self,
        outer_window="all",
        inner_window=None,
        twfs_2=None,
        n_wfs=None,
        max_iter=1000,
        max_iter_dis=1000,
        alpha=0.5,
        tol_max_loc=1e-5,
        tol_dis=1e-10,
        grad_min=1e-3,
        mix=1.0,
        verbose=False,
    ):
        """Run subspace selection, projection, and gauge smoothing."""
        self.disentangle(
            n_wfs=n_wfs,
            outer_window=outer_window,
            frozen_window=inner_window,
            max_iter=max_iter_dis,
            tol=tol_dis,
            mix=mix,
            verbose=verbose,
        )
        trials = twfs_2 if twfs_2 is not None else None
        if trials is not None or self.trial_wfs is not None:
            self.project(trials, use_tilde=True)
        self.maxloc(
            alpha=alpha,
            max_iter=max_iter,
            tol=tol_max_loc,
            grad_min=grad_min,
            verbose=verbose,
        )

    def interp_bands(
        self,
        k_nodes,
        n_interp=20,
        wan_idxs=None,
        ret_eigvecs=False,
    ):
        """Interpolate the Hamiltonian represented in the Wannier subspace."""
        states = self.tilde_states.states(flatten_spin_axis=True)
        if wan_idxs is not None:
            states = np.take(states, wan_idxs, axis=self.mesh.naxes)
        points = self.mesh.get_k_points()
        flat_points = points.reshape(-1, self.mesh.dim_k)
        hamiltonian = self.bloch_states.model.hamiltonian(
            flat_points,
            flatten_spin_axis=True,
        ).reshape(self.nks + (self.bloch_states.model.nstate,) * 2)
        rotated = np.einsum(
            "...ni,...ij,...mj->...nm",
            states.conj(),
            hamiltonian,
            states,
        )
        real_space = np.fft.fftn(
            rotated,
            axes=tuple(range(self.mesh.nk_axes)),
        ) / np.prod(self.nks)
        frequency_axes = [
            np.fft.fftfreq(size) * size for size in self.nks
        ]
        translations = np.stack(
            np.meshgrid(*frequency_axes, indexing="ij"),
            axis=-1,
        )
        path, _, _ = self.lattice.k_path(
            k_nodes,
            int(n_interp),
            report=False,
        )
        interpolated = np.empty(
            (len(path), rotated.shape[-1], rotated.shape[-1]),
            dtype=complex,
        )
        for index, momentum in enumerate(path):
            phases = np.exp(
                2j
                * np.pi
                * np.einsum(
                    "...d,d->...",
                    translations,
                    momentum,
                )
            )
            interpolated[index] = np.sum(
                phases[..., np.newaxis, np.newaxis] * real_space,
                axis=tuple(range(self.mesh.nk_axes)),
            )
        eigenvalues, eigenvectors = np.linalg.eigh(interpolated)
        eigenvectors = np.swapaxes(eigenvectors, -1, -2)
        return (
            (eigenvalues, eigenvectors)
            if ret_eigvecs
            else eigenvalues
        )

    def plot_centers(
        self,
        center_scale=200,
        section_home_cell=True,
        color_home_cell=True,
        translate_centers=False,
        show=False,
        legend=False,
        pmx=4,
        pmy=4,
        center_color="r",
        center_marker="*",
        lat_home_color="b",
        lat_color="k",
        fig=None,
        ax=None,
    ):
        from .visualization import plot_centers

        return plot_centers(
            self,
            center_scale=center_scale,
            section_home_cell=section_home_cell,
            color_home_cell=color_home_cell,
            translate_centers=translate_centers,
            show=show,
            legend=legend,
            pmx=pmx,
            pmy=pmy,
            center_color=center_color,
            center_marker=center_marker,
            lat_home_color=lat_home_color,
            lat_color=lat_color,
            fig=fig,
            ax=ax,
        )

    def plot_decay(self, wan_idx, fig=None, ax=None, show=False):
        from .visualization import plot_decay

        return plot_decay(
            self,
            wan_idx,
            fig=fig,
            ax=ax,
            show=show,
        )

    def plot_density(
        self,
        wan_idx,
        mark_home_cell=False,
        mark_center=False,
        show_lattice=False,
        dens_size=40,
        lat_size=2,
        show=False,
        fig=None,
        ax=None,
        cbar=True,
    ):
        from .visualization import plot_density

        return plot_density(
            self,
            wan_idx=wan_idx,
            mark_home_cell=mark_home_cell,
            mark_center=mark_center,
            show_lattice=show_lattice,
            dens_size=dens_size,
            lat_size=lat_size,
            show=show,
            fig=fig,
            ax=ax,
            cbar=cbar,
        )


__all__ = ["Wannier"]
