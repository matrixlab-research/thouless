"""Wannier90 dataset adapter for PythTB-compatible tight-binding models."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np

from .io.qe import read_bands_qe
from .io.w90 import (
    load_w90_dataset,
    read_bands_w90,
    read_kpoint_path,
)
from .lattice import Lattice
from .tbmodel import TBModel


def _path_distance(points, reciprocal_vectors):
    cartesian = np.asarray(points, dtype=float) @ np.asarray(
        reciprocal_vectors,
        dtype=float,
    )
    result = np.zeros(len(cartesian), dtype=float)
    if len(result) > 1:
        result[1:] = np.cumsum(
            np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
        )
    return result


class W90:
    """Read a Wannier90 real-space Hamiltonian and construct a ``TBModel``."""

    def __init__(self, path, prefix):
        self.folder = Path(path).expanduser()
        if not self.folder.exists():
            raise FileNotFoundError(f"Wannier90 folder not found: {self.folder}")
        self.path = str(self.folder)
        self.prefix = str(prefix)
        dataset = load_w90_dataset(self.folder, self.prefix)
        self._win_lines = (
            self.folder / f"{self.prefix}.win"
        ).read_text(encoding="utf-8", errors="ignore").splitlines()
        self.lat = dataset.lat_cart
        self.num_wan = dataset.num_wan
        self.ham_r = {
            vector: {
                "h": block.h.copy(),
                "deg": int(block.degeneracy),
            }
            for vector, block in dataset.ham_r.items()
        }
        self.xyz_cen = dataset.centres_xyz
        self.red_cen = dataset.centres_red
        self.lattice = Lattice(
            self.lat,
            self.red_cen,
            periodic_dirs=[0, 1, 2],
        )
        self._validate_hr_symmetry()
        self._distance_cache = {}

    def _validate_hr_symmetry(self):
        for vector in self.ham_r:
            if vector == (0, 0, 0):
                continue
            opposite = tuple(-component for component in vector)
            if opposite not in self.ham_r:
                raise ValueError(f"Did not find negative R for R = {vector}!")

    def _distance_matrix(self, vector):
        if vector not in self._distance_cache:
            translation = np.asarray(vector, dtype=float) @ self.lat
            displacement = (
                self.xyz_cen[None, :, :]
                + translation
                - self.xyz_cen[:, None, :]
            )
            self._distance_cache[vector] = np.linalg.norm(
                displacement,
                axis=-1,
            )
        return self._distance_cache[vector]

    @staticmethod
    def _positive_vector(vector):
        return next(
            (
                component > 0
                for component in vector
                if component != 0
            ),
            False,
        )

    def model(
        self,
        zero_energy=0.0,
        min_hopping_norm=None,
        max_distance=None,
        ignorable_imaginary_part=None,
        *,
        onsite_imag_tol=1e-9,
        fill_hermitian=False,
    ):
        """Construct a model, optionally filtering weak or long hoppings."""
        del fill_hermitian
        zero_block = self.ham_r.get((0, 0, 0))
        if zero_block is None:
            raise ValueError("Wannier90 Hamiltonian has no R=(0,0,0) block")
        onsite = np.diag(zero_block["h"]) / float(zero_block["deg"])
        if np.max(np.abs(onsite.imag)) > onsite_imag_tol:
            raise ValueError("Onsite terms have a non-negligible imaginary part")

        model = TBModel(self.lattice)
        model._from_w90 = True
        model.assume_position_operator_diagonal = False
        model.set_onsite(onsite.real - float(zero_energy))

        for vector, block in self.ham_r.items():
            if vector != (0, 0, 0) and not self._positive_vector(vector):
                continue
            hamiltonian = np.asarray(block["h"], dtype=complex) / float(
                block["deg"]
            )
            distances = (
                self._distance_matrix(vector)
                if max_distance is not None
                else None
            )
            for target in range(self.num_wan):
                for source in range(self.num_wan):
                    if vector == (0, 0, 0) and target >= source:
                        continue
                    amplitude = hamiltonian[target, source]
                    if min_hopping_norm is not None and (
                        abs(amplitude) < min_hopping_norm
                    ):
                        continue
                    if max_distance is not None and (
                        distances[target, source] > max_distance
                    ):
                        continue
                    if (
                        ignorable_imaginary_part is not None
                        and abs(amplitude.imag) < ignorable_imaginary_part
                    ):
                        amplitude = complex(amplitude.real)
                    if amplitude != 0:
                        model.set_hop(
                            amplitude,
                            target,
                            source,
                            list(vector),
                        )
        return model

    def dist_hop(self):
        """Return flattened non-onsite distances and hopping amplitudes."""
        distances = []
        hoppings = []
        for vector, block in self.ham_r.items():
            distance = self._distance_matrix(vector)
            hamiltonian = block["h"] / float(block["deg"])
            keep = np.ones((self.num_wan, self.num_wan), dtype=bool)
            if vector == (0, 0, 0):
                np.fill_diagonal(keep, False)
            distances.append(distance[keep])
            hoppings.append(hamiltonian[keep])
        return np.concatenate(distances), np.concatenate(hoppings)

    def shells(self, num_digits=2):
        """Return sorted distinct center-to-center distance shells."""
        values = {
            float(value)
            for vector in self.ham_r
            for value in np.round(
                self._distance_matrix(vector),
                int(num_digits),
            ).ravel()
        }
        return np.asarray(sorted(values), dtype=float)

    def w90_bands_consistency(self):
        """Deprecated alias for :meth:`bands_w90`."""
        warnings.warn(
            "use bands_w90() instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.bands_w90()

    def bands_w90(
        self,
        return_k_cart=False,
        return_k_dist=False,
        return_k_nodes=False,
    ):
        """Read interpolated Wannier90 bands and optional path metadata."""
        k_points, energies = read_bands_w90(
            self.folder,
            self.prefix,
            self.num_wan,
        )
        result = [k_points, energies]
        reciprocal = self.lattice.recip_lat_vecs
        if return_k_dist:
            result.append(_path_distance(k_points, reciprocal))
        if return_k_cart:
            result.append(k_points @ reciprocal)
        if return_k_nodes:
            result.extend(read_kpoint_path(self._win_lines, latex=True))
        return tuple(result)

    def bands_qe(
        self,
        return_k_cart=False,
        return_meta=False,
        return_kdist=False,
        *,
        alat=None,
    ):
        """Read Quantum ESPRESSO bands and convert markers to reduced k."""
        markers, energy_rows, metadata = read_bands_qe(
            self.folder,
            self.prefix,
        )
        band_count = metadata.get(
            "nbnd",
            max((len(row) for row in energy_rows), default=0),
        )
        energies = np.full((len(markers), band_count), np.nan)
        for index, row in enumerate(energy_rows):
            energies[index, : min(len(row), band_count)] = row[:band_count]
        lattice_scale = (
            np.linalg.norm(self.lattice.lat_vecs[0])
            if alat is None
            else float(alat)
        )
        if not np.isfinite(lattice_scale) or lattice_scale <= 0:
            raise ValueError("alat must be a positive finite length")
        cartesian = markers * (2 * np.pi / lattice_scale)
        reduced = cartesian @ np.linalg.inv(self.lattice.recip_lat_vecs)
        result = [reduced, energies]
        if return_kdist:
            result.append(
                _path_distance(reduced, self.lattice.recip_lat_vecs)
            )
        if return_k_cart:
            result.append(cartesian)
        if return_meta:
            result.append(metadata)
        return tuple(result)


__all__ = ["W90"]
