"""Array-backed hopping storage used by PythTB-compatible model builders."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class HoppingTable:
    """Store hopping amplitudes, orbital pairs, and lattice translations.

    The table keeps columnar NumPy arrays for efficient Hamiltonian assembly
    and a key-to-row map for constant-time source-compatible lookup.
    """

    def __init__(self, dim_r: int, spinful: bool):
        if not isinstance(dim_r, (int, np.integer)) or int(dim_r) < 0:
            raise ValueError("dim_r must be a non-negative integer.")
        self.dim_r = int(dim_r)
        self.spinful = bool(spinful)
        self._index = {}
        self._flatten_cache = {}
        self.clear()

    def __len__(self):
        return len(self.from_idx)

    def __iter__(self):
        for row in range(len(self)):
            yield (
                self.amplitudes[row].copy()
                if self.spinful
                else complex(self.amplitudes[row]),
                int(self.from_idx[row]),
                int(self.to_idx[row]),
                self.lattice_vecs[row].copy(),
            )

    def _amplitude(self, value):
        array = np.asarray(value, dtype=complex)
        if self.spinful:
            if array.shape != (2, 2):
                raise ValueError("Spinful hopping amplitudes must be 2x2 matrices.")
            return array
        if array.ndim != 0:
            raise ValueError("Spinless hopping amplitudes must be scalars.")
        return array.reshape(())

    def _translation(self, value):
        array = np.asarray(value, dtype=int)
        if array.shape != (self.dim_r,):
            raise ValueError(f"Lattice vectors must have shape ({self.dim_r},).")
        return array

    def _key(self, i, j, translation):
        return (
            int(i),
            int(j),
            tuple(int(value) for value in translation),
        )

    def _reindex(self):
        self._index = {
            self._key(self.from_idx[row], self.to_idx[row], self.lattice_vecs[row]): row
            for row in range(len(self))
        }
        self._flatten_cache.clear()

    def clear(self):
        shape = (0, 2, 2) if self.spinful else (0,)
        self.amplitudes = np.empty(shape, dtype=complex)
        self.from_idx = np.empty(0, dtype=int)
        self.to_idx = np.empty(0, dtype=int)
        self.lattice_vecs = np.empty((0, self.dim_r), dtype=int)
        self._index.clear()
        self._flatten_cache.clear()

    def components(self):
        return (
            self.amplitudes,
            self.from_idx,
            self.to_idx,
            self.lattice_vecs,
        )

    def append(self, amplitude, i: int, j: int, R: Sequence[int]):
        value = self._amplitude(amplitude)
        translation = self._translation(R)
        self.amplitudes = np.concatenate(
            (self.amplitudes, value.reshape((1,) + value.shape)),
            axis=0,
        )
        self.from_idx = np.append(self.from_idx, int(i))
        self.to_idx = np.append(self.to_idx, int(j))
        self.lattice_vecs = np.concatenate(
            (self.lattice_vecs, translation[np.newaxis, :]),
            axis=0,
        )
        row = len(self) - 1
        self._index[self._key(i, j, translation)] = row
        self._flatten_cache.clear()
        return row

    def extend(self, amplitudes, i_idx, j_idx, lattice_vecs):
        values = list(amplitudes)
        origins = list(i_idx)
        targets = list(j_idx)
        translations = list(lattice_vecs)
        lengths = {len(values), len(origins), len(targets), len(translations)}
        if len(lengths) != 1:
            raise ValueError(
                "Lengths of amplitudes, i_idx, j_idx, and lattice_vecs must match."
            )
        for value, origin, target, translation in zip(
            values,
            origins,
            targets,
            translations,
            strict=True,
        ):
            self.append(value, origin, target, translation)

    def update(self, idx, *, amplitude=None, R=None):
        row = int(idx)
        if amplitude is not None:
            self.amplitudes[row] = self._amplitude(amplitude)
        if R is not None:
            self.lattice_vecs[row] = self._translation(R)
        self._reindex()

    def remove(self, idx):
        row = int(idx)
        if not 0 <= row < len(self):
            raise IndexError("Index out of range.")
        self.amplitudes = np.delete(self.amplitudes, row, axis=0)
        self.from_idx = np.delete(self.from_idx, row)
        self.to_idx = np.delete(self.to_idx, row)
        self.lattice_vecs = np.delete(self.lattice_vecs, row, axis=0)
        self._reindex()

    def add(self, idx, delta):
        self.amplitudes[int(idx)] += self._amplitude(delta)
        self._flatten_cache.clear()

    def remove_orbitals(self, indices):
        removed = sorted({int(index) for index in indices})
        if not removed:
            return
        mask = ~np.isin(self.from_idx, removed) & ~np.isin(self.to_idx, removed)
        self.amplitudes = self.amplitudes[mask]
        self.from_idx = self.from_idx[mask]
        self.to_idx = self.to_idx[mask]
        self.lattice_vecs = self.lattice_vecs[mask]
        for index in reversed(removed):
            self.from_idx[self.from_idx > index] -= 1
            self.to_idx[self.to_idx > index] -= 1
        self._reindex()

    def shift_orbital(self, orb_idx, disp_vec):
        orbital = int(orb_idx)
        displacement = self._translation(disp_vec)
        self.lattice_vecs[self.from_idx == orbital] -= displacement
        self.lattice_vecs[self.to_idx == orbital] += displacement
        self._reindex()

    def normalize_entry(
        self,
        ind_i,
        ind_j,
        ind_R,
        *,
        norb,
        dim_k,
        periodic_dirs,
    ):
        if not isinstance(ind_i, (int, np.integer)) or not isinstance(
            ind_j, (int, np.integer)
        ):
            raise TypeError("Orbital indices must be integers.")
        i, j = int(ind_i), int(ind_j)
        if not 0 <= i < int(norb) or not 0 <= j < int(norb):
            raise ValueError("Orbital index is outside the model basis.")
        if int(dim_k) == 0:
            if ind_R is not None:
                raise ValueError(
                    "No periodic directions, so ind_R should not be specified."
                )
            return i, j, np.zeros(self.dim_r, dtype=int)
        if ind_R is None:
            raise ValueError("Must specify ind_R when periodic directions exist.")
        if isinstance(ind_R, (int, np.integer)):
            if int(dim_k) != 1:
                raise ValueError(
                    "An integer ind_R is only valid for one periodic direction."
                )
            translation = np.zeros(self.dim_r, dtype=int)
            translation[int(list(periodic_dirs)[0])] = int(ind_R)
        else:
            raw = np.asarray(ind_R)
            if raw.shape != (self.dim_r,) or not np.issubdtype(
                raw.dtype, np.integer
            ):
                raise ValueError(
                    f"ind_R must be an integer vector of length {self.dim_r}."
                )
            translation = raw.astype(int, copy=False)
        allowed = set(int(axis) for axis in periodic_dirs)
        if any(value and axis not in allowed for axis, value in enumerate(translation)):
            raise ValueError(
                "ind_R may only have non-zero components along periodic directions."
            )
        return i, j, translation

    def find(self, i, j, R):
        return self._index.get(self._key(i, j, self._translation(R)))

    def flatten_cache(self, norb):
        cache_key = (int(norb), len(self))
        cached = self._flatten_cache.get(cache_key)
        if cached is not None:
            return cached
        flat = self.from_idx * int(norb) + self.to_idx
        order = np.argsort(flat, kind="stable")
        ordered = flat[order]
        starts = (
            np.empty(0, dtype=int)
            if len(ordered) == 0
            else np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
        )
        unique = ordered[starts]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        result = {
            "order": order,
            "starts": starts,
            "uniq": unique,
            "cols_transposed": (unique % int(norb)) * int(norb)
            + unique // int(norb),
            "inverse_order": inverse,
        }
        self._flatten_cache[cache_key] = result
        return result


__all__ = ["HoppingTable"]
