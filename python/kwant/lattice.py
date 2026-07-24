"""Kwant lattice families and translational symmetry."""

from __future__ import annotations

import itertools
from collections import deque

import numpy as np
import tinyarray as ta

from .builder import HoppingKind, SiteFamily, Symmetry


class Monatomic(SiteFamily):
    """One Bravais lattice with one basis site."""

    def __init__(self, prim_vecs, offset=None, name="", norbs=None):
        primitive = np.asarray(prim_vecs, dtype=float)
        if primitive.ndim != 2 or primitive.shape[0] == 0:
            raise ValueError("Primitive vectors must form a nonempty 2D array.")
        if np.linalg.matrix_rank(primitive) < primitive.shape[0]:
            raise ValueError("Primitive vectors must be linearly independent.")
        self.prim_vecs = primitive
        self.lattice_dim = primitive.shape[0]
        self.space_dim = primitive.shape[1]
        self.offset = (
            np.zeros(self.space_dim)
            if offset is None
            else np.asarray(offset, dtype=float)
        )
        if self.offset.shape != (self.space_dim,):
            raise ValueError("Basis offset has wrong dimension.")
        canonical = (
            f"Monatomic({self.prim_vecs.tolist()!r}, {self.offset.tolist()!r}, "
            f"{name!r}, {norbs!r})"
        )
        super().__init__(canonical, name, norbs)
        self.sublattices = (self,)

    def normalize_tag(self, tag):
        array = np.asarray(tag)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.shape != (self.lattice_dim,):
            raise ValueError("Site tag has wrong dimensionality.")
        if not np.all(np.equal(array, np.rint(array))):
            raise ValueError("Site tags must be integers.")
        return ta.array([int(value) for value in array], int)

    def pos(self, tag):
        return ta.array(
            np.asarray(tag, dtype=float) @ self.prim_vecs + self.offset
        )

    def vec(self, tag):
        tag = self.normalize_tag(tag)
        return ta.array(np.asarray(tag, dtype=float) @ self.prim_vecs)

    def closest(self, position):
        position = np.asarray(position, dtype=float)
        if position.shape != (self.space_dim,):
            raise ValueError("Position has wrong dimensionality.")
        reduced = np.linalg.lstsq(
            self.prim_vecs.T,
            position - self.offset,
            rcond=None,
        )[0]
        center = np.rint(reduced).astype(int)
        candidates = (
            center + np.asarray(delta)
            for delta in itertools.product(range(-4, 5), repeat=self.lattice_dim)
        )
        closest = min(
            candidates,
            key=lambda tag: np.linalg.norm(
                tag @ self.prim_vecs + self.offset - position
            ),
        )
        return ta.array(closest, int)

    def shape(self, function, start):
        return Polyatomic(
            self.prim_vecs,
            [self.offset],
            [self.name],
            self.norbs,
        ).shape(function, start)

    def wire(self, center, radius):
        return Polyatomic(
            self.prim_vecs,
            [self.offset],
            [self.name],
            self.norbs,
        ).wire(center, radius)

    def neighbors(self, n=1, eps=1e-8):
        return Polyatomic(self.prim_vecs, [self.offset], [self.name], self.norbs).neighbors(
            n, eps
        )


class Polyatomic:
    """A Bravais lattice with one or more basis families."""

    def __init__(self, prim_vecs, basis, name="", norbs=None):
        primitive = np.asarray(prim_vecs, dtype=float)
        basis = np.asarray(basis, dtype=float)
        if primitive.ndim != 2 or primitive.shape[0] == 0:
            raise ValueError("Primitive vectors must form a nonempty 2D array.")
        if np.linalg.matrix_rank(primitive) < primitive.shape[0]:
            raise ValueError("Primitive vectors must be linearly independent.")
        if basis.ndim != 2 or basis.shape[1] != primitive.shape[1]:
            raise ValueError("Basis positions have wrong dimensionality.")
        self.prim_vecs = primitive
        self.lattice_dim = primitive.shape[0]
        self.space_dim = primitive.shape[1]
        if isinstance(name, str):
            names = [f"{name}{index}" for index in range(len(basis))]
        else:
            names = list(name)
        if np.isscalar(norbs) or norbs is None:
            orbital_counts = [norbs] * len(basis)
        else:
            orbital_counts = list(norbs)
        if len(names) != len(basis) or len(orbital_counts) != len(basis):
            raise ValueError("Name and norbs must match the number of sublattices.")
        self.sublattices = tuple(
            Monatomic(primitive, offset, family_name, family_norbs)
            for offset, family_name, family_norbs in zip(
                basis, names, orbital_counts, strict=True
            )
        )

    def vec(self, tag):
        return self.sublattices[0].vec(tag)

    def shape(self, function, start):
        start = np.asarray(start, dtype=float)
        if start.shape != (self.space_dim,):
            raise ValueError("Shape start has wrong dimensionality.")

        def site_generator(symmetry=None):
            symmetry = getattr(symmetry, "symmetry", symmetry)
            seeds = []
            for family in self.sublattices:
                central = np.asarray(family.closest(start), dtype=int)
                for delta in itertools.product(
                    range(-2, 3), repeat=self.lattice_dim
                ):
                    site = family(*(central + np.asarray(delta)))
                    if function(site.pos):
                        seeds.append(site)
            if not seeds:
                raise ValueError("No sites close to the shape start are inside")
            seeds.sort(key=lambda site: np.linalg.norm(np.asarray(site.pos) - start))

            queue = deque([seeds[0]])
            visited = set()
            while queue:
                site = queue.popleft()
                canonical = (
                    site if symmetry is None else symmetry.to_fd(site)
                )
                if canonical in visited:
                    continue
                visited.add(canonical)
                if not function(canonical.pos):
                    continue
                yield canonical
                tag = np.asarray(canonical.tag, dtype=int)
                for family in self.sublattices:
                    queue.append(family(*tag))
                    for axis in range(self.lattice_dim):
                        for step in (-1, 1):
                            neighbor_tag = tag.copy()
                            neighbor_tag[axis] += step
                            queue.append(family(*neighbor_tag))

        return site_generator

    def wire(self, center, radius):
        center = np.asarray(center, dtype=float)
        direction = np.asarray(self.prim_vecs[0], dtype=float)
        direction /= np.linalg.norm(direction)

        def inside(position):
            displacement = np.asarray(position, dtype=float) - center
            transverse = displacement - np.dot(displacement, direction) * direction
            return np.dot(transverse, transverse) <= float(radius) ** 2

        return self.shape(inside, center)

    def neighbors(self, n=1, eps=1e-8):
        n = int(n)
        if n < 0:
            raise ValueError("Neighbor order must be nonnegative.")
        search = max(2, n + 1)
        length_scale = min(
            np.linalg.norm(vector)
            for vector in self.prim_vecs
            if np.linalg.norm(vector) > 0
        )
        candidates = []
        origin = np.zeros(self.lattice_dim, dtype=int)
        for first_index, first in enumerate(self.sublattices):
            for second_index in range(first_index, len(self.sublattices)):
                second = self.sublattices[second_index]
                for delta in itertools.product(
                    range(-search, search + 1), repeat=self.lattice_dim
                ):
                    displacement = (
                        np.asarray(delta) @ self.prim_vecs
                        + second.offset
                        - first.offset
                    )
                    distance = float(np.linalg.norm(displacement) / length_scale)
                    candidates.append(
                        (distance, tuple(int(value) for value in delta), first, second)
                    )
        distance_groups = []
        for distance in sorted(distance for distance, _, _, _ in candidates):
            if not distance_groups or abs(distance - distance_groups[-1]) > eps:
                distance_groups.append(distance)
        if n >= len(distance_groups):
            return []
        target_distance = distance_groups[n]
        result = []
        seen = set()
        for distance, delta, first, second in sorted(
            candidates,
            key=lambda entry: (
                entry[2].canonical_repr,
                entry[3].canonical_repr,
                entry[1],
            ),
        ):
            if abs(distance - target_distance) > eps:
                continue
            if first == second:
                opposite = tuple(-value for value in delta)
                canonical_delta = max(delta, opposite)
                key = (canonical_delta, first, second)
                if key in seen:
                    continue
                delta = canonical_delta
            key = (delta, first, second)
            if key in seen:
                continue
            seen.add(key)
            result.append(HoppingKind(delta, first, second))
        return result


class TranslationalSymmetry(Symmetry):
    """Discrete translation group represented by Cartesian periods."""

    def __init__(self, *periods):
        array = np.asarray(periods, dtype=float)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("At least one translation period is required.")
        if np.linalg.matrix_rank(array) < array.shape[0]:
            raise ValueError("Translation periods must be linearly independent.")
        self.periods = array
        self._family_periods = {}

    @property
    def num_directions(self):
        return len(self.periods)

    def add_site_family(self, family, other_vectors=None):
        if family in self._family_periods:
            return
        reduced = self.periods @ np.linalg.pinv(family.prim_vecs)
        rounded = np.rint(reduced)
        if not np.allclose(reduced, rounded, atol=1e-8) or not np.allclose(
            rounded @ family.prim_vecs, self.periods, atol=1e-8
        ):
            raise ValueError("Symmetry periods are not commensurate with the lattice.")
        self._family_periods[family] = rounded.astype(int)

    def tag_period(self, family):
        self.add_site_family(family)
        periods = self._family_periods[family]
        if len(periods) != 1:
            raise NotImplementedError("Only one-dimensional lead symmetry is implemented")
        return periods[0]

    def which(self, site):
        self.add_site_family(site.family)
        periods = self._family_periods[site.family]
        coefficients = np.linalg.lstsq(periods.T, np.asarray(site.tag), rcond=None)[0]
        return tuple(np.floor(coefficients + 1e-10).astype(int))

    def act(self, element, a, b=None):
        raw_element = np.asarray(element)
        if (
            raw_element.shape != (self.num_directions,)
            or raw_element.dtype.kind not in "iu"
        ):
            raise ValueError("Group element has wrong dimension.")
        element = raw_element.astype(int)

        def shifted(site):
            self.add_site_family(site.family)
            delta = element @ self._family_periods[site.family]
            return site.family(*(np.asarray(site.tag) + delta))

        return shifted(a) if b is None else (shifted(a), shifted(b))

    def to_fd(self, a, b=None):
        element = self.which(a)
        first = self.act(tuple(-value for value in element), a)
        if b is None:
            return first
        return first, self.act(tuple(-value for value in element), b)

    def reversed(self):
        return type(self)(*(-self.periods))

    def has_subgroup(self, other):
        if isinstance(other, type(None)):
            return False
        from .builder import NoSymmetry

        if isinstance(other, NoSymmetry):
            return True
        if not isinstance(other, TranslationalSymmetry):
            return False
        coefficients = other.periods @ np.linalg.pinv(self.periods)
        rounded = np.rint(coefficients)
        return np.allclose(coefficients, rounded, atol=1e-8) and np.allclose(
            rounded @ self.periods, other.periods, atol=1e-8
        )

    def subgroup(self, *generators):
        array = np.asarray(generators)
        if (
            array.ndim != 2
            or array.shape[1] != self.num_directions
            or array.dtype.kind not in "iu"
            or np.linalg.matrix_rank(array) != array.shape[0]
        ):
            raise ValueError(
                "Subgroup generators must be independent integer sequences"
            )
        return type(self)(*(array.astype(int) @ self.periods))


def general(prim_vecs, basis=None, name="", norbs=None):
    primitive = np.asarray(prim_vecs, dtype=float)
    if basis is None:
        if norbs is not None and not np.isscalar(norbs):
            raise TypeError("norbs must be an integer for a monatomic lattice")
        return Monatomic(primitive, name=name, norbs=norbs)
    basis = np.asarray(basis, dtype=float)
    if len(basis) == 1:
        return Monatomic(primitive, basis[0], name=name, norbs=norbs)
    return Polyatomic(primitive, basis, name=name, norbs=norbs)


def chain(a=1, name="", norbs=None):
    return Monatomic([[a]], name=name, norbs=norbs)


def square(a=1, name="", norbs=None):
    return Monatomic([[a, 0], [0, a]], name=name, norbs=norbs)


def triangular(a=1, name="", norbs=None):
    return Monatomic(
        [[a, 0], [0.5 * a, np.sqrt(3) * a / 2]],
        name=name,
        norbs=norbs,
    )


def cubic(a=1, name="", norbs=None):
    return Monatomic(np.eye(3) * a, name=name, norbs=norbs)


def honeycomb(a=1, name="", norbs=None):
    primitive = [[a, 0], [0.5 * a, np.sqrt(3) * a / 2]]
    basis = [[0, 0], [0, a / np.sqrt(3)]]
    result = Polyatomic(primitive, basis, name=name, norbs=norbs)
    result.a, result.b = result.sublattices
    return result


def kagome(a=1, name="", norbs=None):
    primitive = np.asarray(
        [[a, 0], [0.5 * a, np.sqrt(3) * a / 2]],
        dtype=float,
    )
    basis = np.vstack((np.zeros(2), 0.5 * primitive))
    result = Polyatomic(primitive, basis, name=name, norbs=norbs)
    result.a, result.b, result.c = result.sublattices
    return result


__all__ = [
    "Monatomic",
    "Polyatomic",
    "TranslationalSymmetry",
    "chain",
    "cubic",
    "general",
    "honeycomb",
    "kagome",
    "square",
    "triangular",
]
