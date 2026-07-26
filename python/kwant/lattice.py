"""Kwant lattice families and translational symmetry."""

from __future__ import annotations

import itertools
from collections import deque

import numpy as np
import tinyarray as ta
from thouless import _core

from .builder import HoppingKind, SiteFamily, Symmetry


class Monatomic(SiteFamily):
    """One Bravais lattice with one basis site."""

    def __init__(self, prim_vecs, offset=None, name="", norbs=None):
        primitive = np.asarray(prim_vecs, dtype=float)
        if primitive.ndim != 2 or primitive.shape[0] == 0:
            raise ValueError("Primitive vectors must form a nonempty 2D array.")
        self.prim_vecs = primitive
        self._prim_vecs = primitive
        self.lattice_dim = primitive.shape[0]
        self.space_dim = primitive.shape[1]
        self.offset = (
            np.zeros(self.space_dim)
            if offset is None
            else np.asarray(offset, dtype=float)
        )
        if self.offset.shape != (self.space_dim,):
            raise ValueError("Basis offset has wrong dimension.")
        _core.validate_embedded_lattice(
            self.prim_vecs.tolist(),
            [self.offset.tolist()],
        )
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
        tag = self.normalize_tag(tag)
        return ta.array(
            _core.embedded_lattice_position(
                self.prim_vecs.tolist(),
                self.offset.tolist(),
                list(tag),
            )
        )

    def vec(self, tag):
        tag = self.normalize_tag(tag)
        return ta.array(
            _core.embedded_lattice_position(
                self.prim_vecs.tolist(),
                [0.0] * self.space_dim,
                list(tag),
            )
        )

    def closest(self, position):
        return ta.array(self.n_closest(position)[0], int)

    def n_closest(
        self,
        position,
        n=1,
        group_by_length=False,
        rtol=1e-9,
    ):
        """Return lattice coordinates of the nearest sites."""
        position = np.asarray(position, dtype=float)
        if position.shape != (self.space_dim,):
            raise ValueError("Position has wrong dimensionality.")
        from .linalg.lll import cvp

        return cvp(
            position - self.offset,
            self.prim_vecs,
            n=n,
            group_by_length=group_by_length,
            rtol=rtol,
        )

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
        if basis.ndim != 2 or basis.shape[1] != primitive.shape[1]:
            raise ValueError("Basis positions have wrong dimensionality.")
        _core.validate_embedded_lattice(
            primitive.tolist(),
            basis.tolist(),
        )
        self.prim_vecs = primitive
        self._prim_vecs = primitive
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
        basis = [
            sublattice.offset.tolist()
            for sublattice in self.sublattices
        ]
        relations = _core.embedded_lattice_neighbors(
            self.prim_vecs.tolist(),
            basis,
            n,
            float(eps),
        )
        result = [
            HoppingKind(
                displacement,
                self.sublattices[first],
                self.sublattices[second],
            )
            for displacement, first, second in relations
        ]
        return sorted(
            result,
            key=lambda hopping: (
                hopping.family_a.canonical_repr,
                hopping.family_b.canonical_repr,
                hopping.delta,
            ),
        )


class TranslationalSymmetry(Symmetry):
    """Discrete translation group represented by Cartesian periods."""

    def __init__(self, *periods):
        array = np.asarray(periods, dtype=float)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("At least one translation period is required.")
        _core.validate_embedded_lattice(
            array.tolist(),
            [[0.0] * array.shape[1]],
        )
        self.periods = array
        self.site_family_data = {}
        self.is_reversed = False

    @property
    def num_directions(self):
        return len(self.periods)

    def add_site_family(self, family, other_vectors=None):
        if family in self.site_family_data:
            return
        primitive = np.asarray(family.prim_vecs, dtype=float)
        if self.periods.shape[1] != primitive.shape[1]:
            raise ValueError(
                "Lattice and symmetry have different spatial dimensions"
            )
        if other_vectors is not None:
            other_vectors = np.asarray(other_vectors)
            if other_vectors.ndim != 2:
                raise ValueError("other_vectors must be a two-dimensional array")
            if not np.all(other_vectors == np.rint(other_vectors)):
                raise ValueError("other_vectors must contain only integers")
            other_vectors = other_vectors.astype(int).tolist()
        else:
            other_vectors = []
        period_vectors, adjugate_rows, determinant = (
            _core.translation_fundamental_domain(
                primitive.tolist(),
                self.periods.tolist(),
                other_vectors,
                1e-8,
            )
        )
        self.site_family_data[family] = (
            ta.array(np.asarray(period_vectors, dtype=int).T, int),
            ta.array(adjugate_rows, int),
            determinant,
        )

    def tag_period(self, family):
        self.add_site_family(family)
        periods = self.site_family_data[family][0]
        if periods.shape[1] != 1:
            raise NotImplementedError("Only one-dimensional lead symmetry is implemented")
        period = np.asarray(periods)[:, 0]
        return -period if self.is_reversed else period

    def which(self, site):
        self.add_site_family(site.family)
        _, adjugate_rows, determinant = self.site_family_data[
            site.family
        ]
        result = _core.translation_group_coordinates(
            np.asarray(adjugate_rows, dtype=int).tolist(),
            int(determinant),
            list(site.tag),
        )
        if self.is_reversed:
            result = [-value for value in result]
        return ta.array(result, int)

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
            periods = np.asarray(
                self.site_family_data[site.family][0], dtype=int
            )
            delta = np.asarray(
                _core.translation_group_shift(
                    periods.T.tolist(),
                    element.tolist(),
                ),
                dtype=int,
            )
            if self.is_reversed:
                delta = -delta
            return site.family(*(np.asarray(site.tag) + delta))

        return shifted(a) if b is None else (shifted(a), shifted(b))

    def to_fd(self, a, b=None):
        element = self.which(a)
        first = self.act(tuple(-value for value in element), a)
        if b is None:
            return first
        return first, self.act(tuple(-value for value in element), b)

    def reversed(self):
        result = type(self)(*(-self.periods))
        result.site_family_data = self.site_family_data
        result.is_reversed = not self.is_reversed
        return result

    def has_subgroup(self, other):
        if isinstance(other, type(None)):
            return False
        from .builder import NoSymmetry

        if isinstance(other, NoSymmetry):
            return True
        if not isinstance(other, TranslationalSymmetry):
            return False
        return _core.translation_contains_subgroup(
            self.periods.tolist(),
            other.periods.tolist(),
            1e-8,
        )

    def subgroup(self, *generators):
        array = np.asarray(generators)
        if (
            array.ndim != 2
            or array.shape[1] != self.num_directions
            or array.dtype.kind not in "iu"
        ):
            raise ValueError(
                "Subgroup generators must be independent integer sequences"
            )
        _core.validate_embedded_lattice(
            array.astype(float).tolist(),
            [[0.0] * self.num_directions],
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
        if norbs is not None and not np.isscalar(norbs):
            if len(norbs) != 1:
                raise ValueError("norbs must match the number of basis sites")
            norbs = norbs[0]
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
