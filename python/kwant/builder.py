"""Kwant-compatible graph construction over the Thouless Rust core."""

from __future__ import annotations

import copy
import inspect
import itertools
import warnings
from collections import OrderedDict, deque
from functools import total_ordering

import numpy as np

from thouless import _core


class UserCodeError(RuntimeError):
    """Exception raised when a user-supplied value function fails."""


class HermConjOfFunc:
    """Hermitian-conjugated view of a hopping value function."""

    def __init__(self, function):
        self.function = function

    def __call__(self, first, second, *args, **kwargs):
        result = self.function(second, first, *args, **kwargs)
        array = np.asarray(result)
        return array.conj().T if array.ndim else np.conj(result)


def _conjugate(value):
    if value is None:
        return None
    if isinstance(value, HermConjOfFunc):
        return value.function
    if callable(value):
        return HermConjOfFunc(value)
    if hasattr(value, "conjugate") and hasattr(value, "transpose"):
        return value.conjugate().transpose()
    array = np.asarray(value)
    return array.conj().T if array.ndim else np.conj(value)


@total_ordering
class SiteFamily:
    """Base class for immutable site families."""

    def __init__(self, canonical_repr, name, norbs):
        if norbs is not None and (not isinstance(norbs, int) or norbs <= 0):
            raise ValueError("The number of orbitals must be a positive integer.")
        self.canonical_repr = str(canonical_repr)
        self.name = name
        self.norbs = norbs

    def normalize_tag(self, tag):
        return tuple(tag)

    def __call__(self, *tag):
        if len(tag) == 1 and not isinstance(tag[0], (str, bytes)):
            try:
                normalized = self.normalize_tag(tag[0])
            except TypeError:
                normalized = self.normalize_tag(tag)
        else:
            normalized = self.normalize_tag(tag)
        return Site(self, normalized)

    def __hash__(self):
        return hash(self.canonical_repr)

    def __eq__(self, other):
        return isinstance(other, SiteFamily) and self.canonical_repr == other.canonical_repr

    def __lt__(self, other):
        if not isinstance(other, SiteFamily):
            return NotImplemented
        return self.canonical_repr < other.canonical_repr

    def __repr__(self):
        return self.canonical_repr


class Site(tuple):
    """A site-family and normalized-tag pair."""

    __slots__ = ()

    def __new__(cls, family, tag):
        return tuple.__new__(cls, (family, family.normalize_tag(tag)))

    @property
    def family(self):
        return self[0]

    @property
    def tag(self):
        return self[1]

    @property
    def pos(self):
        return self.family.pos(self.tag)

    def __repr__(self):
        return f"Site({self.family!r}, {self.tag!r})"

    def __getnewargs__(self):
        return self.family, self.tag


class HoppingKind(tuple):
    """A translationally repeated hopping between two site families."""

    __slots__ = ()

    def __new__(cls, delta, family_a, family_b=None):
        try:
            delta = tuple(int(value) for value in delta)
        except (TypeError, ValueError) as error:
            raise ValueError("HoppingKind delta must be an integer site tag") from error
        if family_b is None:
            family_b = family_a
        if not isinstance(family_a, SiteFamily) or not isinstance(family_b, SiteFamily):
            raise TypeError("HoppingKind families must be SiteFamily instances")
        try:
            family_a.normalize_tag(delta)
            family_b.normalize_tag(delta)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "HoppingKind delta is incompatible with its site families"
            ) from error
        return tuple.__new__(cls, (delta, family_a, family_b))

    @property
    def delta(self):
        return self[0]

    @property
    def family_a(self):
        return self[1]

    @property
    def family_b(self):
        return self[2]

    def __call__(self, builder):
        for site in list(builder.sites()):
            if site.family != self.family_a:
                continue
            target_tag = tuple(
                coordinate - shift
                for coordinate, shift in zip(site.tag, self.delta, strict=True)
            )
            target = self.family_b(*target_tag)
            if target in builder:
                yield site, target


class Symmetry:
    """Base class for discrete symmetries acting on sites."""

    @property
    def num_directions(self):
        raise NotImplementedError

    def which(self, site):
        raise NotImplementedError

    def act(self, element, a, b=None):
        raise NotImplementedError

    def in_fd(self, site):
        return not any(self.which(site))

    def to_fd(self, a, b=None):
        element = tuple(-value for value in self.which(a))
        return self.act(element, a) if b is None else self.act(element, a, b)

    def reversed(self):
        raise NotImplementedError


class NoSymmetry(Symmetry):
    """Identity symmetry used by finite builders."""

    num_directions = 0
    periods = np.empty((0, 0))

    def to_fd(self, a, b=None):
        return a if b is None else (a, b)

    def which(self, site):
        return ()

    def act(self, element, a, b=None):
        if tuple(element):
            raise ValueError("NoSymmetry accepts only the empty group element")
        return a if b is None else (a, b)

    def reversed(self):
        return self

    def has_subgroup(self, other):
        return isinstance(other, NoSymmetry)

    def subgroup(self, *generators):
        if generators:
            raise ValueError("NoSymmetry has no nontrivial generators")
        return self


class _Other:
    pass


Other = _Other


class BuilderLead:
    """A periodic Builder together with its scattering-region interface."""

    def __init__(self, builder, interface, padding=None):
        if not isinstance(builder, Builder):
            raise TypeError("BuilderLead requires a Builder")
        self.builder = builder
        self.interface = sorted(interface)
        self.padding = sorted(padding) if padding is not None else []

    def finalized(self):
        return InfiniteSystem(self.builder, self.interface)


class _Graph:
    def __init__(self, node_count, undirected_edges):
        self.num_nodes = int(node_count)
        self._edges = []
        self._edge_ids = {}
        for first, second in undirected_edges:
            for edge in ((first, second), (second, first)):
                self._edge_ids[edge] = len(self._edges)
                self._edges.append(edge)
        self.num_edges = len(self._edges)

    def __iter__(self):
        return iter(self._edges)

    def out_neighbors(self, node):
        return iter(second for first, second in self._edges if first == node)

    def has_edge(self, first, second):
        return (first, second) in self._edge_ids

    def first_edge_id(self, first, second):
        return self._edge_ids[(first, second)]


def _site_ranges(sites):
    sites = tuple(sites)
    if any(site.family.norbs is None for site in sites):
        return None
    ranges = []
    orbital_offset = 0
    previous_family = None
    for index, site in enumerate(sites):
        if site.family != previous_family:
            ranges.append((index, site.family.norbs, orbital_offset))
            previous_family = site.family
        orbital_offset += site.family.norbs
    ranges.append((len(sites), 0, orbital_offset))
    return ranges


def _evaluate(value, sites, args, params):
    if not callable(value):
        return value
    if args and params is not None:
        raise TypeError("'args' and 'params' are mutually exclusive")
    if params is None:
        return value(*sites, *args)
    signature = inspect.signature(value)
    names = list(signature.parameters)[len(sites) :]
    missing = [name for name in names if name not in params]
    if missing:
        raise TypeError(f"Missing required arguments: {missing}")
    return value(*sites, **{name: params[name] for name in names})


def _block(value, rows, columns, onsite=False):
    array = np.asarray(value, dtype=complex)
    if array.ndim == 0:
        if rows != columns:
            raise ValueError("scalar hopping requires equal orbital dimensions")
        result = complex(array) * np.eye(rows, dtype=complex)
    elif array.shape == (rows, columns):
        result = array.copy()
    else:
        raise ValueError(
            f"value has shape {array.shape}, expected scalar or {(rows, columns)}"
        )
    if onsite and not np.allclose(result, result.conj().T):
        raise ValueError("onsite matrix is not Hermitian")
    return result


class Builder:
    """Mutable site graph with optional translational symmetry."""

    def __init__(
        self,
        symmetry=None,
        conservation_law=None,
        time_reversal=None,
        particle_hole=None,
        chiral=None,
    ):
        self.symmetry = NoSymmetry() if symmetry is None else symmetry
        self.conservation_law = conservation_law
        self.time_reversal = time_reversal
        self.particle_hole = particle_hole
        self.chiral = chiral
        self._sites = OrderedDict()
        self._hoppings = OrderedDict()
        self.leads = []

    def __copy__(self):
        result = type(self)(
            self.symmetry,
            self.conservation_law,
            self.time_reversal,
            self.particle_hole,
            self.chiral,
        )
        result._sites = self._sites.copy()
        result._hoppings = self._hoppings.copy()
        result.leads = self.leads.copy()
        return result

    def _canonical_site(self, site):
        if not isinstance(site, Site):
            raise TypeError("Builder site keys must be Site instances")
        return self.symmetry.to_fd(site)

    def _stored_hopping(self, first, second):
        direct = self.symmetry.to_fd(first, second)
        reverse = self.symmetry.to_fd(second, first)
        if direct in self._hoppings:
            return direct, False
        if reverse in self._hoppings:
            return reverse, True
        return direct, False

    def _validate_hopping(self, key):
        if not isinstance(key, tuple):
            raise TypeError("Builder hopping keys must be pairs of sites")
        if len(key) != 2:
            raise IndexError("Builder hopping keys must contain exactly two sites")
        if not all(isinstance(site, Site) for site in key):
            raise TypeError("Builder hopping keys must be pairs of sites")
        direct = self.symmetry.to_fd(*key)
        if direct[0] == direct[1]:
            raise ValueError("A hopping cannot connect a site to itself.")
        return key

    def __contains__(self, key):
        if isinstance(key, Site):
            return self._canonical_site(key) in self._sites
        first, second = self._validate_hopping(key)
        stored, _ = self._stored_hopping(first, second)
        return stored in self._hoppings

    def __getitem__(self, key):
        if isinstance(key, Site):
            return self._sites[self._canonical_site(key)]
        first, second = self._validate_hopping(key)
        stored, reversed_hopping = self._stored_hopping(first, second)
        value = self._hoppings[stored]
        return _conjugate(value) if reversed_hopping else value

    def __setitem__(self, key, value):
        if isinstance(key, Site):
            self._sites[self._canonical_site(key)] = value
            return
        if isinstance(key, HoppingKind):
            for hopping in key(self):
                self[hopping] = value
            return
        if callable(key):
            self[key(self)] = value
            return
        if isinstance(key, tuple):
            raw_first, raw_second = self._validate_hopping(key)
            direct = self.symmetry.to_fd(raw_first, raw_second)
            if self.symmetry.to_fd(raw_first) not in self._sites:
                raise KeyError(raw_first)
            if self.symmetry.to_fd(raw_second) not in self._sites:
                raise KeyError(raw_second)
            stored, reversed_hopping = self._stored_hopping(raw_first, raw_second)
            if reversed_hopping:
                del self._hoppings[stored]
            self._hoppings[direct] = value
            return
        try:
            keys = list(key)
        except TypeError as error:
            raise TypeError("Builder key is not a site, hopping, or iterable") from error
        for item in keys:
            self[item] = value

    def __delitem__(self, key):
        if isinstance(key, Site):
            site = self._canonical_site(key)
            del self._sites[site]
            for hopping in list(self._hoppings):
                if any(self._canonical_site(endpoint) == site for endpoint in hopping):
                    del self._hoppings[hopping]
            return
        if isinstance(key, tuple):
            first, second = self._validate_hopping(key)
            stored, _ = self._stored_hopping(first, second)
            del self._hoppings[stored]
            return
        if callable(key):
            del self[key(self)]
            return
        for item in list(key):
            del self[item]

    def sites(self):
        return iter(self._sites)

    def hoppings(self):
        return iter(self._hoppings)

    def site_value_pairs(self):
        return iter(self._sites.items())

    def hopping_value_pairs(self):
        return iter(self._hoppings.items())

    def neighbors(self, site):
        if not isinstance(site, Site):
            raise TypeError("Builder.neighbors expects a Site")
        canonical = self._canonical_site(site)
        if canonical not in self._sites:
            raise KeyError(site)
        site_shift = tuple(int(value) for value in self.symmetry.which(site))
        yielded = set()
        for first, second in self._hoppings:
            for endpoint, neighbor in ((first, second), (second, first)):
                if self._canonical_site(endpoint) != canonical:
                    continue
                endpoint_shift = tuple(
                    int(value) for value in self.symmetry.which(endpoint)
                )
                shift = tuple(
                    target - source
                    for target, source in zip(
                        site_shift, endpoint_shift, strict=True
                    )
                )
                translated = self.symmetry.act(shift, neighbor)
                if translated not in yielded:
                    yielded.add(translated)
                    yield translated

    def degree(self, site):
        return sum(1 for _ in self.neighbors(site))

    def expand(self, key):
        stack = [iter((key,))]
        while stack:
            try:
                item = next(stack[-1])
            except StopIteration:
                stack.pop()
                continue
            while callable(item):
                item = item(self)
            if isinstance(item, tuple):
                yield item
                continue
            try:
                stack.append(iter(item))
            except TypeError as error:
                raise TypeError(
                    f"{type(item).__name__} object is not a valid key"
                ) from error

    def dangling(self):
        return (site for site in self.sites() if self.degree(site) < 2)

    def eradicate_dangling(self):
        while True:
            dangling = list(self.dangling())
            if not dangling:
                return
            del self[dangling]

    def reversed(self):
        result = copy.copy(self)
        result.symmetry = self.symmetry.reversed()
        return result

    def update(self, other):
        if not isinstance(other, Builder):
            raise TypeError("Builder.update expects another Builder")
        if type(self.symmetry) is not type(other.symmetry):
            raise ValueError("Both builders must have compatible symmetries")
        for site, value in other.site_value_pairs():
            self[site] = value
        for hopping, value in other.hopping_value_pairs():
            self[hopping] = value
        self.leads.extend(other.leads)

    def closest(self, position):
        position = np.asarray(position, dtype=float)
        if not self._sites:
            raise ValueError("Builder is empty")
        best_site = None
        best_distance = np.inf
        for site in self._sites:
            try:
                site_position = np.asarray(site.pos, dtype=float)
            except AttributeError as error:
                raise AttributeError(
                    "Builder.closest requires site families with positions"
                ) from error
            if site_position.shape != position.shape:
                raise ValueError("Position has wrong dimensionality")
            if not self.symmetry.num_directions:
                candidates = [site]
            else:
                coefficients = np.linalg.lstsq(
                    self.symmetry.periods.T,
                    position - site_position,
                    rcond=None,
                )[0]
                center = np.rint(coefficients).astype(int)
                candidates = [
                    self.symmetry.act(center + np.asarray(delta), site)
                    for delta in itertools.product(
                        range(-3, 4), repeat=self.symmetry.num_directions
                    )
                ]
            for candidate in candidates:
                distance = np.linalg.norm(
                    np.asarray(candidate.pos, dtype=float) - position
                )
                if distance < best_distance:
                    best_distance = distance
                    best_site = candidate
        return best_site

    def fill(self, template, shape, start, *, max_sites=10**7):
        if not isinstance(template, Builder):
            raise TypeError("fill template must be a Builder")
        if max_sites <= 0:
            raise ValueError("max_sites must be positive")
        if not template.symmetry.has_subgroup(self.symmetry):
            raise ValueError(
                "Builder symmetry is not a subgroup of the template symmetry"
            )

        if isinstance(start, Site):
            starts = [start]
        else:
            starts = list(start)
            if starts and not isinstance(starts[0], Site):
                starts = [template.closest(start)]
        if any(site not in template for site in starts):
            warnings.warn(
                "fill(): Some starting sites are not in the template builder.",
                RuntimeWarning,
                stacklevel=2,
            )
        starts = [site for site in starts if site in template]
        if not starts:
            return []

        canonical_starts = [self.symmetry.to_fd(site) for site in starts]
        if all(site in self for site in canonical_starts):
            warnings.warn(
                "fill(): The target builder already contains all starting sites.",
                RuntimeWarning,
                stacklevel=2,
            )
            return []
        inside_starts = [
            site for site in canonical_starts if site not in self and shape(site)
        ]
        if not inside_starts:
            warnings.warn(
                "fill(): None of the starting sites is in the desired shape.",
                RuntimeWarning,
                stacklevel=2,
            )
            return []

        original_sites = self._sites.copy()
        original_hoppings = self._hoppings.copy()
        queue = deque(inside_starts)
        queued = set(inside_starts)
        processed = set()
        added = []
        try:
            while queue:
                site = queue.popleft()
                if site in processed:
                    continue
                processed.add(site)
                if site not in self:
                    self[site] = template[site]
                    added.append(site)
                    if len(added) > max_sites:
                        raise RuntimeError(
                            "Maximal number of sites specified by max_sites exceeded"
                        )

                template_site = template.symmetry.to_fd(site)
                site_shift = tuple(
                    int(value) for value in template.symmetry.which(site)
                )
                incident = []
                for (
                    stored_first,
                    stored_second,
                ), hopping_value in template.hopping_value_pairs():
                    for endpoint in (stored_first, stored_second):
                        if template.symmetry.to_fd(endpoint) != template_site:
                            continue
                        endpoint_shift = tuple(
                            int(value)
                            for value in template.symmetry.which(endpoint)
                        )
                        shift = tuple(
                            target - source
                            for target, source in zip(
                                site_shift, endpoint_shift, strict=True
                            )
                        )
                        actual_first, actual_second = template.symmetry.act(
                            shift, stored_first, stored_second
                        )
                        neighbor = (
                            actual_second
                            if endpoint == stored_first
                            else actual_first
                        )
                        incident.append(
                            (
                                neighbor,
                                (actual_first, actual_second),
                                hopping_value,
                            )
                        )
                for neighbor, hopping, hopping_value in incident:
                    neighbor_fd = self.symmetry.to_fd(neighbor)
                    if neighbor_fd not in self and neighbor_fd not in queued:
                        if not shape(neighbor_fd):
                            continue
                        queue.append(neighbor_fd)
                        queued.add(neighbor_fd)
                    elif (
                        neighbor_fd in self
                        and neighbor_fd not in processed
                        and neighbor_fd not in queued
                        and shape(neighbor_fd)
                    ):
                        queue.append(neighbor_fd)
                        queued.add(neighbor_fd)
                    if neighbor_fd in self or neighbor_fd in queued:
                        if neighbor_fd not in self:
                            self[neighbor_fd] = template[neighbor]
                            added.append(neighbor_fd)
                            if len(added) > max_sites:
                                raise RuntimeError(
                                    "Maximal number of sites specified by "
                                    "max_sites exceeded"
                                )
                        if hopping not in self:
                            self[hopping] = hopping_value
            return added
        except Exception:
            self._sites = original_sites
            self._hoppings = original_hoppings
            raise

    def attach_lead(self, lead_builder, origin=None, add_cells=0):
        if not isinstance(lead_builder, Builder) or not lead_builder.symmetry.num_directions:
            raise ValueError("A lead must be a Builder with translational symmetry")
        interface = self._interface_site(lead_builder)
        self.leads.append(BuilderLead(copy.copy(lead_builder), [interface]))
        return []

    def _interface_site(self, lead):
        lead_sites = tuple(lead.sites())
        if not lead_sites:
            raise ValueError("lead has no sites")
        families = {site.family for site in lead_sites}
        candidates = [site for site in self.sites() if site.family in families]
        if not candidates:
            raise ValueError("lead has no matching interface site")
        first_family = lead_sites[0].family
        tag_period = lead.symmetry.tag_period(first_family)
        return max(candidates, key=lambda site: np.dot(site.tag, tag_period))

    def finalized(self):
        if self.symmetry.num_directions:
            return InfiniteSystem(self)
        return FiniteSystem(self)


class InfiniteSystem:
    """Finalized one-dimensional periodic lead."""

    def __init__(self, builder, interface_order=None):
        self._builder = copy.copy(builder)
        self.symmetry = builder.symmetry
        if self.symmetry.num_directions != 1:
            raise ValueError(
                "Infinite systems require exactly one translational direction"
            )

        with_interface = []
        without_interface = []
        for site in builder.sites():
            if any(
                int(self.symmetry.which(neighbor)[0]) == 1
                for neighbor in builder.neighbors(site)
            ):
                with_interface.append(site)
            else:
                without_interface.append(site)

        with_interface.sort()
        without_interface.sort()
        if interface_order is None:
            previous = [self.symmetry.act((-1,), site) for site in with_interface]
            previous.sort()
        else:
            interface_order = list(interface_order)
            if not interface_order and with_interface:
                raise ValueError("interface_order did not contain all interface sites")
            if interface_order != sorted(interface_order):
                raise ValueError("Interface sites must be sorted.")
            previous = []
            ordered_cell = []
            shift = None
            for site in interface_order:
                site_shift = int(self.symmetry.which(site)[0])
                current_shift = -site_shift - 1
                if shift is None:
                    shift = current_shift
                elif shift != current_shift:
                    raise ValueError(
                        "The sites in interface_order do not all belong "
                        "to the same lead cell."
                    )
                previous_site = self.symmetry.act((current_shift,), site)
                previous.append(previous_site)
                ordered_cell.append(self.symmetry.act((1,), previous_site))
            if (
                len(ordered_cell) != len(with_interface)
                or set(ordered_cell) != set(with_interface)
            ):
                raise ValueError(
                    "interface_order did not contain all interface sites"
                )
            with_interface = ordered_cell
        if not with_interface:
            warnings.warn(
                "Infinite system with disconnected cells.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.cell_size = len(with_interface) + len(without_interface)
        self.sites = tuple(with_interface + without_interface + previous)
        self.id_by_site = {site: index for index, site in enumerate(self.sites)}

        undirected_edges = []
        edge_values = []
        for (first, second), value in builder.hopping_value_pairs():
            cell = int(self.symmetry.which(second)[0])
            if cell == 1:
                actual_first, actual_second = self.symmetry.act(
                    (-1,), first, second
                )
            elif cell in (-1, 0):
                actual_first, actual_second = first, second
            else:
                raise ValueError(
                    "Further-than-nearest-neighbor cells are connected "
                    f"by hopping {(first, second)!r}."
                )
            try:
                edge = (
                    self.id_by_site[actual_first],
                    self.id_by_site[actual_second],
                )
            except KeyError as error:
                raise ValueError(
                    f"Cannot represent periodic hopping {(first, second)!r}"
                ) from error
            undirected_edges.append(edge)
            edge_values.append(value)
        self.graph = _Graph(len(self.sites), undirected_edges)
        self.onsites = [
            (builder._sites[site], None) for site in self.sites[: self.cell_size]
        ]
        self.hoppings = []
        for value in edge_values:
            self.hoppings.append((value, None))
            self.hoppings.append((Other, None))

    def hamiltonian(self, first, second, *args, params=None):
        first = int(first)
        second = int(second)
        first_site = self.sites[first]
        second_site = self.sites[second]
        if first == second and first >= self.cell_size:
            first_site = second_site = self.symmetry.to_fd(first_site)
        elif first >= self.cell_size:
            first_site, second_site = self.symmetry.to_fd(first_site, second_site)
        try:
            if first == second:
                return _evaluate(
                    self._builder._sites[first_site],
                    (first_site,),
                    args,
                    params,
                )
            return _evaluate(
                self._builder[first_site, second_site],
                (first_site, second_site),
                args,
                params,
            )
        except Exception as error:
            function = (
                self._builder._sites[first_site]
                if first == second
                else self._builder[first_site, second_site]
            )
            function = (
                function.function if isinstance(function, HermConjOfFunc) else function
            )
            name = getattr(function, "__name__", type(function).__name__)
            raise UserCodeError(
                f'Error occurred in user-supplied value function "{name}"'
            ) from error

    def reversed(self):
        return self._builder.reversed().finalized()


class FiniteSystem:
    """Finalized finite graph evaluated through the Rust Hamiltonian core."""

    def __init__(self, builder):
        self._builder = copy.copy(builder)
        self.sites = tuple(sorted(builder.sites()))
        self.id_by_site = {site: index for index, site in enumerate(self.sites)}
        undirected_edges = [
            (self.id_by_site[first], self.id_by_site[second])
            for first, second in builder.hoppings()
            if first in self.id_by_site and second in self.id_by_site
        ]
        self.graph = _Graph(len(self.sites), undirected_edges)
        self.site_ranges = _site_ranges(self.sites)
        self.onsites = [(builder._sites[site], None) for site in self.sites]
        self.hoppings = []
        for (first, second), value in builder.hopping_value_pairs():
            if first not in self.id_by_site or second not in self.id_by_site:
                continue
            self.hoppings.append((value, None))
            self.hoppings.append((Other, None))
        finalized_leads = []
        lead_interfaces = []
        for lead in builder.leads:
            finalized_leads.append(lead.finalized())
            if isinstance(lead, BuilderLead):
                try:
                    interface = [self.id_by_site[site] for site in lead.interface]
                except KeyError as error:
                    raise ValueError(
                        "Lead is attached to a site that does not belong "
                        f"to the scattering region: {error.args[0]!r}"
                    ) from error
            else:
                interface = [self.id_by_site[builder._interface_site(lead)]]
            lead_interfaces.append(np.asarray(interface, dtype=int))
        self.leads = tuple(finalized_leads)
        self.lead_interfaces = tuple(lead_interfaces)

    def _site_slices(self):
        offsets = [0]
        for site in self.sites:
            if site.family.norbs is None:
                raise ValueError("Number of orbitals not defined for a site family")
            offsets.append(offsets[-1] + site.family.norbs)
        return offsets

    def hamiltonian(self, first, second, *args, params=None):
        first = int(first)
        second = int(second)
        first_site = self.sites[first]
        second_site = self.sites[second]
        try:
            if first == second:
                return _evaluate(
                    self._builder._sites[first_site],
                    (first_site,),
                    args,
                    params,
                )
            return _evaluate(
                self._builder[first_site, second_site],
                (first_site, second_site),
                args,
                params,
            )
        except Exception as error:
            function = (
                self._builder._sites[first_site]
                if first == second
                else self._builder[first_site, second_site]
            )
            function = (
                function.function if isinstance(function, HermConjOfFunc) else function
            )
            name = getattr(function, "__name__", type(function).__name__)
            raise UserCodeError(
                f'Error occurred in user-supplied value function "{name}"'
            ) from error

    def hamiltonian_submatrix(
        self,
        args=(),
        to_sites=None,
        from_sites=None,
        sparse=False,
        return_norb=False,
        *,
        params=None,
    ):
        if sparse:
            raise NotImplementedError("sparse matrix output is not implemented yet")
        offsets = self._site_slices()
        dimension = max((len(site.pos) for site in self.sites), default=0)
        primitive = np.eye(dimension)
        positions = [
            np.pad(np.asarray(site.pos, dtype=float), (0, dimension - len(site.pos)))
            for site in self.sites
        ]
        dofs = [site.family.norbs for site in self.sites]
        onsites = []
        for site in self.sites:
            value = _evaluate(self._builder._sites[site], (site,), args, params)
            onsites.append(
                _block(value, site.family.norbs, site.family.norbs, onsite=True).tolist()
            )
        hoppings = []
        for (first, second), value in self._builder._hoppings.items():
            if first not in self.id_by_site or second not in self.id_by_site:
                continue
            evaluated = _evaluate(value, (first, second), args, params)
            block = _block(
                evaluated,
                first.family.norbs,
                second.family.norbs,
            )
            hoppings.append(
                (
                    self.id_by_site[first],
                    self.id_by_site[second],
                    [0] * dimension,
                    block.tolist(),
                )
            )
        matrix = np.asarray(
            _core.hamiltonian(
                primitive.tolist(),
                [],
                [position.tolist() for position in positions],
                dofs,
                onsites,
                hoppings,
                [],
            ),
            dtype=complex,
        )
        selected_rows = list(range(len(self.sites))) if to_sites is None else list(to_sites)
        selected_columns = (
            list(range(len(self.sites))) if from_sites is None else list(from_sites)
        )
        row_basis = np.concatenate(
            [np.arange(offsets[index], offsets[index + 1]) for index in selected_rows]
        )
        column_basis = np.concatenate(
            [np.arange(offsets[index], offsets[index + 1]) for index in selected_columns]
        )
        result = matrix[np.ix_(row_basis, column_basis)]
        if return_norb:
            return result, np.asarray([dofs[index] for index in selected_rows]), np.asarray(
                [dofs[index] for index in selected_columns]
            )
        return result

    def _transport_data(self, args=(), params=None):
        device = self.hamiltonian_submatrix(args=args, params=params)
        offsets = self._site_slices()
        lead_data = []
        for lead, interface in zip(
            self._builder.leads, self.lead_interfaces, strict=True
        ):
            lead_builder = lead.builder if isinstance(lead, BuilderLead) else lead
            lead_system = lead_builder.finalized()
            lead_sites = lead_system.sites[: lead_system.cell_size]
            if len(lead_sites) != 1:
                raise NotImplementedError(
                    "multi-site principal lead cells are not implemented yet"
                )
            lead_site = lead_sites[0]
            onsite_value = _evaluate(
                lead_builder._sites[lead_site], (lead_site,), args, params
            )
            cell = _block(
                onsite_value,
                lead_site.family.norbs,
                lead_site.family.norbs,
                onsite=True,
            )
            hopping_value = next(iter(lead_builder._hoppings.values()))
            hopping_sites = next(iter(lead_builder._hoppings))
            hopping = _block(
                _evaluate(hopping_value, hopping_sites, args, params),
                lead_site.family.norbs,
                lead_site.family.norbs,
            )
            coupling = np.zeros(
                (device.shape[0], lead_site.family.norbs), dtype=complex
            )
            interface_site = int(interface[0])
            start, stop = offsets[interface_site : interface_site + 2]
            coupling[start:stop] = hopping
            lead_data.append((cell.tolist(), hopping.tolist(), coupling.tolist()))
        return device, lead_data


__all__ = [
    "Builder",
    "BuilderLead",
    "FiniteSystem",
    "HoppingKind",
    "HermConjOfFunc",
    "InfiniteSystem",
    "NoSymmetry",
    "Site",
    "SiteFamily",
    "Symmetry",
    "UserCodeError",
]
