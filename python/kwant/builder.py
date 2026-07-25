"""Kwant-compatible graph construction over the Thouless Rust core."""

from __future__ import annotations

import copy
import inspect
import itertools
import warnings
from collections import Counter, OrderedDict, deque
from functools import total_ordering, update_wrapper

import numpy as np

from thouless import _core
from .graph import Graph as _MutableGraph


class UserCodeError(RuntimeError):
    """Exception raised when a user-supplied value function fails."""


class _ParameterError(TypeError):
    """Invalid framework-level parameter binding."""


class HermConjOfFunc:
    """Hermitian-conjugated view of a hopping value function."""

    def __init__(self, function):
        self.function = function

    def __call__(self, first, second, *args, **kwargs):
        result = self.function(second, first, *args, **kwargs)
        array = np.asarray(result)
        return array.conj().T if array.ndim else np.conj(result)

    @property
    def __signature__(self):
        return inspect.signature(self.function)


class _Substituted:
    """Callable view that gives a value function new parameter names."""

    def __init__(self, func, params):
        self.func = func
        self.params = tuple(params)
        update_wrapper(self, func)

    def __eq__(self, other):
        return (
            isinstance(other, _Substituted)
            and self.func == other.func
            and self.params == other.params
        )

    def __hash__(self):
        return hash((self.func, self.params))

    @property
    def __signature__(self):
        original = inspect.signature(self.func)
        return original.replace(
            parameters=[
                parameter.replace(name=name)
                for parameter, name in zip(
                    original.parameters.values(), self.params, strict=True
                )
            ]
        )

    def __call__(self, *args, **kwargs):
        original_names = tuple(inspect.signature(self.func).parameters)
        renamed = dict(zip(self.params, original_names, strict=True))
        translated = {renamed.get(name, name): value for name, value in kwargs.items()}
        return self.func(*args, **translated)


def _substitute_params(func, substitutions):
    """Return a callable with parameter names changed simultaneously."""

    if not callable(func):
        raise TypeError("Parameter substitution requires a callable value")
    if isinstance(func, _Substituted):
        old_params = func.params
        base_func = func.func
    else:
        old_params = tuple(inspect.signature(func).parameters)
        base_func = func
    new_params = tuple(substitutions.get(name, name) for name in old_params)
    duplicates = sorted(
        name for name, count in Counter(new_params).items() if count > 1
    )
    if duplicates:
        duplicate_names = ", ".join(repr(name) for name in duplicates)
        raise ValueError(
            "Cannot rename parameters "
            f"{duplicate_names}: parameters with the same name exist"
        )
    if new_params == old_params:
        return func
    return _Substituted(base_func, new_params)


def _value_parameters(value, site_arguments):
    """Return the explicit scientific parameters of a value callable."""

    if not callable(value):
        return frozenset()
    try:
        parameters = tuple(inspect.signature(value).parameters.values())
    except (TypeError, ValueError):
        return None
    if any(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in parameters
    ):
        return None
    return frozenset(parameter.name for parameter in parameters[site_arguments:])


def _builder_parameters(builder):
    """Collect all named parameters required by a builder's value functions."""

    result = set()
    values = itertools.chain(
        ((value, 1) for value in builder._sites.values()),
        ((value, 2) for value in builder._hoppings.values()),
    )
    for value, site_arguments in values:
        parameters = _value_parameters(value, site_arguments)
        if parameters is None:
            return None
        result.update(parameters)
    return frozenset(result)


def _site_operator(specification, sites, args, params, *, default=0):
    """Assemble an onsite operator over the orbital basis of ``sites``."""

    from scipy import sparse as scipy_sparse

    orbital_counts = [site.family.norbs for site in sites]
    if any(count is None for count in orbital_counts):
        raise ValueError("Discrete symmetries require site orbital counts")
    total = sum(orbital_counts)

    if not callable(specification) and not isinstance(specification, dict):
        array = np.asarray(specification)
        if array.ndim == 0:
            return scipy_sparse.identity(total, dtype=complex) * array
        if array.shape == (total, total):
            return scipy_sparse.csr_matrix(array)

    blocks = []
    for site, orbitals in zip(sites, orbital_counts, strict=True):
        if isinstance(specification, dict):
            value = specification.get(site.family, default)
        elif callable(specification):
            value = _evaluate(specification, (site,), args, params)
        else:
            value = specification
        array = np.asarray(value)
        if array.ndim == 0:
            array = complex(array) * np.eye(orbitals)
        if array.shape != (orbitals, orbitals):
            raise ValueError(
                "Discrete-symmetry onsite block has incompatible shape"
            )
        blocks.append(scipy_sparse.csr_matrix(array))
    return scipy_sparse.block_diag(blocks, format="csr")


def _discrete_symmetry(builder, sites, args, params):
    from scipy import sparse as scipy_sparse

    from .physics import DiscreteSymmetry

    projectors = None
    if builder.conservation_law is not None:
        law = _site_operator(
            builder.conservation_law,
            sites,
            args,
            params,
        ).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(law)
        rounded = np.rint(eigenvalues)
        if not np.allclose(eigenvalues, rounded):
            raise ValueError("Conservation law must have integer eigenvalues")
        projectors = [
            scipy_sparse.csr_matrix(
                eigenvectors[:, rounded == eigenvalue]
            )
            for eigenvalue in sorted(set(rounded))
        ]

    operators = []
    for specification in (
        builder.time_reversal,
        builder.particle_hole,
        builder.chiral,
    ):
        operators.append(
            None
            if specification is None
            else _site_operator(
                specification,
                sites,
                args,
                params,
            )
        )
    return DiscreteSymmetry(projectors, *operators)


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

    def __new__(cls, family, tag, _already_normalized=False):
        # Kwant's internal fast path passes a third truthy argument when a tag
        # is already normalized. Re-normalizing is inexpensive here and keeps
        # the compatibility representation immutable and hashable.
        del _already_normalized
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


def _ensure_lead_signature(function):
    """Adapt the legacy ``(energy, args)`` lead callback convention."""

    parameters = inspect.signature(function).parameters.values()
    if any(
        parameter.name == "params"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    ):
        return function

    def wrapper(energy, args=(), *, params=None):
        return function(energy, args)

    return wrapper


class SelfEnergyLead:
    """Lead defined by a retarded self-energy callback."""

    def __init__(self, selfenergy_func, interface, parameters):
        self.selfenergy_func = _ensure_lead_signature(selfenergy_func)
        self.interface = tuple(interface)
        self.parameters = frozenset(parameters)

    def finalized(self):
        return self

    def selfenergy(self, energy, args=(), *, params=None):
        return self.selfenergy_func(energy, args, params=params)


class ModesLead:
    """Lead defined by propagating and stabilized mode callbacks."""

    _uses_stabilized_selfenergy = True

    def __init__(self, modes_func, interface, parameters):
        self.modes_func = _ensure_lead_signature(modes_func)
        self.interface = tuple(interface)
        self.parameters = frozenset(parameters)

    def finalized(self):
        return self

    def modes(self, energy, args=(), *, params=None):
        return self.modes_func(energy, args, params=params)

    def selfenergy(self, energy, args=(), *, params=None):
        stabilized = self.modes(energy, args=args, params=params)[1]
        return stabilized.selfenergy()


def _make_finalized_graph(node_count, undirected_edges, edge_values):
    graph = _MutableGraph()
    graph.num_nodes = int(node_count)
    directed_edges = []
    insertion_values = []
    for (first, second), value in zip(
        undirected_edges,
        edge_values,
        strict=True,
    ):
        directed_edges.extend(((first, second), (second, first)))
        insertion_values.extend(((value, None), (Other, None)))
    graph.add_edges(directed_edges)

    indexed = graph.compressed(edge_nr_translation=True)
    ordered_values = [None] * len(insertion_values)
    for edge_number, value in enumerate(insertion_values):
        ordered_values[indexed.edge_id(edge_number)] = value
    return graph.compressed(), ordered_values


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
        raise _ParameterError("'args' and 'params' are mutually exclusive")
    if params is None:
        return value(*sites, *args)
    signature = inspect.signature(value)
    names = list(signature.parameters)[len(sites) :]
    missing = [name for name in names if name not in params]
    if missing:
        raise _ParameterError(f"Missing required arguments: {missing}")
    return value(*sites, **{name: params[name] for name in names})


def _block(value, rows, columns, onsite=False):
    array = np.asarray(value, dtype=complex)
    if array.ndim == 0:
        if rows != columns:
            raise ValueError("scalar hopping requires equal orbital dimensions")
        result = complex(array) * np.eye(rows, dtype=complex)
    elif array.ndim == 1 and array.size == rows * columns:
        result = array.reshape(rows, columns).copy()
    elif array.shape == (rows, columns):
        result = array.copy()
    else:
        raise ValueError(
            f"value has shape {array.shape}, expected scalar or {(rows, columns)}"
        )
    if onsite and not np.allclose(result, result.conj().T):
        raise ValueError("onsite matrix is not Hermitian")
    return result


def _onsite_dimension(value, default):
    array = np.asarray(value)
    if array.ndim == 0:
        return 1 if default is None else default
    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        return array.shape[0]
    if (
        default is not None
        and array.ndim == 1
        and array.size == default * default
    ):
        return default
    raise ValueError("Onsite values must be scalars or square matrices")


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
        if keys and any(isinstance(item, Site) for item in keys) and not all(
            isinstance(item, Site) for item in keys
        ):
            raise TypeError(
                "A Builder key sequence cannot mix sites and hoppings"
            )
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
        keys = list(key)
        if keys and any(isinstance(item, Site) for item in keys) and not all(
            isinstance(item, Site) for item in keys
        ):
            raise TypeError(
                "A Builder key sequence cannot mix sites and hoppings"
            )
        for item in keys:
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

    def substituted(self, **substitutions):
        """Return a copy whose value-function parameters have new names."""

        if self.leads:
            raise ValueError(
                "Parameter substitution must be done before attaching leads"
            )

        callable_values = []
        seen = set()
        system_parameters = set()
        for values, site_arguments in (
            (self._sites.values(), 1),
            (self._hoppings.values(), 2),
        ):
            for value in values:
                if not callable(value):
                    continue
                identity = id(value)
                if identity not in seen:
                    seen.add(identity)
                    callable_values.append(value)
                parameters = _value_parameters(value, site_arguments)
                if parameters is not None:
                    system_parameters.update(parameters)

        unused = set(substitutions).difference(system_parameters)
        if unused:
            warnings.warn(
                "Parameters "
                f"{unused} are not used by any onsite or hopping value "
                "function in this system.",
                RuntimeWarning,
                stacklevel=2,
            )

        replacements = {
            id(value): _substitute_params(value, substitutions)
            for value in callable_values
        }
        result = copy.copy(self)
        result._sites = OrderedDict(
            (site, replacements.get(id(value), value))
            for site, value in self._sites.items()
        )
        result._hoppings = OrderedDict(
            (hopping, replacements.get(id(value), value))
            for hopping, value in self._hoppings.items()
        )
        return result

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
                site_was_present = site in self
                if not site_was_present:
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
        if self.symmetry.num_directions:
            raise ValueError("Leads can only be attached to finite builders")
        if not isinstance(lead_builder, Builder):
            raise ValueError("A lead must be a Builder with translational symmetry")
        if add_cells < 0 or int(add_cells) != add_cells:
            raise ValueError("add_cells must be a non-negative integer")
        if not lead_builder.symmetry.num_directions:
            raise ValueError("A lead must be a Builder with translational symmetry")
        if lead_builder.symmetry.num_directions != 1:
            raise ValueError("A lead must have exactly one translational direction")

        symmetry = lead_builder.symmetry
        hopping_range = max(
            (
                abs(int(symmetry.which(second)[0]))
                for _, second in lead_builder.hoppings()
            ),
            default=0,
        )
        if hopping_range > 1:
            expanded = Builder(
                symmetry.subgroup((hopping_range,)),
                conservation_law=lead_builder.conservation_law,
                time_reversal=lead_builder.time_reversal,
                particle_hole=lead_builder.particle_hole,
                chiral=lead_builder.chiral,
            )
            expanded.fill(
                lead_builder,
                lambda site: True,
                list(lead_builder.sites()),
                max_sites=float("inf"),
            )
            lead_builder = expanded
            symmetry = expanded.symmetry

        lead_sites = tuple(lead_builder.sites())
        if not lead_sites:
            raise ValueError("Lead to be attached contains no sites")
        lead_families = {site.family for site in lead_sites}
        system_families = {site.family for site in self.sites()}
        missing_families = lead_families.difference(system_families)
        if missing_families:
            raise ValueError(
                "Lead site families do not appear in the scattering region: "
                f"{tuple(missing_families)}"
            )

        domains = {
            int(symmetry.which(site)[0])
            for site in self.sites()
            if site.family in lead_families
            and symmetry.to_fd(site) in lead_builder
        }
        if origin is not None:
            origin_domain = int(symmetry.which(origin)[0])
            domains = {
                domain for domain in domains if domain <= origin_domain
            }
        if not domains:
            raise ValueError("Builder does not intersect with the lead")
        original_maximum_domain = max(domains)
        maximum_domain = original_maximum_domain + int(add_cells)
        minimum_domain = min(domains)

        def lead_shape(site):
            domain = int(symmetry.which(site)[0])
            if domain < minimum_domain:
                return False
            return domain <= maximum_domain + 1

        next_cell = {
            symmetry.act((maximum_domain + 1,), site)
            for site in lead_sites
        }
        all_added = self.fill(
            lead_builder,
            lead_shape,
            next_cell,
            max_sites=float("inf"),
        )
        del self[next_cell]

        interface = set()
        for site in lead_sites:
            for neighbor in lead_builder.neighbors(site):
                translated = symmetry.act(
                    (maximum_domain + 1,),
                    neighbor,
                )
                if int(symmetry.which(translated)[0]) == maximum_domain:
                    interface.add(translated)
        added = [
            site
            for site in all_added
            if site not in next_cell
            and (
                site in interface
                or original_maximum_domain
                < int(symmetry.which(site)[0])
                <= maximum_domain
            )
        ]
        unwanted = set(all_added).difference(next_cell, added)
        if unwanted:
            del self[unwanted]
        self.leads.append(
            BuilderLead(lead_builder, interface, added)
        )
        return added

    def _interface_sites(self, lead, origin=None):
        lead_sites = tuple(lead.sites())
        if not lead_sites:
            raise ValueError("lead has no sites")
        lead_orbits = set(lead_sites)
        interface_orbits = {
            site
            for site in lead_sites
            if any(
                int(lead.symmetry.which(neighbor)[0]) == 1
                for neighbor in lead.neighbors(site)
            )
        }
        candidates = [
            site
            for site in self.sites()
            if site.family in {lead_site.family for lead_site in lead_sites}
            and lead.symmetry.to_fd(site) in lead_orbits
        ]
        if not candidates:
            raise ValueError("lead has no matching interface site")
        maximum_domain = max(
            int(lead.symmetry.which(site)[0]) for site in candidates
        )
        if origin is not None:
            maximum_domain = min(
                maximum_domain,
                int(lead.symmetry.which(origin)[0]),
            )
        interface = [
            site
            for site in candidates
            if int(lead.symmetry.which(site)[0]) == maximum_domain
            and lead.symmetry.to_fd(site) in interface_orbits
        ]
        if len(interface) != len(interface_orbits):
            raise ValueError("Builder does not completely interrupt the lead")
        return sorted(interface)

    def finalized(self):
        if self.symmetry.num_directions:
            return InfiniteSystem(self)
        return FiniteSystem(self)


class InfiniteSystem:
    """Finalized one-dimensional periodic lead."""

    def __init__(self, builder, interface_order=None):
        self._builder = copy.copy(builder)
        self.symmetry = builder.symmetry
        self.parameters = _builder_parameters(builder)
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
        self.site_ranges = _site_ranges(self.sites)

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
        self.graph, self.hoppings = _make_finalized_graph(
            len(self.sites),
            undirected_edges,
            edge_values,
        )
        self.onsites = [
            (builder._sites[site], None) for site in self.sites[: self.cell_size]
        ]

    def hamiltonian(self, first, second, *args, params=None):
        if args and params is not None:
            raise TypeError("'args' and 'params' are mutually exclusive")
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
        except _ParameterError:
            raise
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

    def discrete_symmetry(self, args=(), *, params=None):
        return _discrete_symmetry(
            self._builder,
            self.sites[: self.cell_size],
            args,
            params,
        )

    def _site_dofs(self, args=(), params=None):
        return [
            _onsite_dimension(
                self.hamiltonian(index, index, *args, params=params),
                site.family.norbs,
            )
            for index, site in enumerate(self.sites)
        ]

    def _site_slices(self, args=(), params=None):
        offsets = [0]
        for dofs in self._site_dofs(args, params):
            offsets.append(offsets[-1] + dofs)
        return offsets

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
        row_sites = (
            list(range(len(self.sites))) if to_sites is None else list(to_sites)
        )
        column_sites = (
            list(range(len(self.sites)))
            if from_sites is None
            else list(from_sites)
        )
        site_dofs = self._site_dofs(args, params)
        row_norbs = [site_dofs[index] for index in row_sites]
        column_norbs = [site_dofs[index] for index in column_sites]
        result = np.zeros((sum(row_norbs), sum(column_norbs)), dtype=complex)
        row_offsets = np.cumsum([0, *row_norbs])
        column_offsets = np.cumsum([0, *column_norbs])
        for row, first in enumerate(row_sites):
            for column, second in enumerate(column_sites):
                if first != second and not self.graph.has_edge(first, second):
                    continue
                value = self.hamiltonian(first, second, *args, params=params)
                block = _block(
                    value,
                    row_norbs[row],
                    column_norbs[column],
                    onsite=first == second,
                )
                result[
                    row_offsets[row] : row_offsets[row + 1],
                    column_offsets[column] : column_offsets[column + 1],
                ] = block
        output = result
        if sparse:
            from scipy import sparse as scipy_sparse

            output = scipy_sparse.coo_matrix(result)
        if return_norb:
            return output, np.asarray(row_norbs), np.asarray(column_norbs)
        return output

    def cell_hamiltonian(self, args=(), sparse=False, *, params=None):
        cell_sites = range(self.cell_size)
        return self.hamiltonian_submatrix(
            args,
            cell_sites,
            cell_sites,
            sparse=sparse,
            params=params,
        )

    def inter_cell_hopping(self, args=(), sparse=False, *, params=None):
        cell_sites = range(self.cell_size)
        interface_sites = range(self.cell_size, self.graph.num_nodes)
        return self.hamiltonian_submatrix(
            args,
            cell_sites,
            interface_sites,
            sparse=sparse,
            params=params,
        )

    def modes(self, energy=0, args=(), *, params=None):
        from . import physics

        cell = self.cell_hamiltonian(args=args, params=params)
        hopping = self.inter_cell_hopping(args=args, params=params)
        shifted = cell - float(energy) * np.eye(cell.shape[0])
        return physics.modes(shifted, hopping)

    def _surface_selfenergy(self, energy, args, params, mode_count):
        cell = self.cell_hamiltonian(args=args, params=params)
        inter_cell = self.inter_cell_hopping(args=args, params=params)
        return np.asarray(
            _core.lead_retarded_self_energy(
                cell.tolist(),
                inter_cell.tolist(),
                energy=float(energy),
                maximum_rank=int(mode_count),
            ),
            dtype=complex,
        )

    def selfenergy(self, energy=0, args=(), *, params=None):
        from . import physics

        cell = self.cell_hamiltonian(args=args, params=params)
        hopping = self.inter_cell_hopping(args=args, params=params)
        shifted = cell - float(energy) * np.eye(cell.shape[0])
        return physics.selfenergy(shifted, hopping)

    def reversed(self):
        return self._builder.reversed().finalized()


class FiniteSystem:
    """Finalized finite graph evaluated through the Rust Hamiltonian core."""

    def __init__(self, builder):
        self._builder = copy.copy(builder)
        self.parameters = _builder_parameters(builder)
        self.sites = tuple(sorted(builder.sites()))
        self.id_by_site = {site: index for index, site in enumerate(self.sites)}
        undirected_edges = [
            (self.id_by_site[first], self.id_by_site[second])
            for first, second in builder.hoppings()
            if first in self.id_by_site and second in self.id_by_site
        ]
        edge_values = [
            value
            for (first, second), value in builder.hopping_value_pairs()
            if first in self.id_by_site and second in self.id_by_site
        ]
        self.graph, self.hoppings = _make_finalized_graph(
            len(self.sites),
            undirected_edges,
            edge_values,
        )
        self.site_ranges = _site_ranges(self.sites)
        self.onsites = [(builder._sites[site], None) for site in self.sites]
        finalized_leads = []
        lead_interfaces = []
        lead_paddings = []
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
            elif hasattr(lead, "interface"):
                try:
                    interface = [
                        self.id_by_site[site] for site in lead.interface
                    ]
                except KeyError as error:
                    raise ValueError(
                        "Lead is attached to a site that does not belong "
                        f"to the scattering region: {error.args[0]!r}"
                    ) from error
            else:
                interface = [
                    self.id_by_site[site]
                    for site in builder._interface_sites(lead)
                ]
            lead_interfaces.append(np.asarray(interface, dtype=int))
            padding = (
                lead.padding if isinstance(lead, BuilderLead) else ()
            )
            lead_paddings.append(
                np.asarray(
                    [
                        self.id_by_site[site]
                        for site in padding
                        if site in self.id_by_site
                    ],
                    dtype=int,
                )
            )
        self.leads = list(finalized_leads)
        self.lead_interfaces = tuple(lead_interfaces)
        self.lead_paddings = tuple(lead_paddings)

    def _evaluated_onsites(self, args=(), params=None):
        return [
            _evaluate(self._builder._sites[site], (site,), args, params)
            for site in self.sites
        ]

    def discrete_symmetry(self, args=(), *, params=None):
        return _discrete_symmetry(
            self._builder,
            self.sites,
            args,
            params,
        )

    def _site_dofs(self, args=(), params=None):
        return [
            _onsite_dimension(value, site.family.norbs)
            for site, value in zip(
                self.sites,
                self._evaluated_onsites(args, params),
                strict=True,
            )
        ]

    def _site_slices(self, args=(), params=None):
        offsets = [0]
        for dofs in self._site_dofs(args, params):
            offsets.append(offsets[-1] + dofs)
        return offsets

    def hamiltonian(self, first, second, *args, params=None):
        if args and params is not None:
            raise TypeError("'args' and 'params' are mutually exclusive")
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
        except _ParameterError:
            raise
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
        onsite_values = self._evaluated_onsites(args, params)
        dofs = [
            _onsite_dimension(value, site.family.norbs)
            for site, value in zip(self.sites, onsite_values, strict=True)
        ]
        offsets = [0]
        for count in dofs:
            offsets.append(offsets[-1] + count)
        site_positions = []
        for site in self.sites:
            try:
                position = np.asarray(site.pos, dtype=float)
            except AttributeError:
                position = np.empty(0, dtype=float)
            site_positions.append(position)
        dimension = max((len(position) for position in site_positions), default=0)
        primitive = np.eye(dimension)
        positions = [
            np.pad(position, (0, dimension - len(position)))
            for position in site_positions
        ]
        onsites = []
        for value, count in zip(onsite_values, dofs, strict=True):
            onsites.append(_block(value, count, count, onsite=True).tolist())
        hoppings = []
        for (first, second), value in self._builder._hoppings.items():
            if first not in self.id_by_site or second not in self.id_by_site:
                continue
            evaluated = _evaluate(value, (first, second), args, params)
            block = _block(
                evaluated,
                dofs[self.id_by_site[first]],
                dofs[self.id_by_site[second]],
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
        output = result
        if sparse:
            from scipy import sparse as scipy_sparse

            output = scipy_sparse.coo_matrix(result)
        if return_norb:
            return output, np.asarray([dofs[index] for index in selected_rows]), np.asarray(
                [dofs[index] for index in selected_columns]
            )
        return output

    def _transport_data(self, args=(), params=None):
        device = self.hamiltonian_submatrix(args=args, params=params)
        offsets = self._site_slices(args, params)
        lead_data = []
        for lead, interface in zip(
            self._builder.leads, self.lead_interfaces, strict=True
        ):
            if isinstance(lead, (ModesLead, SelfEnergyLead)):
                device_basis = np.concatenate(
                    [
                        np.arange(offsets[index], offsets[index + 1])
                        for index in interface
                    ]
                )
                interface_dimension = len(device_basis)
                cell = np.zeros(
                    (interface_dimension, interface_dimension),
                    dtype=complex,
                )
                lead_hopping = np.zeros_like(cell)
                coupling = np.zeros(
                    (device.shape[0], interface_dimension),
                    dtype=complex,
                )
                coupling[
                    np.ix_(
                        device_basis,
                        np.arange(interface_dimension),
                    )
                ] = np.eye(interface_dimension)
                lead_data.append(
                    (
                        cell.tolist(),
                        lead_hopping.tolist(),
                        coupling.tolist(),
                    )
                )
                continue
            lead_builder = lead.builder if isinstance(lead, BuilderLead) else lead
            lead_system = lead_builder.finalized()
            cell = lead_system.cell_hamiltonian(args=args, params=params)
            inter_cell = lead_system.inter_cell_hopping(args=args, params=params)
            cell_dimension = cell.shape[0]
            interface_dimension = inter_cell.shape[1]
            cell_to_previous = np.zeros(
                (cell_dimension, cell_dimension),
                dtype=complex,
            )
            cell_to_previous[:, :interface_dimension] = inter_cell
            lead_hopping = cell_to_previous.conj().T
            device_basis = np.concatenate(
                [
                    np.arange(offsets[index], offsets[index + 1])
                    for index in interface
                ]
            )
            if len(device_basis) != interface_dimension:
                raise ValueError(
                    "Lead interface orbital count does not match its principal cell"
                )
            coupling = np.zeros((device.shape[0], cell_dimension), dtype=complex)
            coupling[device_basis, :] = inter_cell.conj().T
            lead_data.append(
                (cell.tolist(), lead_hopping.tolist(), coupling.tolist())
            )
        return device, lead_data

    def precalculate(self, energy=0, args=(), leads=None, what="modes", *, params=None):
        if what not in ("modes", "selfenergy", "all"):
            raise ValueError("what must be 'modes', 'selfenergy', or 'all'")
        result = copy.copy(self)
        result._precalculated_energy = float(energy)
        result._precalculated_what = what
        return result


def _peierls_phase(field):
    """Return a straight-bond Peierls phase in symmetric gauge."""

    def phase(first, second):
        first_position = np.asarray(first.pos, dtype=float)
        second_position = np.asarray(second.pos, dtype=float)
        midpoint = 0.5 * (first_position + second_position)
        local_field = field(midpoint) if callable(field) else field
        local_field = np.asarray(local_field, dtype=float)
        if np.all(local_field == 0):
            return 1.0 + 0.0j
        if len(first_position) == 2:
            magnetic_field = (
                float(local_field)
                if local_field.ndim == 0
                else float(local_field[-1])
            )
            flux = magnetic_field * (
                first_position[0] * second_position[1]
                - first_position[1] * second_position[0]
            )
        elif len(first_position) == 3:
            magnetic_field = (
                np.full(3, float(local_field))
                if local_field.ndim == 0
                else local_field
            )
            flux = float(
                np.dot(
                    magnetic_field,
                    np.cross(first_position, second_position),
                )
            )
        else:
            raise ValueError(
                "Peierls phases require two- or three-dimensional positions"
            )
        return np.exp(1j * np.pi * flux)

    return phase


def _phase_wrapped_hopping(value, parameter):
    if callable(value):
        original_parameters = list(inspect.signature(value).parameters.values())

        def hopping(first, second, *args, **params):
            phase = params.pop(parameter)
            return value(first, second, *args, **params) * phase(first, second)

        update_wrapper(hopping, value)
    else:
        original_parameters = [
            inspect.Parameter("site1", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("site2", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]

        def hopping(first, second, *args, **params):
            if args:
                raise TypeError("Constant hopping accepts only a Peierls phase")
            phase = params.pop(parameter)
            if params:
                raise TypeError(
                    f"Unexpected hopping parameters: {sorted(params)}"
                )
            return value * phase(first, second)

    hopping.__signature__ = inspect.Signature(
        [
            *original_parameters,
            inspect.Parameter(
                parameter,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ),
        ]
    )
    return hopping


def _builder_with_peierls_phase(builder, parameter):
    result = Builder(
        builder.symmetry,
        conservation_law=builder.conservation_law,
        particle_hole=builder.particle_hole,
        chiral=builder.chiral,
    )
    result._sites = builder._sites.copy()
    result._hoppings = OrderedDict(
        (
            hopping,
            _phase_wrapped_hopping(value, parameter),
        )
        for hopping, value in builder._hoppings.items()
    )
    for index, lead in enumerate(builder.leads):
        if not isinstance(lead, BuilderLead):
            raise ValueError(
                "Peierls phase insertion requires builder-defined leads"
            )
        result.leads.append(
            BuilderLead(
                _builder_with_peierls_phase(
                    lead.builder,
                    f"{parameter}_lead{index}",
                ),
                lead.interface,
                lead.padding,
            )
        )
    return result


def add_peierls_phase(builder, peierls_parameter="phi", fix_gauge=True):
    """Insert explicit bond-phase parameters into a builder and its leads."""

    if not isinstance(builder, Builder):
        raise TypeError("add_peierls_phase expects a Builder")
    phased = _builder_with_peierls_phase(builder, peierls_parameter).finalized()
    if not fix_gauge:
        return phased

    lead_count = len(builder.leads)

    def gauge(field, *lead_fields):
        if len(lead_fields) != lead_count:
            raise ValueError(
                f"Expected {lead_count} lead magnetic fields, "
                f"received {len(lead_fields)}"
            )
        parameters = {peierls_parameter: _peierls_phase(field)}
        parameters.update(
            {
                f"{peierls_parameter}_lead{index}": _peierls_phase(lead_field)
                for index, lead_field in enumerate(lead_fields)
            }
        )
        return parameters

    return phased, gauge


__all__ = [
    "add_peierls_phase",
    "Builder",
    "BuilderLead",
    "FiniteSystem",
    "HoppingKind",
    "HermConjOfFunc",
    "InfiniteSystem",
    "ModesLead",
    "NoSymmetry",
    "Site",
    "SiteFamily",
    "SelfEnergyLead",
    "Symmetry",
    "UserCodeError",
]
