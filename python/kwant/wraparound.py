"""Periodic Builder folding backed by the Thouless Rust core."""

from __future__ import annotations

import inspect
import warnings
from collections import OrderedDict, defaultdict

import numpy as np

from thouless import _core

from .builder import Builder, HermConjOfFunc, NoSymmetry
from .lattice import TranslationalSymmetry
from ._common import get_parameters


def _signature(names):
    return inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_ONLY)
            for name in names
        ]
    )


def _argument_map(names, args, kwargs, site_count=0, momenta=()):
    if len(args) > len(names):
        if kwargs:
            raise TypeError("too many arguments for wrapped value")
        source_count = len(names) - site_count - len(momenta)
        supplied = args[site_count:]
        selected = [
            *args[:site_count],
            *supplied[:source_count],
        ]
        if momenta:
            selected.extend(supplied[-len(momenta) :])
        args = tuple(selected)
    values = dict(zip(names, args, strict=False))
    for name, value in kwargs.items():
        if name not in names:
            raise TypeError(f"unexpected wrapped parameter {name!r}")
        if name in values:
            raise TypeError(f"multiple values for wrapped parameter {name!r}")
        values[name] = value
    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(f"missing wrapped parameters: {missing}")
    return values


def _matrix_value(value):
    array = np.asarray(value, dtype=complex)
    scalar = array.ndim == 0
    if scalar:
        return array.reshape(1, 1), True
    if array.ndim != 2:
        raise ValueError("periodic values must be scalars or matrices")
    return array, False


def _fold_values(values, translations, include_adjoints, momentum):
    converted = [_matrix_value(value) for value in values]
    target_shape = (1, 1)
    for array, scalar in converted:
        if not scalar:
            try:
                target_shape = np.broadcast_shapes(target_shape, array.shape)
            except ValueError as error:
                raise ValueError(
                    "periodic contributions have incompatible matrix shapes"
                ) from error
    terms = [
        (
            np.broadcast_to(array, target_shape),
            list(translation),
            bool(include_adjoint),
        )
        for (array, _), translation, include_adjoint in zip(
            converted,
            translations,
            include_adjoints,
            strict=True,
        )
    ]
    result = np.asarray(
        _core.periodic_fold_terms(terms, list(momentum)),
        dtype=complex,
    )
    if all(scalar for _, scalar in converted):
        return result[0, 0]
    return result


class _WrappedValue:
    site_count = 0

    def __init__(self, value, source_parameters, momenta):
        self.value = value
        self.source_parameters = tuple(source_parameters)
        self.momenta = tuple(momenta)
        site_names = tuple(f"_site{index}" for index in range(self.site_count))
        self.parameter_names = site_names + self.source_parameters + self.momenta
        self.__signature__ = _signature(self.parameter_names)

    def _arguments(self, args, kwargs):
        return _argument_map(
            self.parameter_names,
            args,
            kwargs,
            self.site_count,
            self.momenta,
        )

    def _source_value(self, sites, arguments):
        if not callable(self.value):
            return self.value
        parameters = [arguments[name] for name in self.source_parameters]
        return self.value(*sites, *parameters)


class _WrappedSite(_WrappedValue):
    site_count = 1

    def __call__(self, *args, **kwargs):
        arguments = self._arguments(args, kwargs)
        return self._source_value((arguments["_site0"],), arguments)


class _WrappedHopping(_WrappedValue):
    site_count = 2

    def __init__(
        self,
        value,
        source_parameters,
        momenta,
        translation,
        wrapped_symmetry,
    ):
        super().__init__(value, source_parameters, momenta)
        self.translation = tuple(int(value) for value in translation)
        self.wrapped_symmetry = wrapped_symmetry

    def __call__(self, *args, **kwargs):
        arguments = self._arguments(args, kwargs)
        first = arguments["_site0"]
        second = arguments["_site1"]
        translated_second = self.wrapped_symmetry.act(
            self.translation, second
        )
        value = self._source_value((first, translated_second), arguments)
        momentum = [arguments[name] for name in self.momenta]
        return _fold_values(
            [value],
            [self.translation],
            [False],
            momentum,
        )


class _WrappedHoppingAsSite(_WrappedValue):
    site_count = 1

    def __init__(
        self,
        value,
        source_parameters,
        momenta,
        translation,
        wrapped_symmetry,
    ):
        super().__init__(value, source_parameters, momenta)
        self.translation = tuple(int(value) for value in translation)
        self.wrapped_symmetry = wrapped_symmetry

    def __call__(self, *args, **kwargs):
        arguments = self._arguments(args, kwargs)
        first = arguments["_site0"]
        second = self.wrapped_symmetry.act(self.translation, first)
        value = self._source_value((first, second), arguments)
        momentum = [arguments[name] for name in self.momenta]
        return _fold_values(
            [value],
            [self.translation],
            [True],
            momentum,
        )


class _SumValue:
    def __init__(self, site_count, values, momenta):
        self.site_count = site_count
        self.values = tuple(values)
        self.momenta = tuple(momenta)
        site_names = tuple(f"_site{index}" for index in range(site_count))
        parameters = OrderedDict((name, None) for name in site_names)
        for value in values:
            if not callable(value):
                continue
            names = tuple(inspect.signature(value).parameters)
            for name in names[site_count:]:
                if name in self.momenta:
                    continue
                parameters.setdefault(name, None)
        for name in self.momenta:
            parameters[name] = None
        self.parameter_names = tuple(parameters)
        self.__signature__ = _signature(self.parameter_names)

    def __call__(self, *args, **kwargs):
        arguments = _argument_map(
            self.parameter_names,
            args,
            kwargs,
            self.site_count,
            self.momenta,
        )
        sites = [arguments[f"_site{index}"] for index in range(self.site_count)]
        evaluated = []
        for value in self.values:
            if callable(value):
                names = tuple(inspect.signature(value).parameters)
                selected = [*sites]
                selected.extend(arguments[name] for name in names[self.site_count :])
                evaluated.append(value(*selected))
            else:
                evaluated.append(value)
        return _fold_values(
            evaluated,
            [()] * len(evaluated),
            [False] * len(evaluated),
            (),
        )


def _source_parameters(value, site_count):
    if not callable(value):
        return ()
    return get_parameters(value)[site_count:]


def _adjoint(value):
    if callable(value):
        return HermConjOfFunc(value)
    array = np.asarray(value)
    if array.ndim == 0:
        return np.conj(value)
    return array.conj().T


class WrappedBuilder(Builder):
    """Builder carrying the symmetry metadata removed by wrapping."""

    def finalized(self):
        result = super().finalized()
        result._momentum_names = self._momentum_names
        result._wrapped_symmetry = self._wrapped_symmetry
        return result


def wraparound(builder, keep=None, *, coordinate_names="xyz"):
    """Replace selected translation generators by Bloch momenta."""
    if not isinstance(builder, Builder):
        raise TypeError("wraparound expects a Builder")
    direction_count = builder.symmetry.num_directions
    if direction_count == 0:
        raise ValueError("wraparound requires translational symmetry")
    if len(coordinate_names) < direction_count:
        raise ValueError(
            "All symmetry directions must have a name specified in coordinate_names"
        )

    momentum_names = [
        f"k_{coordinate_names[index]}" for index in range(direction_count)
    ]
    periods = list(np.asarray(builder.symmetry.periods, dtype=float))
    retained_index = None
    if keep is None:
        retained_symmetry = NoSymmetry()
        wrapped_periods = periods
    else:
        if not isinstance(keep, (int, np.integer)):
            raise TypeError("keep must be an integer symmetry direction")
        retained_index = int(keep)
        if retained_index < 0:
            retained_index += direction_count
        if not 0 <= retained_index < direction_count:
            raise ValueError("keep is not a symmetry direction")
        retained_period = periods.pop(retained_index)
        momentum_names.pop(retained_index)
        retained_symmetry = TranslationalSymmetry(retained_period)
        wrapped_periods = periods
    wrapped_symmetry = (
        TranslationalSymmetry(*wrapped_periods)
        if wrapped_periods
        else NoSymmetry()
    )
    momentum_names = tuple(momentum_names)

    result = WrappedBuilder(retained_symmetry)
    result._momentum_names = momentum_names
    result._wrapped_symmetry = builder.symmetry

    def wrapped_site_value(value):
        if not callable(value):
            return value
        return _WrappedSite(
            value,
            _source_parameters(value, 1),
            momentum_names,
        )

    result.conservation_law = wrapped_site_value(builder.conservation_law)
    result.chiral = wrapped_site_value(builder.chiral)
    if builder.particle_hole is not None or builder.time_reversal is not None:
        warnings.warn(
            "particle-hole and time-reversal symmetries are ignored after "
            "periodic directions become momentum parameters",
            RuntimeWarning,
            stacklevel=2,
        )
    result.particle_hole = None
    result.time_reversal = None

    sites = {}
    hoppings = defaultdict(list)
    for site, value in builder.site_value_pairs():
        canonical = result.symmetry.to_fd(site)
        sites[canonical] = [wrapped_site_value(value)]

    for (first, second), value in builder.hopping_value_pairs():
        full_domain = tuple(int(item) for item in builder.symmetry.which(second))
        translation = tuple(
            item
            for index, item in enumerate(full_domain)
            if index != retained_index
        )
        wrapped_second = wrapped_symmetry.act(
            tuple(-item for item in translation),
            second,
        )
        first, wrapped_second = result.symmetry.to_fd(first, wrapped_second)

        parameters = _source_parameters(value, 2)
        if first == wrapped_second:
            sites[first].append(
                _WrappedHoppingAsSite(
                    value,
                    parameters,
                    momentum_names,
                    translation,
                    wrapped_symmetry,
                )
            )
            continue

        if any(translation) or callable(value):
            folded_value = _WrappedHopping(
                value,
                parameters,
                momentum_names,
                translation,
                wrapped_symmetry,
            )
        else:
            folded_value = value

        reverse = result.symmetry.to_fd(wrapped_second, first)
        if reverse in hoppings:
            folded_value = _adjoint(folded_value)
            hoppings[reverse].append(folded_value)
        else:
            hoppings[first, wrapped_second].append(folded_value)

    for site, values in sites.items():
        result[site] = (
            values[0]
            if len(values) == 1
            else _SumValue(1, values, momentum_names)
        )
    for hopping, values in hoppings.items():
        result[hopping] = (
            values[0]
            if len(values) == 1
            else _SumValue(2, values, momentum_names)
        )
    return result


def plot_2d_bands(*args, **kwargs):
    """Plotting remains tracked separately from periodic model folding."""
    del args, kwargs
    raise NotImplementedError(
        "plot_2d_bands is not implemented; see "
        "https://github.com/matrixlab-research/thouless/issues/5"
    )


__all__ = ["WrappedBuilder", "wraparound", "plot_2d_bands"]
