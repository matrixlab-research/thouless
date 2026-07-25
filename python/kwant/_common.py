"""Small public compatibility helpers used across Kwant modules."""

from __future__ import annotations

from contextlib import contextmanager
import inspect
import warnings

import numpy as np


class KwantDeprecationWarning(Warning):
    """Warning category for compatibility deprecations."""


def ensure_rng(rng=None):
    """Return a NumPy random generator using Kwant 1.5 seed conventions."""
    if rng is None:
        return np.random.mtrand._rand
    if isinstance(rng, (int, np.integer)):
        return np.random.RandomState(int(rng))
    if hasattr(rng, "random_sample") or hasattr(rng, "random"):
        return rng
    raise ValueError("Expecting a seed or an object that offers the numpy.random API")


def get_parameters(function):
    """Return required positional parameters of a value function."""

    parameters = inspect.signature(function).parameters
    result = []
    for name, parameter in parameters.items():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if parameter.default is not inspect.Parameter.empty:
                raise ValueError(
                    "Arguments of value functions must not have default values"
                )
            result.append(name)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            raise ValueError(
                "Keyword-only arguments are not allowed in value functions"
            )
        else:
            raise ValueError("Value functions must not take *args or **kwargs")
    return tuple(result)


@contextmanager
def reraise_warnings(level=3):
    """Re-emit warnings with a stack level that points to the public caller."""
    with warnings.catch_warnings(record=True) as caught:
        yield
    for warning in caught:
        warnings.warn(
            warning.message,
            warning.category,
            stacklevel=level,
        )


__all__ = [
    "KwantDeprecationWarning",
    "ensure_rng",
    "get_parameters",
    "reraise_warnings",
]
