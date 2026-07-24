"""Small public compatibility helpers used across Kwant modules."""

from __future__ import annotations

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


__all__ = ["KwantDeprecationWarning", "ensure_rng"]
