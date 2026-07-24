"""Deterministic pseudo-random values derived from immutable input data."""

from __future__ import annotations

import hashlib


def _payload(value, salt):
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode()
    else:
        data = repr(value).encode()
    return str(salt).encode() + b"\0" + data


def uniform(value, salt=""):
    """Return a deterministic value uniformly distributed in ``[0, 1)``."""
    digest = hashlib.blake2b(_payload(value, salt), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def gauss(value, salt=""):
    """Return a deterministic standard-normal variate."""
    import numpy as np

    first = max(uniform(value, f"{salt}:radius"), np.finfo(float).tiny)
    second = uniform(value, f"{salt}:angle")
    return np.sqrt(-2 * np.log(first)) * np.cos(2 * np.pi * second)


__all__ = ["gauss", "uniform"]
