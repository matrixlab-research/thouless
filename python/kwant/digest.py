"""Kwant-compatible deterministic random-access variates."""

from __future__ import annotations

from thouless import _core


def _bytes(value):
    if isinstance(value, str):
        return value.encode("utf8")
    return memoryview(value).tobytes()


def uniform2(value, salt=""):
    """Return two deterministic independent values in ``[0, 1)``."""
    return _core.digest_uniform_pair(_bytes(value), _bytes(salt))


def uniform(value, salt=""):
    """Return a deterministic value uniformly distributed in ``[0, 1)``."""
    return uniform2(value, salt)[0]


def gauss(value, salt=""):
    """Return a deterministic standard-normal variate."""
    return _core.digest_gaussian(_bytes(value), _bytes(salt))


__all__ = ["gauss", "uniform", "uniform2"]
