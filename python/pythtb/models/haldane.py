"""Haldane-model constructor."""

from . import haldane as _haldane


def haldane(delta, t1, t2, phi=None):
    if phi is None:
        return _haldane(delta, t1, t2)
    return _haldane(delta, t1, t2, phi)

__all__ = ["haldane"]
