"""Checkerboard-model constructor."""

from . import checkerboard as _checkerboard


def checkerboard(delta, t):
    return _checkerboard(delta, t)

__all__ = ["checkerboard"]
