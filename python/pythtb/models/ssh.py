"""Su-Schrieffer-Heeger-model constructor."""

from . import ssh as _ssh


def ssh(v, w):
    return _ssh(v, w)

__all__ = ["ssh"]
