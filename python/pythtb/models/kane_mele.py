"""Kane-Mele-model constructor."""

from . import kane_mele as _kane_mele


def kane_mele(delta, t, soc, rashba):
    return _kane_mele(delta, t, soc, rashba)

__all__ = ["kane_mele"]
