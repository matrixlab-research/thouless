"""Fu-Kane-Mele-model constructor."""

from . import fu_kane_mele as _fu_kane_mele


def fu_kane_mele(t, soc, dt=(0.0, 0.0, 0.0, 0.0)):
    return _fu_kane_mele(t, soc, dt)

__all__ = ["fu_kane_mele"]
