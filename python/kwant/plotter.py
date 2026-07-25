"""Plotting entry points tracked as an optional compatibility gap."""

_engine = None


def set_engine(engine):
    global _engine
    if engine not in {"matplotlib", "plotly"}:
        raise RuntimeError(f"unknown plotting engine {engine!r}")
    _engine = engine


def get_engine():
    return _engine


def _unimplemented(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "Kwant plotting is not implemented; see "
        "https://github.com/matrixlab-research/thouless/issues/5"
    )


plot = _unimplemented
map = _unimplemented
bands = _unimplemented
spectrum = _unimplemented
mask_interpolate = _unimplemented


__all__ = [
    "set_engine",
    "get_engine",
    "plot",
    "map",
    "bands",
    "spectrum",
    "mask_interpolate",
]
