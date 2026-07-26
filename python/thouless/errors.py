"""Stable exceptions raised by the public Thouless Python API."""


class ThoulessError(Exception):
    """Base class for public Thouless failures."""


class InvalidInputError(ThoulessError, ValueError):
    """A scientific value or geometry is invalid."""


class ShapeError(InvalidInputError):
    """An array has an incompatible shape or dtype."""


class NumericalError(ThoulessError, RuntimeError):
    """A numerical solve did not produce a valid result."""


class UnsupportedError(ThoulessError, NotImplementedError):
    """The requested backend or feature is unsupported."""


class ResourceError(ThoulessError, MemoryError):
    """The requested operation exceeds an explicit resource boundary."""


class InternalError(ThoulessError, RuntimeError):
    """A native invariant failed unexpectedly."""


__all__ = [
    "InternalError",
    "InvalidInputError",
    "NumericalError",
    "ResourceError",
    "ShapeError",
    "ThoulessError",
    "UnsupportedError",
]
