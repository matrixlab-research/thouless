"""Private data conversion at the Python/Rust boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

from .errors import InternalError, InvalidInputError, NumericalError, ShapeError

T = TypeVar("T")
ArrayLike = npt.ArrayLike


def call(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Call the private extension and expose stable public error classes."""
    try:
        return function(*args, **kwargs)
    except IndexError as error:
        raise ShapeError(str(error)) from error
    except ValueError as error:
        raise InvalidInputError(str(error)) from error
    except RuntimeError as error:
        raise NumericalError(str(error)) from error
    except (MemoryError, NotImplementedError):
        raise
    except Exception as error:  # pragma: no cover - defensive ABI boundary
        raise InternalError(str(error)) from error


def real_vector(value: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ShapeError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise InvalidInputError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(array)


def real_matrix(value: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ShapeError(f"{name} must be two-dimensional")
    if not np.all(np.isfinite(array)):
        raise InvalidInputError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(array)


def complex_vector(value: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != 1:
        raise ShapeError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise InvalidInputError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(array)


def complex_matrix(value: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != 2:
        raise ShapeError(f"{name} must be two-dimensional")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise InvalidInputError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(array)


def complex_grid(value: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != 3:
        raise ShapeError(f"{name} must have shape (samples, rows, columns)")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise InvalidInputError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(array)


def integer_vector(value: Sequence[int], *, name: str) -> list[int]:
    try:
        result = [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ShapeError(f"{name} must be an integer sequence") from error
    return result
