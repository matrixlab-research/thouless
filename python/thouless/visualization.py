"""Regular fields derived from local densities and bond currents."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from . import _core
from ._binding import call


@dataclass(frozen=True)
class RegularField:
    values: np.ndarray
    shape: tuple[int, ...]
    components: int
    bounds: tuple[tuple[float, float], ...]


def _field(result: tuple[object, object, object, object]) -> RegularField:
    values, shape, components, bounds = result
    return RegularField(
        np.asarray(values, dtype=np.float64),
        tuple(int(value) for value in shape),
        int(components),
        tuple((float(low), float(high)) for low, high in bounds),
    )


def interpolate_density(
    points: npt.ArrayLike,
    values: npt.ArrayLike,
    reference_edges: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
    *,
    absolute_width: float | None = None,
    relative_width: float | None = None,
    samples_per_width: int = 9,
) -> RegularField:
    return _field(
        call(
            _core.interpolate_density_field,
            np.asarray(points, dtype=np.float64).tolist(),
            np.asarray(values, dtype=np.float64).tolist(),
            [
                (
                    np.asarray(first, dtype=np.float64).tolist(),
                    np.asarray(second, dtype=np.float64).tolist(),
                )
                for first, second in reference_edges
            ],
            absolute_width,
            relative_width,
            int(samples_per_width),
        )
    )


def interpolate_current(
    edges: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
    currents: npt.ArrayLike,
    *,
    absolute_width: float | None = None,
    relative_width: float | None = None,
    samples_per_width: int = 9,
) -> RegularField:
    return _field(
        call(
            _core.interpolate_current_field,
            [
                (
                    np.asarray(first, dtype=np.float64).tolist(),
                    np.asarray(second, dtype=np.float64).tolist(),
                )
                for first, second in edges
            ],
            np.asarray(currents, dtype=np.float64).tolist(),
            absolute_width,
            relative_width,
            int(samples_per_width),
        )
    )


__all__ = ["RegularField", "interpolate_current", "interpolate_density"]
