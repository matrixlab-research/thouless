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
    """Scalar or vector field sampled on a regular Cartesian grid.

    Attributes:
        values: Flattened grid values with component index varying fastest.
        shape: Number of samples along each Cartesian axis.
        components: One for densities or the Cartesian dimension for currents.
        bounds: Inclusive low and high coordinate bounds on every axis.
    """

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
    """Smooth point densities onto a regular Cartesian grid.

    Args:
        points: Site coordinates with shape ``(site_count, dimension)``.
        values: Scalar value at every site.
        reference_edges: Representative bonds used to infer a relative width.
        absolute_width: Explicit positive Gaussian width.
        relative_width: Width as a fraction of the representative bond scale.
        samples_per_width: Grid resolution per Gaussian width.

    Returns:
        Regular scalar field. Exactly one of ``absolute_width`` and
        ``relative_width`` must be supplied.
    """
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
    """Smooth directed bond currents into a divergence-compatible vector field.

    Args:
        edges: Directed pairs of Cartesian endpoint coordinates.
        currents: Signed current assigned to every edge.
        absolute_width: Explicit positive smoothing width.
        relative_width: Width as a fraction of the representative edge scale.
        samples_per_width: Grid resolution per smoothing width.

    Returns:
        Regular vector field with one component per Cartesian axis.
    """
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
