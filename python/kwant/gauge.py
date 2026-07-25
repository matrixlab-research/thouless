"""Magnetic-gauge geometry backed by the Thouless Rust core."""

from __future__ import annotations

import numpy as np

from thouless import _core


def _field_samples(field, points, weight_dimension):
    if callable(field):
        values = [field(point) for point in points]
    else:
        values = [field] * len(points)
    samples = []
    for value in values:
        value = np.asarray(value, dtype=float)
        if weight_dimension == 1 and value.ndim == 0:
            samples.append([float(value)])
        elif value.shape == (weight_dimension,):
            samples.append(value.tolist())
        else:
            raise ValueError("magnetic field has an incompatible dimension")
    return samples


def _surface_integral(field, loop, tol=1e-8, average=False):
    """Integrate a scalar 2D or vector 3D field through a polygon."""
    del average
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("surface-integration tolerance must be positive")
    loop = np.asarray(loop, dtype=float)
    points, oriented_weights = _core.gauge_surface_quadrature(loop.tolist())
    points = [np.asarray(point, dtype=float) for point in points]
    weight_dimension = len(oriented_weights[0])
    samples = _field_samples(field, points, weight_dimension)
    return _core.gauge_surface_integral(oriented_weights, samples)


def _loops_in_finite(system):
    """Return a minimum cycle basis of a finalized finite-system graph."""
    edges = {
        tuple(sorted((int(first), int(second))))
        for first, second in system.graph
        if int(first) != int(second)
    }
    return _core.gauge_minimum_cycle_basis(
        int(system.graph.num_nodes),
        sorted(edges),
    )


def _undirected_edges(graph, node_limit=None):
    edges = {
        tuple(sorted((int(first), int(second))))
        for first, second in graph
        if int(first) != int(second)
        and (
            node_limit is None
            or (int(first) < node_limit and int(second) < node_limit)
        )
    }
    return sorted(edges)


def _validate_lead_unit_cell(lead):
    cell_size = int(lead.cell_size)
    if not _core.gauge_graph_is_connected(
        cell_size,
        _undirected_edges(lead.graph, cell_size),
    ):
        raise ValueError("lead unit cell not connected")


def _uniform_field(field, dimension):
    if callable(field):
        return None
    field = np.asarray(field, dtype=float)
    if dimension in (1, 2) and field.ndim == 0:
        return [float(field)]
    if dimension == 3 and field.shape == (3,):
        return field.tolist()
    raise ValueError("magnetic field has an incompatible dimension")


def _peierls_phase(field, axis=None):
    cache = {}

    def phase(first, second):
        try:
            return cache[first, second]
        except KeyError:
            pass
        first_position = np.asarray(first.pos, dtype=float)
        second_position = np.asarray(second.pos, dtype=float)
        if first_position.shape != second_position.shape:
            raise ValueError("hopping endpoints have incompatible positions")
        dimension = len(first_position)
        if dimension == 1:
            value = 1 + 0j
            cache[first, second] = value
            cache[second, first] = value
            return value
        uniform = _uniform_field(field, dimension)
        if uniform is not None and axis is not None and dimension == 2:
            value = _core.gauge_uniform_axial_field_phase(
                first_position.tolist(),
                second_position.tolist(),
                axis.tolist(),
                uniform[0],
            )
        elif uniform is not None:
            value = _core.gauge_uniform_field_phase(
                first_position.tolist(),
                second_position.tolist(),
                uniform,
            )
        else:
            if axis is not None and dimension == 2:
                points, oriented_weights = _core.gauge_axial_line_quadrature(
                    first_position.tolist(),
                    second_position.tolist(),
                    axis.tolist(),
                )
            else:
                points, oriented_weights = _core.gauge_line_quadrature(
                    first_position.tolist(),
                    second_position.tolist(),
                )
            points = [np.asarray(point, dtype=float) for point in points]
            samples = _field_samples(field, points, len(oriented_weights[0]))
            flux = _core.gauge_surface_integral(oriented_weights, samples)
            value = _core.gauge_phase_from_flux(flux)
        cache[first, second] = value
        cache[second, first] = np.conj(value)
        return value

    return phase


def _common_translation_axis(system):
    leads = tuple(getattr(system, "leads", ()))
    if leads:
        periods = [
            np.asarray(lead.symmetry.periods[0], dtype=float)
            for lead in leads
            if getattr(lead, "symmetry", None) is not None
        ]
    elif getattr(system, "symmetry", None) is not None:
        symmetry_periods = getattr(system.symmetry, "periods", ())
        periods = (
            [np.asarray(symmetry_periods[0], dtype=float)]
            if len(symmetry_periods) == 1
            else []
        )
    else:
        periods = []
    if not periods or any(period.shape != (2,) for period in periods):
        return None
    axis = periods[0] / np.linalg.norm(periods[0])
    if any(abs(np.dot(axis, period / np.linalg.norm(period))) < 1 - 1e-10
           for period in periods[1:]):
        return None
    return axis


class magnetic_gauge:
    """Construct globally consistent Peierls phases for a finalized system."""

    def __init__(self, system):
        from .builder import FiniteSystem, InfiniteSystem

        if not isinstance(system, (FiniteSystem, InfiniteSystem)):
            raise TypeError("Expected a finalized Builder")
        self.system = system
        leads = tuple(getattr(system, "leads", ()))
        if hasattr(system, "cell_size"):
            _validate_lead_unit_cell(system)
        for lead in leads:
            if hasattr(lead, "cell_size"):
                _validate_lead_unit_cell(lead)
        interfaces = [
            np.asarray(interface, dtype=int).tolist()
            for interface in getattr(system, "lead_interfaces", ())
        ]
        if interfaces and not _core.gauge_interfaces_are_acyclic(
            int(system.graph.num_nodes),
            interfaces,
        ):
            raise ValueError("lead interfaces overconstrain the magnetic gauge")
        self._field_count = 1 + len(leads)
        self._axis = _common_translation_axis(system)

    def __call__(self, *fields, tol=1e-8, average=False):
        del average
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("gauge tolerance must be positive")
        if len(fields) != self._field_count:
            raise TypeError(
                f"expected {self._field_count} magnetic field argument(s), "
                f"got {len(fields)}"
            )
        phases = tuple(_peierls_phase(field, self._axis) for field in fields)
        return phases[0] if len(phases) == 1 else phases


__all__ = ["_loops_in_finite", "_surface_integral", "magnetic_gauge"]
