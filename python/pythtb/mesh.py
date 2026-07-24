"""Sampling-mesh compatibility for PythTB 2.0."""

from __future__ import annotations

import numpy as np


class _Axis:
    def __init__(self, axis_type, name):
        self.type = axis_type
        self.name = name
        self.size = 0
        self.loop_components: list[int] = []
        self.endpoint_components: list[int] = []
        self.winds_bz_components: list[int] = []

    @property
    def is_loop(self):
        return bool(self.loop_components)

    @property
    def has_endpoint(self):
        return bool(self.endpoint_components)


class Mesh:
    """A regular mesh in reciprocal and parameter space."""

    def __init__(self, axis_types, axis_names=None, dim_k=None):
        if any(kind not in ("k", "l") for kind in axis_types):
            raise ValueError("Axis types must be either 'k' or 'l'.")
        if axis_names is None:
            counts = {"k": 0, "l": 0}
            axis_names = []
            for kind in axis_types:
                axis_names.append(f"{kind}_{counts[kind]}")
                counts[kind] += 1
        if len(axis_names) != len(axis_types):
            raise ValueError("Axis types and axis names must have the same length.")
        self._axes = [
            _Axis(kind, name) for kind, name in zip(axis_types, axis_names, strict=True)
        ]
        self._dim_k = (
            sum(kind == "k" for kind in axis_types) if dim_k is None else int(dim_k)
        )
        if self.nk_axes > self._dim_k:
            raise ValueError("Number of k axes cannot exceed specified dimension.")
        self._flat = np.empty((0, self.dim_total))
        self._points = None
        self._k_vectors = []
        self._lambda_vectors = []

    @property
    def axes(self):
        return self._axes

    @property
    def axis_names(self):
        return [axis.name for axis in self.axes]

    @property
    def axis_types(self):
        return [axis.type for axis in self.axes]

    @property
    def naxes(self):
        return len(self.axes)

    @property
    def nk_axes(self):
        return sum(axis.type == "k" for axis in self.axes)

    @property
    def nl_axes(self):
        return sum(axis.type == "l" for axis in self.axes)

    @property
    def dim_k(self):
        return self._dim_k

    @property
    def dim_lambda(self):
        return self.nl_axes

    @property
    def dim_total(self):
        return self.dim_k + self.dim_lambda

    @property
    def shape_axes(self):
        return tuple(axis.size for axis in self.axes)

    @property
    def shape(self):
        return self.shape_axes + (self.dim_total,)

    @property
    def flat(self):
        return self._flat

    @property
    def points(self):
        return self._points

    @property
    def filled(self):
        return self._points is not None

    @property
    def is_grid(self):
        return self.naxes == self.dim_total

    @property
    def is_k_torus(self):
        k_axes = [axis for axis in self.axes if axis.type == "k"]
        return (
            self.is_grid
            and self.dim_k > 0
            and len(k_axes) == self.dim_k
            and all(axis.winds_bz_components for axis in k_axes)
        )

    def is_axis_looped(self, axis_idx):
        return self.axes[axis_idx].is_loop

    def is_axis_closed(self, axis_idx):
        return self.axes[axis_idx].has_endpoint

    def loop(self, axis_idx, component_idx):
        axis = self.axes[axis_idx]
        if component_idx not in axis.loop_components:
            axis.loop_components.append(component_idx)

    @staticmethod
    def _broadcast(value, count, default):
        if value is None:
            return [default] * count
        if isinstance(value, (bool, int, float)):
            return [value] * count
        result = list(value)
        if len(result) != count:
            raise ValueError("mesh option has the wrong number of entries")
        return result

    def build_grid(
        self,
        shape,
        gamma_centered=False,
        k_endpoints=False,
        lambda_start=0.0,
        lambda_stop=1.0,
        lambda_endpoints=True,
    ):
        shape = tuple(int(size) for size in shape)
        if len(shape) != self.naxes or any(size < 1 for size in shape):
            raise ValueError("shape must provide one positive size per mesh axis")
        gamma = self._broadcast(gamma_centered, self.nk_axes, False)
        k_ends = self._broadcast(k_endpoints, self.nk_axes, False)
        lam_start = self._broadcast(lambda_start, self.nl_axes, 0.0)
        lam_stop = self._broadcast(lambda_stop, self.nl_axes, 1.0)
        lam_ends = self._broadcast(lambda_endpoints, self.nl_axes, True)

        k_index = 0
        lambda_index = 0
        axis_values = []
        self._k_vectors = []
        self._lambda_vectors = []
        for axis, size in zip(self.axes, shape, strict=True):
            axis.size = size
            if axis.type == "k":
                start = -0.5 if gamma[k_index] else 0.0
                stop = 0.5 if gamma[k_index] else 1.0
                values = np.linspace(start, stop, size, endpoint=bool(k_ends[k_index]))
                component = k_index
                axis.loop_components = [component]
                axis.winds_bz_components = [component]
                if k_ends[k_index]:
                    axis.endpoint_components = [component]
                self._k_vectors.append(values)
                k_index += 1
            else:
                values = np.linspace(
                    lam_start[lambda_index],
                    lam_stop[lambda_index],
                    size,
                    endpoint=bool(lam_ends[lambda_index]),
                )
                component = self.dim_k + lambda_index
                if lam_ends[lambda_index]:
                    axis.endpoint_components = [component]
                self._lambda_vectors.append(values)
                lambda_index += 1
            axis_values.append(values)

        grids = np.meshgrid(*axis_values, indexing="ij")
        points = np.zeros(shape + (self.dim_total,), dtype=float)
        k_index = 0
        lambda_index = 0
        for axis, grid in zip(self.axes, grids, strict=True):
            if axis.type == "k":
                points[..., k_index] = grid
                k_index += 1
            else:
                points[..., self.dim_k + lambda_index] = grid
                lambda_index += 1
        self._points = points
        self._flat = points.reshape(-1, self.dim_total)
        return self

    def build_custom(self, points):
        points = np.asarray(points, dtype=float)
        if self.naxes != 1:
            raise ValueError("custom paths require exactly one mesh axis")
        if points.ndim != 2 or points.shape[1] != self.dim_total:
            raise ValueError("custom points must have shape (N, dim_k + dim_lambda)")
        if points.shape[0] < 1:
            raise ValueError("custom paths must contain at least one point")
        self.axes[0].size = points.shape[0]
        self._points = points.copy()
        self._flat = points.copy()
        self._k_vectors = [points[:, index] for index in range(self.dim_k)]
        self._lambda_vectors = [
            points[:, self.dim_k + index] for index in range(self.dim_lambda)
        ]
        return self

    def get_k_points(self):
        if not self._k_vectors:
            return np.empty((0, self.dim_k))
        grids = np.meshgrid(*self._k_vectors, indexing="ij")
        return np.stack(grids, axis=-1)

    def get_param_points(self):
        if not self._lambda_vectors:
            return np.empty((0, self.dim_lambda))
        grids = np.meshgrid(*self._lambda_vectors, indexing="ij")
        return np.stack(grids, axis=-1)


__all__ = ["Mesh"]
