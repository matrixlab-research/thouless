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
    def is_k_axis(self):
        return self.type == "k"

    @property
    def is_lambda_axis(self):
        return self.type == "l"

    @property
    def has_endpoint(self):
        return bool(self.endpoint_components)

    @property
    def winds_bz(self):
        return bool(self.winds_bz_components)


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
    def k_axes(self):
        return [axis for axis in self.axes if axis.is_k_axis]

    @property
    def lambda_axes(self):
        return [axis for axis in self.axes if axis.is_lambda_axis]

    @property
    def k_axis_indices(self):
        return [index for index, axis in enumerate(self.axes) if axis.is_k_axis]

    @property
    def lambda_axis_indices(self):
        return [index for index, axis in enumerate(self.axes) if axis.is_lambda_axis]

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
    def k_component_indices(self):
        return list(range(self.dim_k))

    @property
    def lambda_component_indices(self):
        return list(range(self.dim_k, self.dim_total))

    @property
    def shape_axes(self):
        return tuple(axis.size for axis in self.axes)

    @property
    def shape(self):
        return self.shape_axes + (self.dim_total,)

    @property
    def npoints(self):
        return int(np.prod(self.shape_axes))

    @property
    def shape_k(self):
        return tuple(axis.size for axis in self.k_axes)

    @property
    def shape_lambda(self):
        return tuple(axis.size for axis in self.lambda_axes)

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

    @property
    def loop_mask(self):
        return self._component_mask("loop_components")

    @property
    def endpoint_mask(self):
        return self._component_mask("endpoint_components")

    @property
    def bz_winding_mask(self):
        return self._component_mask("winds_bz_components")

    def _component_mask(self, attribute):
        mask = np.zeros((self.naxes, self.dim_total), dtype=bool)
        for axis_index, axis in enumerate(self.axes):
            for component in getattr(axis, attribute):
                mask[axis_index, component] = True
        return mask

    def _axis_component_query(self, axis_idx, component, attribute):
        if not 0 <= axis_idx < self.naxes:
            raise IndexError(f"axis_idx {axis_idx} out of bounds for {self.naxes} axes")
        if component == "any":
            return bool(getattr(self.axes[axis_idx], attribute))
        if not isinstance(component, int):
            raise TypeError("component must be an integer or 'any'")
        if not -self.dim_total <= component < self.dim_total:
            raise IndexError(
                f"component_idx {component} out of bounds for {self.dim_total} components"
            )
        return component % self.dim_total in getattr(self.axes[axis_idx], attribute)

    def is_axis_looped(self, axis_idx, comp="any"):
        return self._axis_component_query(axis_idx, comp, "loop_components")

    def is_axis_closed(self, axis_idx, comp="any"):
        return self._axis_component_query(axis_idx, comp, "endpoint_components")

    def is_axis_bz_winding(self, axis_idx, comp="any"):
        return self._axis_component_query(axis_idx, comp, "winds_bz_components")

    def loop(self, axis_idx, component_idx, winds_bz=False, closed=False):
        if not 0 <= axis_idx < self.naxes:
            raise IndexError(f"axis_idx {axis_idx} out of bounds for {self.naxes} axes")
        if not 0 <= component_idx < self.dim_total:
            raise IndexError(
                f"component_idx {component_idx} out of bounds for {self.dim_total} components"
            )
        axis = self.axes[axis_idx]
        if component_idx not in axis.loop_components:
            axis.loop_components.append(component_idx)
        if winds_bz:
            if not axis.is_k_axis or component_idx >= self.dim_k:
                raise ValueError("Brillouin-zone winding requires a k-axis and k-component")
            if component_idx not in axis.winds_bz_components:
                axis.winds_bz_components.append(component_idx)
        if closed and component_idx not in axis.endpoint_components:
            axis.endpoint_components.append(component_idx)

    def unloop(self, axis_idx, component_idx, unwind_bz=False, open=False):
        if not 0 <= axis_idx < self.naxes:
            raise IndexError(f"axis_idx {axis_idx} out of bounds for {self.naxes} axes")
        if not 0 <= component_idx < self.dim_total:
            raise IndexError(
                f"component_idx {component_idx} out of bounds for {self.dim_total} components"
            )
        axis = self.axes[axis_idx]
        if component_idx in axis.loop_components:
            axis.loop_components.remove(component_idx)
        if unwind_bz and component_idx in axis.winds_bz_components:
            axis.winds_bz_components.remove(component_idx)
        if open and component_idx in axis.endpoint_components:
            axis.endpoint_components.remove(component_idx)

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
        if points.ndim != self.naxes + 1 or points.shape[-1] != self.dim_total:
            raise ValueError(
                "custom points must have shape (*mesh_shape, dim_k + dim_lambda)"
            )
        if any(size < 1 for size in points.shape[:-1]):
            raise ValueError("custom meshes must contain at least one point per axis")
        for axis, size in zip(self.axes, points.shape[:-1], strict=True):
            axis.size = size
        self._points = points.copy()
        self._flat = points.reshape(-1, self.dim_total)
        self._k_vectors = [points[..., index] for index in range(self.dim_k)]
        self._lambda_vectors = [
            points[..., self.dim_k + index] for index in range(self.dim_lambda)
        ]
        return self

    def get_axis_range(self, axis_index, component_index):
        if not self.filled:
            raise ValueError("Mesh points are not initialized.")
        if not 0 <= axis_index < self.naxes:
            raise IndexError(
                f"axis_index {axis_index} out of bounds for mesh with {self.naxes} axes"
            )
        if not 0 <= component_index < self.dim_total:
            raise IndexError(
                f"component_index {component_index} out of bounds for "
                f"{self.dim_total} components"
            )
        selector = [0] * self.naxes
        selector[axis_index] = slice(None)
        values = self.points[tuple(selector) + (component_index,)]
        return np.asarray(values).reshape(-1)

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
