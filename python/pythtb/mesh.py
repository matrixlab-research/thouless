"""Sampling-mesh compatibility for PythTB 2.0."""

from __future__ import annotations

import numpy as np


class Axis:
    """One sampling axis and its loop, endpoint, and BZ-winding metadata."""

    def __init__(self, axis_type, name=None):
        if axis_type not in ("k", "l"):
            raise TypeError("Axis type must be either 'k' or 'l'.")
        self._type = axis_type
        self._name = f"{axis_type}_axis" if name is None else name
        self._size = 0
        self._loop_components: list[int] = []
        self._endpoint_components: list[int] = []
        self._winds_bz_components: list[int] = []

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError("Axis name must be a string.")
        self._name = value

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        if not isinstance(value, (int, np.integer)):
            raise TypeError("Axis size must be an integer.")
        if int(value) < 0:
            raise ValueError("Axis size must be non-negative.")
        self._size = int(value)

    @property
    def is_loop(self):
        return bool(self.loop_components)

    @property
    def loop_components(self):
        return self._loop_components

    @loop_components.setter
    def loop_components(self, value):
        self._loop_components = [int(component) for component in value]

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
    def endpoint_components(self):
        return self._endpoint_components

    @endpoint_components.setter
    def endpoint_components(self, value):
        self._endpoint_components = [int(component) for component in value]

    @property
    def winds_bz(self):
        return bool(self.winds_bz_components)

    @property
    def winds_bz_components(self):
        return self._winds_bz_components

    @winds_bz_components.setter
    def winds_bz_components(self, value):
        self._winds_bz_components = [int(component) for component in value]

    def add_loop_component(self, comp_idx):
        if comp_idx not in self.loop_components:
            self.loop_components.append(comp_idx)

    def remove_loop_component(self, comp_idx):
        if comp_idx in self.loop_components:
            self.loop_components.remove(comp_idx)

    def add_endpoint_component(self, comp_idx):
        if comp_idx not in self.endpoint_components:
            self.endpoint_components.append(comp_idx)

    def remove_endpoint_component(self, comp_idx):
        if comp_idx in self.endpoint_components:
            self.endpoint_components.remove(comp_idx)

    def add_wind_bz_component(self, comp_idx):
        if comp_idx not in self.winds_bz_components:
            self.winds_bz_components.append(comp_idx)

    def remove_wind_bz_component(self, comp_idx):
        if comp_idx in self.winds_bz_components:
            self.winds_bz_components.remove(comp_idx)

    def __repr__(self):
        return f"Axis(type={self.type}, name={self.name}, size={self.size})"


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
            Axis(kind, name) for kind, name in zip(axis_types, axis_names, strict=True)
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
        self._nodes = None

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
    def component_types(self):
        return tuple(["k"] * self.dim_k + ["l"] * self.dim_lambda)

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
    def nodes(self):
        return None if self._nodes is None else self._nodes.copy()

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
    def loop_axes(self):
        return [axis for axis in self.axes if axis.is_loop]

    @property
    def endpoint_axes(self):
        return [axis for axis in self.axes if axis.has_endpoint]

    @property
    def bz_winding_axes(self):
        return [
            axis for axis in self.axes if axis.is_k_axis and axis.winds_bz
        ]

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
        axis.add_loop_component(component_idx)
        if winds_bz:
            if not axis.is_k_axis or component_idx >= self.dim_k:
                raise ValueError("Brillouin-zone winding requires a k-axis and k-component")
            axis.add_wind_bz_component(component_idx)
        if closed:
            axis.add_endpoint_component(component_idx)

    def unloop(self, axis_idx, component_idx, unwind_bz=False, open=False):
        if not 0 <= axis_idx < self.naxes:
            raise IndexError(f"axis_idx {axis_idx} out of bounds for {self.naxes} axes")
        if not 0 <= component_idx < self.dim_total:
            raise IndexError(
                f"component_idx {component_idx} out of bounds for {self.dim_total} components"
            )
        axis = self.axes[axis_idx]
        axis.remove_loop_component(component_idx)
        if unwind_bz:
            axis.remove_wind_bz_component(component_idx)
        if open:
            axis.remove_endpoint_component(component_idx)

    def info(self, show=True):
        """Return or print mesh dimensions and topology metadata."""
        mesh_type = (
            "uninitialized"
            if not self.filled
            else "grid" if self.is_grid else "path"
        )
        lines = [
            "----------------------------------------",
            "            Mesh report",
            "----------------------------------------",
            f"type              = {mesh_type}",
            f"axis types        = {self.axis_types}",
            f"axis names        = {self.axis_names}",
            f"shape             = {self.shape}",
            f"component types   = {self.component_types}",
            f"k-space torus     = {self.is_k_torus}",
        ]
        for index, axis in enumerate(self.axes):
            lines.append(
                f"axis {index}: loops={axis.loop_components}, "
                f"endpoints={axis.endpoint_components}, "
                f"winds_bz={axis.winds_bz_components}"
            )
        report = "\n".join(lines)
        if show:
            print(report)
            return None
        return report

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
        lambda_endpoints=True,
        lambda_start=0.0,
        lambda_stop=1.0,
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
        self._nodes = None
        return self

    def build_path(self, nodes, n_interp=1):
        """Build a piecewise-linear path through combined k/parameter space."""
        if self.naxes != 1:
            raise ValueError("For a path, the mesh must have exactly one axis.")
        if not isinstance(n_interp, (int, np.integer)) or int(n_interp) < 1:
            raise ValueError("n_interp must be a positive integer.")
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != self.dim_total:
            raise ValueError(
                f"nodes must have shape (N_nodes, {self.dim_total})"
            )
        if len(nodes) < 1 or not np.all(np.isfinite(nodes)):
            raise ValueError("nodes must contain finite path coordinates")
        segments = []
        for start, stop in zip(nodes[:-1], nodes[1:], strict=True):
            fractions = np.linspace(
                0.0,
                1.0,
                int(n_interp),
                endpoint=False,
            )
            segments.append(
                start[np.newaxis, :]
                + fractions[:, np.newaxis]
                * (stop - start)[np.newaxis, :]
            )
        points = (
            np.vstack([*segments, nodes[-1:]])
            if segments
            else nodes.copy()
        )
        self.axes[0].size = len(points)
        self._flat = points.copy()
        self._points = points.copy()
        self._nodes = nodes.copy()
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
        self._nodes = None
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
        if not self.filled:
            raise ValueError("Mesh points are not initialized.")
        selectors = [
            slice(None) if axis.is_k_axis else 0 for axis in self.axes
        ]
        result = np.asarray(
            self.points[tuple(selectors) + (slice(0, self.dim_k),)]
        )
        return result.reshape(self.shape_k + (self.dim_k,))

    def get_param_points(self):
        if not self.filled:
            raise ValueError("Mesh points are not initialized.")
        selectors = [
            0 if axis.is_k_axis else slice(None) for axis in self.axes
        ]
        result = np.asarray(
            self.points[
                tuple(selectors)
                + (slice(self.dim_k, self.dim_total),)
            ]
        )
        return result.reshape(self.shape_lambda + (self.dim_lambda,))

    @staticmethod
    def gen_hyper_cube(
        *n_points,
        start=0.0,
        stop=1.0,
        endpoint=False,
        flat=True,
    ):
        """Generate a regular Cartesian hypercube in arbitrary dimension."""
        if not n_points or any(
            not isinstance(size, (int, np.integer)) or int(size) < 1
            for size in n_points
        ):
            raise ValueError("n_points must contain positive integers")
        dimension = len(n_points)
        starts = Mesh._broadcast(start, dimension, 0.0)
        stops = Mesh._broadcast(stop, dimension, 1.0)
        endpoints = Mesh._broadcast(endpoint, dimension, False)
        axes = [
            np.linspace(
                starts[index],
                stops[index],
                int(size),
                endpoint=bool(endpoints[index]),
            )
            for index, size in enumerate(n_points)
        ]
        cube = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        return cube.reshape(-1, dimension) if flat else cube


__all__ = ["Axis", "Mesh"]
