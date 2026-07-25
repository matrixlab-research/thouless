"""Kwant plotting compatibility over Rust-native field interpolation."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from scipy import interpolate as scipy_interpolate
from scipy import spatial

from thouless import _core

from . import _plotter
from .builder import Builder, FiniteSystem, InfiniteSystem


def set_engine(engine):
    if engine not in _plotter.engines:
        raise RuntimeError(f"plotting engine {engine!r} is not available")
    _plotter.engine = engine


def get_engine():
    return _plotter.engine


def _require_engine():
    if _plotter.engine is None:
        raise RuntimeError("no plotting engine is installed")
    return _plotter.engine


def _finish_matplotlib(fig, file, show):
    if file is not None:
        fig.savefig(file)
    elif show:
        from matplotlib import pyplot

        pyplot.show()
    return fig


def _finish_plotly(fig, file, show):
    if file is not None:
        fig.write_html(file)
    if show:
        fig.show()
    return fig


def _positions(sites, pos_transform=None):
    positions = []
    for site in sites:
        position = np.asarray(site.pos, dtype=float)
        if pos_transform is not None:
            position = np.asarray(pos_transform(position), dtype=float)
        if position.ndim != 1 or len(position) not in (1, 2, 3):
            raise ValueError("site positions must have one, two, or three dimensions")
        positions.append(position)
    if not positions:
        return np.empty((0, 2), dtype=float)
    dimension = len(positions[0])
    if any(len(position) != dimension for position in positions):
        raise ValueError("site positions have inconsistent dimensions")
    return np.asarray(positions, dtype=float)


def _builder_components(system):
    components = [(list(system.sites()), list(system.hoppings()))]
    for lead in system.leads:
        lead_builder = getattr(lead, "builder", lead)
        if isinstance(lead_builder, Builder):
            components.append(
                (list(lead_builder.sites()), list(lead_builder.hoppings()))
            )
    return components


def _finalized_components(system):
    sites = list(system.sites)
    hoppings = [
        (sites[first], sites[second])
        for first, second in system.graph
        if first < second and first < len(sites) and second < len(sites)
    ]
    components = [(sites, hoppings)]
    for lead in getattr(system, "leads", ()):
        lead_sites = list(lead.sites[: lead.cell_size])
        lead_hoppings = [
            (lead.sites[first], lead.sites[second])
            for first, second in lead.graph
            if first < second
            and first < lead.cell_size
            and second < lead.cell_size
        ]
        components.append((lead_sites, lead_hoppings))
    return components


def _components(system):
    if isinstance(system, Builder):
        return _builder_components(system)
    if isinstance(system, (FiniteSystem, InfiniteSystem)):
        return _finalized_components(system)
    raise TypeError("expected a Builder or finalized system")


def _resolve_specification(specification, items, default, argument_count):
    if specification is None:
        return default
    if callable(specification):
        return [specification(*item) if argument_count == 2 else specification(item)
                for item in items]
    if isinstance(specification, str) or np.asarray(specification).ndim == 0:
        return specification
    values = list(specification)
    if len(values) != len(items):
        raise ValueError("plot specification length does not match the system")
    return values


def plot(
    sys,
    num_lead_cells=2,
    unit=None,
    site_symbol="o",
    site_size=0.25,
    site_color=None,
    site_edgecolor=None,
    site_lw=None,
    hop_color=None,
    hop_lw=None,
    lead_site_symbol=None,
    lead_site_size=None,
    lead_color=None,
    colorbar=True,
    file=None,
    show=True,
    dpi=None,
    fig_size=None,
    ax=None,
    pos_transform=None,
    cmap=None,
):
    del num_lead_cells, unit, lead_site_symbol, lead_site_size, lead_color, colorbar
    del site_edgecolor, site_lw, hop_lw
    if isinstance(sys, Builder):
        for name, value in (("site_size", site_size), ("site_symbol", site_symbol)):
            if not isinstance(value, str) and np.asarray(value).ndim > 0:
                raise TypeError(f"{name} arrays require a finalized system")
    components = _components(sys)
    transformed = [
        (sites, hoppings, _positions(sites, pos_transform))
        for sites, hoppings in components
    ]
    dimensions = {positions.shape[1] for _, _, positions in transformed if len(positions)}
    dimension = dimensions.pop() if dimensions else 2
    if dimensions:
        raise ValueError("all plotted components must have the same dimension")
    engine = _require_engine()

    if engine == "matplotlib":
        from matplotlib import pyplot
        from matplotlib.collections import LineCollection
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        if ax is None:
            fig = pyplot.figure(figsize=fig_size, dpi=dpi)
            ax = fig.add_subplot(111, projection="3d" if dimension == 3 else None)
        else:
            fig = ax.figure
        for sites, hoppings, positions in transformed:
            index = {site: position for site, position in zip(sites, positions, strict=True)}
            colors = _resolve_specification(site_color, sites, "k", 1)
            if len(positions):
                if dimension == 3:
                    ax.scatter(*positions.T, c=colors, s=np.asarray(site_size) * 20, cmap=cmap)
                else:
                    ax.scatter(positions[:, 0], positions[:, 1], c=colors,
                               s=np.asarray(site_size) * 20, cmap=cmap,
                               marker=site_symbol if isinstance(site_symbol, str) else "o")
            segments = [
                np.asarray([index[first], index[second]])
                for first, second in hoppings
                if first in index and second in index
            ]
            hop_colors = _resolve_specification(hop_color, hoppings, "k", 2)
            scalar_colors = (
                isinstance(hop_colors, list)
                and hop_colors
                and all(np.asarray(color).ndim == 0 and not isinstance(color, str)
                        for color in hop_colors)
            )
            collection_type = Line3DCollection if dimension == 3 else LineCollection
            if scalar_colors:
                collection = collection_type(segments, cmap=cmap)
                collection.set_array(np.asarray(hop_colors, dtype=float))
            else:
                collection = collection_type(segments, colors=hop_colors)
            ax.add_collection(collection)
        return _finish_matplotlib(fig, file, show)

    from plotly import graph_objects

    fig = graph_objects.Figure()
    for sites, hoppings, positions in transformed:
        index = {site: position for site, position in zip(sites, positions, strict=True)}
        colors = _resolve_specification(site_color, sites, "black", 1)
        segments = [
            np.asarray([index[first], index[second]])
            for first, second in hoppings
            if first in index and second in index
        ]
        if dimension == 3:
            if len(positions):
                fig.add_trace(graph_objects.Scatter3d(
                    x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                    mode="markers", marker={"color": colors},
                ))
            for segment in segments:
                fig.add_trace(graph_objects.Scatter3d(
                    x=segment[:, 0], y=segment[:, 1], z=segment[:, 2],
                    mode="lines", showlegend=False,
                ))
        else:
            if len(positions):
                fig.add_trace(graph_objects.Scatter(
                    x=positions[:, 0], y=positions[:, 1], mode="markers",
                    marker={"color": colors},
                ))
            for segment in segments:
                fig.add_trace(graph_objects.Scatter(
                    x=segment[:, 0], y=segment[:, 1], mode="lines",
                    showlegend=False,
                ))
    return _finish_plotly(fig, file, show)


def mask_interpolate(coords, values, a=None, method="nearest", oversampling=3):
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values)
    if coords.ndim != 2 or coords.shape[1] != 2 or not len(coords):
        raise ValueError("coordinates must be a nonempty two-dimensional point array")
    if len(coords) != len(values):
        raise ValueError("the number of coordinates and values must agree")
    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0)
    tree = spatial.cKDTree(coords)
    sample = coords[: min(10, len(coords))]
    distances = tree.query(sample, min(2, len(coords)))[0]
    minimum_distance = (
        np.min(distances[:, 1])
        if len(coords) > 1
        else np.inf
    )
    diameter = np.linalg.norm(maximum - minimum)
    if minimum_distance < 1e-6 * diameter:
        warnings.warn(
            "Some sites have nearly coinciding positions, interpolation may be confusing.",
            RuntimeWarning,
            stacklevel=2,
        )
    if a is None:
        a = minimum_distance
    a = float(a)
    if not np.isfinite(a) or a <= 0 or a < 1e-6 * diameter:
        raise ValueError("the reference distance a is too small")
    if method not in {"nearest", "linear", "cubic"}:
        raise ValueError("unknown interpolation method")
    shape = np.rint(((maximum - minimum) / a + 1) * oversampling).astype(int)
    shape = np.maximum(shape, 1)
    delta = 0.5 * (oversampling - 1) * a / oversampling
    minimum = minimum - delta
    maximum = maximum + delta
    slices = tuple(
        slice(minimum[axis], maximum[axis], 1j * shape[axis])
        for axis in range(2)
    )
    grid = tuple(np.ogrid[slices])
    image = scipy_interpolate.griddata(coords, values, grid, method=method)
    image = np.asarray(image, dtype=float)
    mask_points = np.mgrid[slices].reshape(2, -1).T
    mask = tree.query(mask_points, eps=0.4)[0] > 0.99 * a
    masked = np.ma.masked_array(image, mask.reshape(image.shape))
    result = masked if get_engine() == "matplotlib" else masked.filled(np.nan)
    return result, image, minimum, maximum


def _site_data(system, values):
    if isinstance(system, Builder):
        sites = list(system.sites())
        if not callable(values):
            raise ValueError("Builder values must be supplied as a callable")
        data = [values(site) for site in sites]
        return sites, np.asarray(data)
    if isinstance(system, FiniteSystem):
        sites = list(system.sites)
        data = [values(site) for site in sites] if callable(values) else values
        data = np.asarray(data)
        if len(data) != len(sites):
            raise ValueError("the number of values must match the number of sites")
        return sites, data
    raise TypeError("expected a Builder or finalized finite system")


def map(
    sys,
    value,
    colorbar=True,
    cmap=None,
    vmin=None,
    vmax=None,
    a=None,
    method="nearest",
    oversampling=3,
    num_lead_cells=0,
    file=None,
    show=True,
    dpi=None,
    fig_size=None,
    ax=None,
    pos_transform=None,
):
    del colorbar, num_lead_cells
    sites, values = _site_data(sys, value)
    positions = _positions(sites, pos_transform)
    if positions.shape[1] != 2:
        raise ValueError("map requires two-dimensional positions")
    image, _, minimum, maximum = mask_interpolate(
        positions, values, a=a, method=method, oversampling=oversampling
    )
    engine = _require_engine()
    if engine == "matplotlib":
        from matplotlib import pyplot

        if ax is None:
            fig, ax = pyplot.subplots(figsize=fig_size, dpi=dpi)
        else:
            fig = ax.figure
        ax.imshow(
            np.asarray(image).T,
            origin="lower",
            extent=(minimum[0], maximum[0], minimum[1], maximum[1]),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        return _finish_matplotlib(fig, file, show)
    from plotly import graph_objects

    fig = graph_objects.Figure(data=graph_objects.Heatmap(z=np.asarray(image).T))
    return _finish_plotly(fig, file, show)


def _spectrum_data(system, x, y, params, mask):
    if isinstance(system, FiniteSystem):
        hamiltonian = lambda **kwargs: system.hamiltonian_submatrix(params=kwargs)
    elif callable(system):
        hamiltonian = system
    else:
        raise TypeError("expected a finite system or Hamiltonian callable")
    axes = (x,) if y is None else (x, y)
    names = tuple(axis[0] for axis in axes)
    coordinates = tuple(np.asarray(axis[1]) for axis in axes)
    spectra = []
    for point in itertools.product(*coordinates):
        varying = dict(zip(names, point, strict=True))
        if mask is not None and mask(**varying):
            spectra.append(None)
            continue
        arguments = dict(params or {})
        arguments.update(varying)
        spectra.append(np.linalg.eigvalsh(np.atleast_2d(hamiltonian(**arguments))))
    first = next((spectrum for spectrum in spectra if spectrum is not None), None)
    if first is None:
        raise ValueError("the spectrum mask excludes every parameter point")
    missing = np.full(first.shape, np.nan)
    values = np.asarray([missing if spectrum is None else spectrum for spectrum in spectra])
    return values.reshape(*(len(axis) for axis in coordinates), -1), coordinates, names


def spectrum(
    syst,
    x,
    y=None,
    params=None,
    mask=None,
    file=None,
    show=True,
    dpi=None,
    fig_size=None,
    ax=None,
):
    values, coordinates, names = _spectrum_data(syst, x, y, params, mask)
    engine = _require_engine()
    if engine == "matplotlib":
        from matplotlib import pyplot

        if ax is None:
            fig = pyplot.figure(figsize=fig_size, dpi=dpi)
            ax = fig.add_subplot(111, projection="3d" if y is not None else None)
        else:
            fig = ax.figure
        if y is None:
            for band in values.T:
                ax.plot(coordinates[0], band)
            ax.set_xlabel(names[0])
        else:
            if not hasattr(ax, "plot_surface"):
                raise TypeError("two-parameter spectra require three-dimensional axes")
            grid_x, grid_y = np.meshgrid(*coordinates, indexing="ij")
            for band in np.moveaxis(values, -1, 0):
                ax.plot_surface(grid_x, grid_y, band)
            ax.set_xlabel(names[0])
            ax.set_ylabel(names[1])
        return _finish_matplotlib(fig, file, show)

    if any(argument is not None for argument in (dpi, fig_size, ax)):
        raise RuntimeError("dpi, fig_size, and ax are unavailable with Plotly")
    from plotly import graph_objects

    fig = graph_objects.Figure()
    if y is None:
        for band in values.T:
            fig.add_trace(graph_objects.Scatter(x=coordinates[0], y=band))
    else:
        for band in np.moveaxis(values, -1, 0):
            fig.add_trace(graph_objects.Surface(
                x=coordinates[0], y=coordinates[1], z=band.T
            ))
    return _finish_plotly(fig, file, show)


def bands(
    sys,
    args=(),
    momenta=65,
    file=None,
    show=True,
    dpi=None,
    fig_size=None,
    ax=None,
    *,
    params=None,
):
    from .physics import Bands

    if not isinstance(sys, InfiniteSystem):
        raise TypeError("bands requires a finalized infinite system")
    momenta = np.asarray(momenta)
    if momenta.ndim != 1:
        momenta = np.linspace(-np.pi, np.pi, int(momenta))
    evaluator = Bands(sys, args=args, params=params)
    energies = np.asarray([evaluator(momentum) for momentum in momenta])
    return spectrum(
        lambda k: np.diag(energies[np.argmin(abs(momenta - k))]),
        ("k", momenta),
        file=file,
        show=show,
        dpi=dpi,
        fig_size=fig_size,
        ax=ax,
    )


def _reshape_field(output):
    values, shape, components, bounds = output
    array = np.asarray(values, dtype=float).reshape(
        *shape, *(() if components == 1 else (components,))
    )
    if components == 1:
        array = array[..., np.newaxis]
    return array, tuple(tuple(bound) for bound in bounds)


def _mask_field(field, bounds, points, cutoff):
    slices = tuple(
        slice(minimum, maximum, 1j * size)
        for (minimum, maximum), size in zip(bounds, field.shape, strict=False)
    )
    coordinates = np.mgrid[slices].reshape(len(bounds), -1).T
    mask = spatial.cKDTree(points).query(
        coordinates, distance_upper_bound=cutoff
    )[0] == np.inf
    return np.ma.masked_array(field, mask.reshape(field.shape[: len(bounds)]))


def interpolate_density(syst, density, relwidth=None, abswidth=None, n=9, mask=True):
    if not isinstance(syst, FiniteSystem):
        raise TypeError("the system needs to be finalized")
    values = np.asarray(density, dtype=float)
    if values.ndim != 1 or len(values) != len(syst.sites):
        raise ValueError("density and sites arrays must have the same length")
    points = [np.asarray(site.pos, dtype=float).tolist() for site in syst.sites]
    edges = [
        (points[first], points[second])
        for first, second in syst.graph
        if first < second
    ]
    field, bounds = _reshape_field(
        _core.interpolate_density_field(
            points,
            values.tolist(),
            edges,
            absolute_width=abswidth,
            relative_width=relwidth,
            samples_per_width=int(n),
        )
    )
    if mask:
        width = 2 * (np.min(np.asarray(points), axis=0)[0] - bounds[0][0])
        field = _mask_field(field, bounds, np.asarray(points), 0.6 * width)
    return field, bounds


def interpolate_current(syst, current, relwidth=None, abswidth=None, n=9):
    if not isinstance(syst, FiniteSystem):
        raise TypeError("the system needs to be finalized")
    current = np.asarray(current, dtype=float)
    if current.ndim != 1 or len(current) != syst.graph.num_edges:
        raise ValueError("current and hopping arrays must have the same length")
    unique = {}
    edges = []
    currents = []
    for index, (first, second) in enumerate(syst.graph):
        if (second, first) in unique:
            currents[unique[second, first]] -= current[index]
        else:
            unique[first, second] = len(edges)
            edges.append((
                np.asarray(syst.sites[second].pos, dtype=float).tolist(),
                np.asarray(syst.sites[first].pos, dtype=float).tolist(),
            ))
            currents.append(float(current[index]))
    currents = (np.asarray(currents) / 2).tolist()
    return _reshape_field(
        _core.interpolate_current_field(
            edges,
            currents,
            absolute_width=abswidth,
            relative_width=relwidth,
            samples_per_width=int(n),
        )
    )


def scalarplot(
    field,
    box,
    cmap=None,
    colorbar=True,
    file=None,
    show=True,
    dpi=None,
    fig_size=None,
    ax=None,
    **kwargs,
):
    del colorbar, kwargs
    engine = _require_engine()
    data = np.asarray(field).squeeze()
    if engine == "matplotlib":
        from matplotlib import pyplot

        if ax is None:
            fig, ax = pyplot.subplots(figsize=fig_size, dpi=dpi)
        else:
            fig = ax.figure
        ax.imshow(data.T, origin="lower", cmap=cmap,
                  extent=(box[0][0], box[0][1], box[1][0], box[1][1]))
        return _finish_matplotlib(fig, file, show)
    from plotly import graph_objects

    fig = graph_objects.Figure(data=graph_objects.Heatmap(z=data.T))
    return _finish_plotly(fig, file, show)


def streamplot(
    field,
    box,
    cmap=None,
    bgcolor=None,
    linecolor="k",
    file=None,
    show=True,
    dpi=None,
    fig_size=None,
    ax=None,
    **kwargs,
):
    del cmap, bgcolor, kwargs
    engine = _require_engine()
    data = np.asarray(field)
    if data.shape[-1] != 2:
        raise ValueError("stream plots require a two-dimensional vector field")
    x = np.linspace(*box[0], data.shape[0])
    y = np.linspace(*box[1], data.shape[1])
    if engine == "matplotlib":
        from matplotlib import pyplot

        if ax is None:
            fig, ax = pyplot.subplots(figsize=fig_size, dpi=dpi)
        else:
            fig = ax.figure
        ax.streamplot(x, y, data[..., 0].T, data[..., 1].T, color=linecolor)
        return _finish_matplotlib(fig, file, show)
    from plotly import graph_objects

    fig = graph_objects.Figure()
    return _finish_plotly(fig, file, show)


def current(syst, current, relwidth=0.05, **kwargs):
    return streamplot(*interpolate_current(syst, current, relwidth=relwidth), **kwargs)


def density(syst, density, relwidth=0.05, **kwargs):
    return scalarplot(*interpolate_density(syst, density, relwidth=relwidth), **kwargs)


__all__ = [
    "bands",
    "current",
    "density",
    "get_engine",
    "interpolate_current",
    "interpolate_density",
    "map",
    "mask_interpolate",
    "plot",
    "scalarplot",
    "set_engine",
    "spectrum",
    "streamplot",
]
