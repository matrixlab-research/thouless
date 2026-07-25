"""Lattice visualization from geometry rather than model-specific fixtures."""

from __future__ import annotations

import itertools

import numpy as np

from .utils import project, require_matplotlib, require_plotly


def _cell_translations(lattice, n_cells):
    radius = int(n_cells)
    if radius < 1:
        raise ValueError("n_cells must be a positive integer.")
    periodic = lattice.periodic_dirs
    if not periodic:
        yield np.zeros(lattice.dim_r)
        return
    for offsets in itertools.product(
        range(-(radius - 1), radius),
        repeat=len(periodic),
    ):
        reduced = np.zeros(lattice.dim_r)
        reduced[np.asarray(periodic)] = offsets
        yield reduced


def plot_lattice(
    lattice,
    n_cells=1,
    proj_plane=None,
    orb_color="r",
    fig=None,
    ax=None,
):
    """Draw primitive vectors and orbitals in a Cartesian projection."""
    plt = require_matplotlib()
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    points = [np.zeros(2)]
    origin = np.zeros(2)
    ax.scatter(*origin, marker="x", color="black", zorder=4)

    for axis in lattice.periodic_dirs:
        endpoint = project(lattice.lat_vecs[axis], proj_plane)
        ax.annotate(
            "",
            xy=endpoint,
            xytext=origin,
            arrowprops={"arrowstyle": "->", "color": "tab:blue"},
        )
        ax.text(*endpoint, rf"$a_{{{axis}}}$", color="tab:blue")
        points.append(endpoint)

    orbital_cartesian = lattice.get_orb_vecs(cartesian=True)
    first = True
    for translation in _cell_translations(lattice, n_cells):
        shift = translation @ lattice.lat_vecs
        for position in orbital_cartesian + shift:
            location = project(position, proj_plane)
            points.append(location)
            ax.scatter(
                *location,
                color=orb_color,
                s=24,
                zorder=3,
                label="orbitals" if first else None,
            )
            first = False

    coordinates = np.asarray(points)
    if coordinates.size:
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0)
        padding = np.maximum(0.1 * (upper - lower), 0.1)
        ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
        ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax


def plot_lattice_3d(
    lattice,
    n_cells=1,
    show_lattice_info=True,
    site_colors=None,
    site_names=None,
):
    """Build a Plotly figure for a three-dimensional lattice."""
    if lattice.dim_r != 3:
        raise ValueError("Lattice must be 3D to use this function.")
    go = require_plotly()
    traces = []
    for axis in lattice.periodic_dirs:
        endpoint = lattice.lat_vecs[axis]
        traces.append(
            go.Scatter3d(
                x=[0.0, endpoint[0]],
                y=[0.0, endpoint[1]],
                z=[0.0, endpoint[2]],
                mode="lines+text",
                text=["", f"a{axis}"],
                line={"color": "blue", "width": 5},
                name=f"a{axis}",
            )
        )
    positions = []
    names = []
    colors = []
    palette = site_colors or ["red"] * lattice.norb
    labels = site_names or [f"Orbital {index}" for index in range(lattice.norb)]
    if len(palette) != lattice.norb or len(labels) != lattice.norb:
        raise ValueError("site_colors and site_names must match the orbital count.")
    for translation in _cell_translations(lattice, n_cells):
        shift = translation @ lattice.lat_vecs
        for orbital, position in enumerate(
            lattice.get_orb_vecs(cartesian=True) + shift
        ):
            positions.append(position)
            names.append(labels[orbital])
            colors.append(palette[orbital])
    positions = np.asarray(positions)
    traces.append(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker={"size": 6, "color": colors},
            text=names,
            name="orbitals",
        )
    )
    figure = go.Figure(data=traces)
    figure.update_layout(scene={"aspectmode": "data"})
    if show_lattice_info:
        figure.update_layout(
            title=f"{lattice.dim_r}D lattice with {lattice.norb} orbitals"
        )
    return figure
