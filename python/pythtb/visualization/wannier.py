"""Backend-independent plots of Wannier localization diagnostics."""

from __future__ import annotations

import itertools

import numpy as np

from .utils import require_matplotlib


def plot_density(
    wan,
    wan_idx,
    mark_home_cell=False,
    mark_center=False,
    show_lattice=False,
    dens_size=40,
    lat_size=2,
    fig=None,
    ax=None,
    show=False,
    cbar=True,
):
    """Plot real-space Wannier probability on orbital sites."""
    plt = require_matplotlib()
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    data = wan._get_sc_weights(wan_idx)
    weights = np.maximum(data["all"]["wt"], np.finfo(float).tiny)
    artist = ax.scatter(
        data["all"]["xs"],
        data["all"]["ys"],
        c=weights,
        s=dens_size,
        cmap="plasma",
    )
    if show_lattice:
        ax.scatter(
            data["all"]["xs"],
            data["all"]["ys"],
            color="black",
            s=lat_size,
        )
    if mark_home_cell:
        ax.scatter(
            data["home"]["xs"],
            data["home"]["ys"],
            facecolors="none",
            edgecolors="blue",
            s=lat_size,
        )
    if mark_center:
        center = np.zeros(2)
        center[: min(2, wan.centers.shape[1])] = wan.centers[wan_idx, :2]
        ax.scatter(*center, marker="x", color="green")
    if cbar:
        fig.colorbar(artist, ax=ax, label=f"|w_{wan_idx}(r)|²")
    ax.set_aspect("equal", adjustable="box")
    if show:
        plt.show()
    return fig, ax


def plot_decay(wan, wan_idx, fig=None, ax=None, show=False):
    """Plot Wannier probability against distance from its center."""
    plt = require_matplotlib()
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    data = wan._get_sc_weights(wan_idx)["all"]
    weights = np.maximum(data["wt"], np.finfo(float).tiny)
    ax.scatter(data["r"], weights, s=10)
    ax.set_yscale("log")
    ax.set_xlabel("|r - r_c|")
    ax.set_ylabel(f"|w_{wan_idx}(r)|²")
    if show:
        plt.show()
    return fig, ax


def plot_centers(
    wan,
    center_scale=15,
    section_home_cell=True,
    color_home_cell=True,
    translate_centers=False,
    show=False,
    legend=True,
    pmx=4,
    pmy=4,
    center_color="r",
    center_marker="*",
    lat_home_color="b",
    lat_color="k",
    fig=None,
    ax=None,
):
    """Plot lattice sites and Wannier centers in two dimensions."""
    plt = require_matplotlib()
    if wan.lattice.dim_r < 2:
        raise ValueError("Wannier-center plotting requires at least two dimensions.")
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    data = wan._get_sc_weights(0)
    ax.scatter(
        data["all"]["xs"],
        data["all"]["ys"],
        color=lat_color,
        s=12,
    )
    if color_home_cell:
        ax.scatter(
            data["home"]["xs"],
            data["home"]["ys"],
            color=lat_home_color,
            s=18,
        )
    vectors = wan.lattice.lat_vecs
    if section_home_cell:
        corners = np.asarray(
            [
                [0.0, 0.0],
                vectors[0, :2],
                (vectors[0] + vectors[1])[:2],
                vectors[1, :2],
                [0.0, 0.0],
            ]
        )
        ax.plot(corners[:, 0], corners[:, 1], "k--", linewidth=1)
    translations = [(0, 0)]
    if translate_centers:
        translations = list(
            itertools.product(range(-pmx, pmx + 1), range(-pmy, pmy + 1))
        )
    for translation_index, (cell_x, cell_y) in enumerate(translations):
        shift = cell_x * vectors[0] + cell_y * vectors[1]
        for index, (center, spread) in enumerate(
            zip(wan.centers, wan.spread, strict=True)
        ):
            ax.scatter(
                center[0] + shift[0],
                center[1] + shift[1],
                s=center_scale * max(1.0, 1.0 + float(spread)),
                color=center_color,
                marker=center_marker,
                label=(
                    "Wannier centers"
                    if index == 0 and translation_index == 0
                    else None
                ),
            )
    if legend:
        ax.legend()
    center = 0.5 * (vectors[0] + vectors[1])
    ax.set_xlim(center[0] - pmx, center[0] + pmx)
    ax.set_ylim(center[1] - pmy, center[1] + pmy)
    ax.set_aspect("equal", adjustable="box")
    if show:
        plt.show()
    return fig, ax
