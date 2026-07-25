"""Tight-binding model and band-structure visualization."""

from __future__ import annotations

import numpy as np

from .lattice import plot_lattice, plot_lattice_3d
from .utils import project, require_matplotlib


def _spinless_site_values(model):
    onsite = np.asarray(model.onsite)
    if model.spinful:
        return np.real(np.trace(onsite, axis1=-2, axis2=-1) / 2)
    return np.real(onsite)


def plot_tbmodel(
    model,
    proj_plane=None,
    eig_dr=None,
    draw_hoppings=True,
    annotate_onsite=False,
    ph_color="black",
    orb_color="red",
    show=True,
):
    """Draw a tight-binding model and optional eigenstate amplitudes."""
    del show
    fig, ax = plot_lattice(
        model.lattice,
        proj_plane=proj_plane,
        orb_color=orb_color,
    )
    orbital_cartesian = model.get_orb_vecs(cartesian=True)
    if draw_hoppings:
        for hopping in model._hoppings:
            start = orbital_cartesian[hopping["target"]]
            translation = np.asarray(hopping["offset"]) @ model.lat_vecs
            end = orbital_cartesian[hopping["source"]] + translation
            start_2d = project(start, proj_plane)
            end_2d = project(end, proj_plane)
            ax.plot(
                [start_2d[0], end_2d[0]],
                [start_2d[1], end_2d[1]],
                color="tab:green",
                linewidth=1.5,
                zorder=1,
            )
    if annotate_onsite:
        for index, (position, energy) in enumerate(
            zip(orbital_cartesian, _spinless_site_values(model), strict=True)
        ):
            ax.annotate(
                f"{index}: {energy:.3g}",
                project(position, proj_plane),
                xytext=(4, 4),
                textcoords="offset points",
            )
    if eig_dr is not None:
        state = np.asarray(eig_dr, dtype=complex)
        if model.spinful:
            state = state.reshape(model.norb, model.nspin)
            weight = np.sum(np.abs(state) ** 2, axis=-1)
            phase = np.angle(state[:, 0])
        else:
            state = state.reshape(model.norb)
            weight = np.abs(state) ** 2
            phase = np.angle(state)
        positions = np.asarray(
            [project(position, proj_plane) for position in orbital_cartesian]
        )
        if ph_color == "black":
            colors = "black"
        elif ph_color == "red-blue":
            colors = np.cos(phase)
        elif ph_color == "wheel":
            colors = np.mod(phase, 2 * np.pi)
        else:
            raise ValueError("ph_color must be 'black', 'red-blue', or 'wheel'.")
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=40 + 300 * weight,
            c=colors,
            cmap="hsv" if ph_color == "wheel" else "coolwarm",
            zorder=5,
        )
    return fig, ax


def plot_tbmodel_3d(
    model,
    draw_hoppings=True,
    show_model_info=True,
    site_colors=None,
    site_names=None,
    show=True,
):
    """Build a Plotly figure for a three-dimensional model."""
    figure = plot_lattice_3d(
        model.lattice,
        site_colors=site_colors,
        site_names=site_names,
        show_lattice_info=show_model_info,
    )
    if draw_hoppings:
        import plotly.graph_objects as go

        orbitals = model.get_orb_vecs(cartesian=True)
        for hopping in model._hoppings:
            start = orbitals[hopping["target"]]
            end = (
                orbitals[hopping["source"]]
                + np.asarray(hopping["offset"]) @ model.lat_vecs
            )
            figure.add_trace(
                go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode="lines",
                    line={"color": "green", "width": 3},
                    showlegend=False,
                )
            )
    if show:
        figure.show()
        return None
    return figure


def plot_bands(
    model,
    k_path,
    nk=101,
    evals=None,
    evecs=None,
    ktick_labels=None,
    bands_label=None,
    proj_orb_idx=None,
    proj_spin=False,
    fig=None,
    ax=None,
    scat_size=3,
    lw=2,
    lc="b",
    ls="solid",
    cmap="plasma",
    cbar=True,
):
    """Plot model eigenvalues along a reciprocal-space path."""
    plt = require_matplotlib()
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    points, distance, node_distance = model.k_path(k_path, nk, report=False)
    if evals is None or (
        (proj_orb_idx is not None or proj_spin) and evecs is None
    ):
        solved = model.solve_ham(
            points,
            return_eigvecs=proj_orb_idx is not None or proj_spin,
            flatten_spin_axis=False,
        )
        if isinstance(solved, tuple):
            evals, evecs = solved
        else:
            evals = solved
    evals = np.atleast_2d(np.asarray(evals, dtype=float))

    projected = proj_orb_idx is not None or proj_spin
    if projected:
        states = np.asarray(evecs, dtype=complex)
        if model.spinful:
            states = states.reshape(states.shape[:-1] + (model.norb, model.nspin))
        if proj_orb_idx is not None:
            indices = np.asarray(proj_orb_idx, dtype=int)
            if model.spinful:
                weights = np.abs(states[..., indices, :]) ** 2
                colors = weights.sum(axis=(-2, -1))
            else:
                weights = np.abs(states[..., indices]) ** 2
                colors = weights.sum(axis=-1)
        else:
            if not model.spinful:
                raise ValueError("Spin projection requires a spinful model.")
            colors = np.sum(np.abs(states[..., 0]) ** 2, axis=-1)
        artist = None
        for band in range(evals.shape[-1]):
            artist = ax.scatter(
                distance,
                evals[:, band],
                c=colors[:, band],
                cmap=cmap,
                s=scat_size,
                vmin=0,
                vmax=1,
                label=bands_label if band == 0 else None,
            )
        if cbar and artist is not None:
            fig.colorbar(artist, ax=ax)
    else:
        for band in range(evals.shape[-1]):
            ax.plot(
                distance,
                evals[:, band],
                color=lc,
                linewidth=lw,
                linestyle=ls,
                label=bands_label if band == 0 else None,
            )
    for position in node_distance:
        ax.axvline(position, color="black", linewidth=0.5)
    ax.set_xlim(node_distance[0], node_distance[-1])
    ax.set_xticks(node_distance)
    if ktick_labels is not None:
        ax.set_xticklabels(ktick_labels)
    ax.set_ylabel("Energy")
    if bands_label is not None:
        ax.legend()
    return fig, ax
