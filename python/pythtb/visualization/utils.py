"""Lazy plotting dependencies and geometric projections."""

from __future__ import annotations

import numpy as np


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "PythTB visualization requires the optional matplotlib dependency."
        ) from error
    return plt


def require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "Three-dimensional visualization requires the optional plotly dependency."
        ) from error
    return go


def project(vector, plane=None):
    """Project a zero- to three-dimensional vector into a drawing plane."""
    values = np.asarray(vector, dtype=float)
    if values.ndim != 1 or values.size > 3:
        raise ValueError("Visualization supports vectors with at most three components.")
    if values.size == 0:
        return np.zeros(2)
    if plane is not None:
        indices = tuple(int(index) for index in plane)
        if len(indices) != 2 or any(
            index < 0 or index >= values.size for index in indices
        ):
            raise ValueError("proj_plane must name two valid Cartesian components.")
        return values[np.asarray(indices)]
    result = np.zeros(2)
    result[: min(2, values.size)] = values[:2]
    return result
