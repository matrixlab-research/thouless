"""Optional plotting-backend discovery without selecting a GUI backend."""

from importlib.util import find_spec
import warnings

mpl_available = find_spec("matplotlib") is not None
plotly_available = find_spec("plotly") is not None

if not mpl_available:
    warnings.warn(
        "matplotlib is not available; plotting with that engine is disabled",
        RuntimeWarning,
        stacklevel=2,
    )

if not plotly_available:
    warnings.warn(
        "plotly is not available; plotting with that engine is disabled",
        RuntimeWarning,
        stacklevel=2,
    )

engines = frozenset(
    engine
    for engine, available in (
        ("matplotlib", mpl_available),
        ("plotly", plotly_available),
    )
    if available
)
engine = (
    "matplotlib"
    if mpl_available
    else "plotly"
    if plotly_available
    else None
)
