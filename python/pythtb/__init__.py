"""PythTB 2.0 compatibility layer for the Thouless Rust core."""

import logging

from .lattice import Lattice
from .mesh import Mesh
from .tbmodel import TBModel, tb_model
from .utils import (
    finite_diff_coeffs,
    finite_difference,
    get_trial_wfs,
    is_Hermitian,
    levi_civita,
    pauli_decompose,
)
from .w90 import W90
from .wannier import Wannier
from .wfarray import WFArray, wf_array

__version__ = "2.0.0+thouless"
__author__ = "Thouless contributors"
__license__ = "MIT"

_LOGGER_NAME = __name__.split(".")[0]
_DEFAULT_HANDLER = None


def configure_logging(
    level="INFO",
    *,
    handler=None,
    fmt="%(levelname)s %(name)s: %(message)s",
    propagate=False,
):
    """Configure the package logger without stacking duplicate handlers."""
    global _DEFAULT_HANDLER
    logger = logging.getLogger(_LOGGER_NAME)
    if isinstance(level, str):
        numeric_level = logging.getLevelName(level.upper())
        if not isinstance(numeric_level, int):
            raise ValueError(f"Unknown logging level: {level}")
    elif isinstance(level, int):
        numeric_level = level
    else:
        raise TypeError("Logging level must be an int or a named level")
    if handler is None:
        if _DEFAULT_HANDLER is None:
            _DEFAULT_HANDLER = logging.StreamHandler()
        handler = _DEFAULT_HANDLER
    logger.handlers = [handler]
    if fmt:
        handler.setFormatter(logging.Formatter(fmt))
    logger.setLevel(numeric_level)
    logger.propagate = bool(propagate)
    return handler


def set_log_level(level):
    """Set the package logging level using the default stream handler."""
    configure_logging(level)


__all__ = [
    "Lattice",
    "Mesh",
    "TBModel",
    "WFArray",
    "W90",
    "Wannier",
    "finite_diff_coeffs",
    "finite_difference",
    "get_trial_wfs",
    "is_Hermitian",
    "levi_civita",
    "pauli_decompose",
    "tb_model",
    "wf_array",
]
