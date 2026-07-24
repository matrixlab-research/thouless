"""PythTB 2.0 compatibility layer for the Thouless Rust core."""

from .lattice import Lattice
from .mesh import Mesh
from .tbmodel import TBModel, tb_model
from .wfarray import WFArray, wf_array

__version__ = "2.0.0+thouless"
__author__ = "Thouless contributors"
__license__ = "MIT"

__all__ = [
    "Lattice",
    "Mesh",
    "TBModel",
    "WFArray",
    "tb_model",
    "wf_array",
]
