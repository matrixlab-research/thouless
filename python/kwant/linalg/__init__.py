"""Kwant linear-algebra compatibility modules."""

from . import lll
from .decomp_schur import (
    convert_r2c_gen_schur,
    convert_r2c_schur,
    evecs_from_gen_schur,
    evecs_from_schur,
    gen_schur,
    order_gen_schur,
    order_schur,
    schur,
)

__all__ = [
    "convert_r2c_gen_schur",
    "convert_r2c_schur",
    "evecs_from_gen_schur",
    "evecs_from_schur",
    "gen_schur",
    "lll",
    "order_gen_schur",
    "order_schur",
    "schur",
]
