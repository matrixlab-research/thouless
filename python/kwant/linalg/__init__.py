"""Kwant linear-algebra compatibility modules."""

from . import lapack, lll, mumps
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
    "lapack",
    "lll",
    "mumps",
    "order_gen_schur",
    "order_schur",
    "schur",
]
