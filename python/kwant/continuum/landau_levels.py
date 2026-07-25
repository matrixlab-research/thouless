"""Landau-level projection of continuum Hamiltonians."""

from __future__ import annotations

from collections import defaultdict
import inspect
import keyword

import numpy as np
import sympy
import tinyarray as ta
from sympy.core.function import AppliedUndef
from thouless import _core

from .. import builder, lattice
from ._common import (
    momentum_operators,
    monomials,
    position_operators,
    sympify,
)
from .discretizer import _spacing, discretize_symbolic


ladder_lower, ladder_raise = sympy.symbols(
    r"a a^\dagger",
    commutative=False,
)


def _normalize_momenta(momenta=None):
    if momenta is None:
        return tuple(momentum_operators[:2])
    if len(momenta) != 2:
        raise ValueError("Two momenta must be specified.")
    if all(type(value) is int and 0 <= value < 3 for value in momenta):
        return tuple(momentum_operators[value] for value in momenta)
    names = [momentum.name for momentum in momentum_operators]
    if all(isinstance(value, str) and value in names for value in momenta):
        return tuple(momentum_operators[names.index(value)] for value in momenta)
    raise ValueError("Momenta must all be integers or strings.")


def _normal_coordinate(momenta):
    remaining = next(
        momentum
        for momentum in momentum_operators
        if momentum not in momenta
    )
    return position_operators[momentum_operators.index(remaining)]


def to_landau_basis(hamiltonian, momenta=None):
    """Replace two momentum components by harmonic-oscillator operators."""
    hamiltonian = sympify(hamiltonian)
    momenta = _normalize_momenta(momenta)
    normal_coordinate = _normal_coordinate(momenta)
    field = sympy.Symbol("B")
    scale = sympy.sqrt(sympy.Abs(field) / 2)
    transformed = hamiltonian.subs(
        {
            momenta[0]: scale * (ladder_raise + ladder_lower),
            momenta[1]: sympy.I * scale * (ladder_lower - ladder_raise),
        }
    )
    return transformed, momenta, normal_coordinate


def _ladder_term(operator_string):
    """Encode an ordered ladder monomial as signed integer powers."""
    result = []
    for factor in operator_string.as_ordered_factors():
        operator, exponent = factor.as_base_exp()
        sign = (
            -1
            if operator == ladder_lower
            else 1
            if operator == ladder_raise
            else 0
        )
        result.append(sign * int(exponent))
    return tuple(result)


def _ladder_term_name(ladder_term):
    encoded = [
        str(value) if value >= 0 else f"_{-value}"
        for value in ladder_term
    ]
    return "_ladder_" + "_".join(encoded)


def _evaluate_ladder_term(ladder_term, n, B):
    """Evaluate an ordered ladder string on ``|n>``."""
    if n < 0:
        raise ValueError("Landau level index must be nonnegative.")
    return _core.continuum_landau_ladder_coefficient(
        list(ladder_term),
        int(n),
        float(B),
    )


class LandauLattice(lattice.Monatomic):
    """One real-space coordinate plus an integer Landau-level index."""

    def __init__(self, grid_spacing, offset=None, name="", norbs=None):
        lattice_offset = None if offset is None else [offset, 0]
        super().__init__(
            [[grid_spacing, 0], [0, 1]],
            lattice_offset,
            name,
            norbs,
        )

    def pos(self, tag):
        return ta.array(
            (
                self.prim_vecs[0, 0] * tag[0] + self.offset[0],
            )
        )

    @staticmethod
    def landau_index(tag):
        return tag[-1]


def _has_normal_coordinate(coordinate, expression):
    momentum = momentum_operators[position_operators.index(coordinate)]
    atoms = set(expression.atoms())
    return coordinate in atoms or momentum in atoms


def _landau_value(terms, normal_coordinate, grid_spacing, onsite):
    compiled = []
    parameter_names = set()
    for ladder_term, expression in terms:
        expression = expression.subs(
            {_spacing[normal_coordinate]: grid_spacing}
        )
        symbols = {
            symbol.name
            for symbol in expression.atoms(sympy.Symbol)
            if symbol.name != normal_coordinate
        }
        functions = {
            str(function.func)
            for function in expression.atoms(AppliedUndef, sympy.Function)
            if str(function.func) != "Abs"
        }
        names = sorted(
            (
                {normal_coordinate}
                if any(
                    symbol.name == normal_coordinate
                    for symbol in expression.atoms(sympy.Symbol)
                )
                else set()
            )
            | symbols
            | functions
        )
        compiled.append(
            (
                ladder_term,
                names,
                sympy.lambdify(
                    names,
                    expression,
                    modules=[{"Abs": np.abs}, "numpy"],
                ),
            )
        )
        parameter_names.update(symbols)
        parameter_names.update(functions)
    parameter_names = sorted(parameter_names)
    for parameter in parameter_names:
        if not parameter.isidentifier() or keyword.iskeyword(parameter):
            raise ValueError(
                f"Invalid name in used symbols: {parameter}\n"
                "Names of symbols used in Hamiltonian must be valid "
                "Python identifiers and may not be keywords"
            )

    site_names = ["from_site"] if onsite else ["to_site", "from_site"]
    signature = inspect.Signature(
        [
            *(
                inspect.Parameter(
                    site_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for site_name in site_names
            ),
            *(
                inspect.Parameter(
                    parameter,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for parameter in parameter_names
            ),
        ]
    )

    def value(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        from_site = bound.arguments["from_site"]
        to_site = from_site if onsite else bound.arguments["to_site"]
        field = bound.arguments.get("B", 1)
        reference = to_site if field < 0 and not onsite else from_site
        level = reference.family.landau_index(reference.tag)
        coordinate_value = from_site.pos[0]
        parameters = {
            parameter: bound.arguments[parameter]
            for parameter in parameter_names
        }
        parameters[normal_coordinate] = coordinate_value
        result = 0
        for ladder_term, names, evaluator in compiled:
            coefficient = _evaluate_ladder_term(
                ladder_term,
                level,
                field,
            )
            if coefficient:
                result = result + coefficient * evaluator(
                    *(parameters[name] for name in names)
                )
        array = np.asarray(result)
        return ta.array(array, complex) if array.ndim else complex(array)

    value.__signature__ = signature
    value.__name__ = "onsite" if onsite else "hopping"
    parameters = ", ".join([*site_names, *parameter_names])
    value._source = (
        f"def {value.__name__}({parameters}):\n"
        "    # Landau ladder terms evaluated at runtime"
    )
    return value


def discretize_landau(
    hamiltonian,
    N,
    momenta=None,
    grid_spacing=1,
):
    """Discretize a continuum Hamiltonian in a truncated Landau basis."""
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError("N must be positive")
    transformed, _, normal_coordinate = to_landau_basis(
        hamiltonian,
        momenta,
    )
    tight_binding, _ = discretize_symbolic(
        transformed,
        coords=[normal_coordinate.name],
    )
    grouped = defaultdict(list)
    for spatial_offset, expression in tight_binding.items():
        for operator_string, coefficient in monomials(
            expression,
            gens=(ladder_lower, ladder_raise),
        ).items():
            ladder_term = _ladder_term(operator_string)
            grouped[(*spatial_offset, sum(ladder_term))].append(
                (ladder_term, coefficient)
            )

    sample = next(iter(grouped.values()))[0][1]
    norbs = sample.rows if isinstance(sample, sympy.MatrixBase) else 1
    onsite_terms = grouped.pop((0, 0), None)
    grid = LandauLattice(grid_spacing, norbs=norbs)
    symmetry = (
        lattice.TranslationalSymmetry([grid_spacing, 0])
        if _has_normal_coordinate(normal_coordinate, transformed)
        else builder.NoSymmetry()
    )
    system = builder.Builder(symmetry)
    sites = [grid(0, level) for level in range(int(N))]
    if onsite_terms is None:
        system[sites] = ta.zeros((norbs, norbs))
    else:
        system[sites] = _landau_value(
            onsite_terms,
            normal_coordinate.name,
            grid_spacing,
            True,
        )

    system[builder.HoppingKind((0, 1), grid)] = ta.zeros(
        (norbs, norbs)
    )
    for offset, terms in grouped.items():
        system[builder.HoppingKind(offset, grid)] = _landau_value(
            terms,
            normal_coordinate.name,
            grid_spacing,
            False,
        )
    return system


__all__ = [
    "LandauLattice",
    "discretize_landau",
    "ladder_lower",
    "ladder_raise",
    "to_landau_basis",
]
