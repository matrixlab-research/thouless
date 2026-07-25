"""Symbolic primitives shared by continuum-model discretization."""

from __future__ import annotations

import keyword
from collections import defaultdict
import warnings

import numpy as np
import sympy
import sympy.abc
import sympy.physics.quantum
from sympy.core import Basic
from sympy.core.function import AppliedUndef
from sympy.physics.matrices import msigma

from .._common import reraise_warnings


momentum_operators = sympy.symbols("k_x k_y k_z", commutative=False)
position_operators = sympy.symbols("x y z", commutative=False)

_pauli = (sympy.eye(2), msigma(1), msigma(2), msigma(3))
_namespace = sympy.abc._clash.copy()
_namespace.update(
    {symbol.name: symbol for symbol in (*momentum_operators, *position_operators)}
)
_namespace.update(
    {
        "kron": sympy.physics.quantum.TensorProduct,
        "eye": sympy.eye,
        "identity": sympy.eye,
    }
)
_namespace.update(
    {f"sigma_{axis}": matrix for axis, matrix in zip("0xyz", _pauli)}
)
_namespace.pop("I", None)
_namespace.pop("pi", None)


def _undefined_functions(expression):
    substitutions = {
        function: sympy.Function(str(function.func))(*function.args)
        for function in expression.atoms(sympy.Function)
    }
    return expression.subs(substitutions)


def sympify(expression, locals=None):
    """Parse a scalar or matrix continuum Hamiltonian.

    Cartesian positions and momenta are represented by noncommuting symbols.
    List literals become matrices, and callers may add a validated local
    namespace whose string values are parsed under the same rules.
    """
    if isinstance(expression, Basic):
        if locals:
            warnings.warn(
                'Input expression is already SymPy object: "locals" will not be used.',
                RuntimeWarning,
                stacklevel=2,
            )
        return _undefined_functions(expression)

    definitions = {} if locals is None else dict(locals)
    for name in definitions:
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or keyword.iskeyword(name)
        ):
            raise ValueError(
                f"Invalid key in 'locals': {name!r}\n"
                "Keys must be identifiers and may not be keywords"
            )

    parsed_definitions = {}
    for name, value in definitions.items():
        if isinstance(value, np.ndarray):
            parsed_definitions[name] = sympy.Matrix(value)
        elif isinstance(value, sympy.MatrixBase):
            parsed_definitions[name] = value
        else:
            parsed_definitions[name] = sympify(value)
    for name, value in _namespace.items():
        parsed_definitions.setdefault(name, value)

    result = sympy.sympify(expression, locals=parsed_definitions)
    if isinstance(result, list):
        result = sympy.Matrix(result)
    return sympy.sympify(result)


def lambdify(expression, locals=None):
    """Compile a symbolic continuum expression into a keyword-callable."""
    with reraise_warnings(level=4):
        expression = sympify(expression, locals)
    arguments = [symbol.name for symbol in expression.atoms(sympy.Symbol)]
    arguments.extend(
        str(function.func)
        for function in expression.atoms(AppliedUndef, sympy.Function)
    )
    return sympy.lambdify(sorted(arguments), expression)


def make_commutative(expression, *symbols):
    """Replace named noncommuting symbols by commuting counterparts."""
    noncommuting = [
        sympy.Symbol(symbol.name, commutative=False) for symbol in symbols
    ]
    return expression.subs(
        {
            symbol: sympy.Symbol(symbol.name)
            for symbol in noncommuting
        }
    )


def _expression_monomials(expression, generators):
    expression = sympy.expand(expression)
    output = defaultdict(lambda: sympy.Integer(0))
    for summand in expression.as_ordered_terms():
        key = []
        coefficient = []
        for factor in summand.as_ordered_factors():
            base, _ = factor.as_base_exp()
            (key if base in generators else coefficient).append(factor)
        output[sympy.Mul(*key)] += sympy.Mul(*coefficient)
    return dict(output)


def monomials(expression, gens=None):
    """Group an expression by ordered monomials in selected generators."""
    generators = (
        expression.atoms(sympy.Symbol)
        if gens is None
        else [sympify(generator) for generator in gens]
    )
    if not isinstance(expression, sympy.MatrixBase):
        return _expression_monomials(expression, generators)

    output = defaultdict(lambda: sympy.zeros(*expression.shape))
    for row in range(expression.rows):
        for column in range(expression.cols):
            for key, value in _expression_monomials(
                expression[row, column],
                generators,
            ).items():
                output[key][row, column] += value
    return dict(output)


def gcd(*values):
    """Return the greatest common divisor of one or more integers."""
    if len(values) == 1:
        return values[0]
    pending = list(values)
    while len(pending) > 1:
        first, second = pending[-2:]
        del pending[-2:]
        while first:
            first, second = second % first, first
        pending.append(second)
    return abs(second)


__all__ = [
    "gcd",
    "lambdify",
    "make_commutative",
    "momentum_operators",
    "monomials",
    "position_operators",
    "sympify",
]
