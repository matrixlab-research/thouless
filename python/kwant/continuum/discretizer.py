"""Finite-difference discretization of symbolic continuum Hamiltonians."""

from __future__ import annotations

from collections import defaultdict
import inspect
import itertools
import keyword
import warnings

import numpy as np
import sympy
import tinyarray as ta
from sympy.core.function import AppliedUndef
from thouless import _core

from .. import builder, lattice
from .._common import KwantDeprecationWarning, reraise_warnings
from ._common import (
    momentum_operators,
    position_operators,
    sympify,
)


_wf = sympy.Function("_internal_unique_name", commutative=False)
_momenta = {symbol.name: symbol for symbol in momentum_operators}
_positions = {symbol.name: symbol for symbol in position_operators}
_spacing = {
    coordinate: sympy.Symbol(f"a_{coordinate}") for coordinate in "xyz"
}


class _DiscretizedBuilder(builder.Builder):
    """A translational template that retains its grid and source rendering."""

    def __init__(self, coords, grid, symmetry):
        super().__init__(symmetry)
        self.lattice = grid
        self._coords = tuple(coords)

    def __str__(self):
        pairs = list(self.site_value_pairs())
        if len(pairs) != 1:
            raise ValueError(
                "Cannot pretty-print _DiscretizedBuilder: "
                "must contain a single site."
            )
        origin, onsite = pairs[0]
        if any(origin.tag):
            raise ValueError(
                "Cannot pretty-print _DiscretizedBuilder: "
                "site must be located at origin."
            )
        sections = [
            "# Discrete coordinates: ",
            " ".join(self._coords),
            "\n\n# Onsite element:\n",
            onsite._source if callable(onsite) else repr(onsite),
        ]
        for (first, second), value in self.hopping_value_pairs():
            sections.extend(
                [
                    "\n\n# Hopping from ",
                    str(tuple(second.tag)),
                    ":\n",
                    value._source if callable(value) else repr(value),
                ]
            )
        return "".join(sections)

    def _repr_html_(self):
        return str(self)


def _validate_coords(coords):
    coords = list(coords)
    if coords != sorted(coords):
        raise ValueError("The argument 'coords' must be sorted.")
    if any(coordinate not in "xyz" for coordinate in coords):
        raise ValueError(
            "The argument 'coords' may only contain 'x', 'y', or 'z'."
        )
    if not coords:
        raise ValueError("Discrete coordinates cannot be empty.")
    return coords


def _native_hoppings(summand, coords):
    momenta = {f"k_{coordinate}" for coordinate in coords}
    coefficients = []
    descriptors = []
    for factor in summand.as_ordered_factors():
        base, exponent = factor.as_base_exp()
        if isinstance(base, sympy.Symbol) and base.name in momenta:
            if not exponent.is_integer or int(exponent) < 0:
                raise ValueError("Momentum powers must be nonnegative integers.")
            descriptors.append(
                (
                    coords.index(base.name[-1]),
                    0,
                    int(exponent),
                )
            )
        else:
            identifier = len(coefficients)
            coefficients.append(factor)
            descriptors.append((None, identifier, 1))
    output = defaultdict(lambda: sympy.Integer(0))
    for offset, weight, inverse_powers, shifted_coefficients in (
        _core.continuum_finite_difference_stencil(
            len(coords),
            descriptors,
        )
    ):
        factors = []
        for identifier, shifts in shifted_coefficients:
            substitutions = {
                _positions[coordinate]: (
                    _positions[coordinate]
                    + sympy.Rational(numerator, denominator)
                    * _spacing[coordinate]
                )
                for coordinate, (numerator, denominator) in zip(
                    coords,
                    shifts,
                    strict=True,
                )
                if numerator
            }
            factors.append(coefficients[identifier].subs(substitutions))
        exact_weight = (
            sympy.Rational(str(weight.real))
            + sympy.I * sympy.Rational(str(weight.imag))
        )
        for coordinate, power in zip(
            coords,
            inverse_powers,
            strict=True,
        ):
            exact_weight /= _spacing[coordinate] ** power
        output[tuple(offset)] += exact_weight * sympy.Mul(*factors)
    return dict(output)


def _discretize_expression(expression, coords):
    symbol_names = {symbol.name for symbol in expression.atoms(sympy.Symbol)}
    if not set(_momenta).intersection(symbol_names):
        return {(0,) * len(coords): expression}

    result = defaultdict(lambda: sympy.Integer(0))
    for summand in sympy.expand(expression).as_ordered_terms():
        for offset, value in _native_hoppings(summand, coords).items():
            result[offset] += value
    return dict(result)


def discretize_symbolic(hamiltonian, coords=None, *, locals=None):
    """Convert a symbolic differential operator into lattice hoppings."""
    with reraise_warnings():
        hamiltonian = sympify(hamiltonian, locals)
    if (
        isinstance(hamiltonian, sympy.Float)
        and float(hamiltonian).is_integer()
    ):
        hamiltonian = sympy.Integer(int(hamiltonian))
    elif (
        isinstance(hamiltonian, sympy.MatrixBase)
        and all(
            isinstance(value, sympy.Float) and float(value).is_integer()
            for value in hamiltonian
        )
    ):
        hamiltonian = hamiltonian.applyfunc(
            lambda value: sympy.Integer(int(value))
        )

    names = {symbol.name for symbol in hamiltonian.atoms(sympy.Symbol)}
    reserved = {"a_x", "a_y", "a_z"}.intersection(names)
    if reserved:
        raise TypeError(
            "'a_x', 'a_y' and 'a_z' are symbols used internally "
            "to represent grid spacings; please use a different symbol."
        )
    if coords is None:
        coords = sorted(
            name[-1] for name in names if name in _momenta
        )
        if not coords:
            raise ValueError(
                "Failed to read any discrete coordinates. "
                "Use the 'coords' parameter when no momentum operator is present."
            )
    coords = _validate_coords(coords)

    matrix_input = isinstance(hamiltonian, sympy.MatrixBase)
    matrix = hamiltonian if matrix_input else sympy.Matrix([hamiltonian])
    zero = (0,) * len(coords)
    tight_binding = defaultdict(lambda: sympy.zeros(*matrix.shape))
    tight_binding[zero] = sympy.zeros(*matrix.shape)
    for row in range(matrix.rows):
        for column in range(matrix.cols):
            for offset, value in _discretize_expression(
                matrix[row, column],
                coords,
            ).items():
                tight_binding[offset][row, column] += value

    ordered_offsets = sorted(tight_binding)
    wanted = set(ordered_offsets[len(ordered_offsets) // 2 :])
    result = {
        offset: value
        for offset, value in tight_binding.items()
        if offset in wanted
    }
    if not matrix_input:
        result = {
            offset: value[0, 0] for offset, value in result.items()
        }
    return result, coords


def _value_function(expression, coords, grid_spacing, onsite, name):
    expression = expression.subs(
        {
            _spacing[coordinate]: grid_spacing[index]
            for index, coordinate in enumerate(coords)
        }
    )
    coordinate_names = set(coords).intersection(
        symbol.name for symbol in expression.atoms(sympy.Symbol)
    )
    function_calls = expression.atoms(AppliedUndef, sympy.Function)
    function_names = {str(function.func) for function in function_calls}
    symbol_names = {
        symbol.name
        for symbol in expression.atoms(sympy.Symbol)
        if symbol.name not in set(coords)
    }
    parameter_names = sorted(symbol_names | function_names)
    for parameter in parameter_names:
        if not parameter.isidentifier() or keyword.iskeyword(parameter):
            raise ValueError(
                f"Invalid name in used symbols: {parameter}\n"
                "Names of symbols used in Hamiltonian must be valid "
                "Python identifiers and may not be keywords"
            )

    evaluation_names = sorted(
        coordinate_names | symbol_names | function_names
    )
    evaluator = sympy.lambdify(evaluation_names, expression, modules="numpy")
    site_names = ["site"] if onsite else ["site1", "site2"]
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
        site = bound.arguments[site_names[0]]
        values = {}
        if coordinate_names:
            values.update(
                {
                    coordinate: coordinate_value
                    for coordinate, coordinate_value in zip(
                        coords,
                        site.pos,
                        strict=True,
                    )
                    if coordinate in coordinate_names
                }
            )
        values.update(
            {
                parameter: bound.arguments[parameter]
                for parameter in parameter_names
            }
        )
        result = evaluator(*(values[name] for name in evaluation_names))
        array = np.asarray(result)
        return (
            ta.array(array, complex)
            if array.ndim
            else complex(array)
        )

    value.__name__ = name
    value.__signature__ = signature
    parameters = ", ".join([*site_names, *parameter_names])
    value._source = f"def {name}({parameters}):\n    return {expression}"

    if not parameter_names and not coordinate_names:
        result = evaluator()
        array = np.asarray(result)
        return (
            ta.array(array, complex)
            if array.ndim
            else complex(array)
        )
    return value


def build_discretized(
    tb_hamiltonian,
    coords,
    *,
    grid=None,
    locals=None,
    grid_spacing=None,
):
    """Build a translational template from symbolic onsite and hopping terms."""
    coords = _validate_coords(coords)
    if grid_spacing is not None:
        warnings.warn(
            'The "grid_spacing" parameter is deprecated. Use "grid" instead.',
            KwantDeprecationWarning,
            stacklevel=3,
        )
    if grid is None:
        grid = 1 if grid_spacing is None else grid_spacing
    elif grid_spacing is not None:
        raise ValueError('"grid_spacing" and "grid" are mutually exclusive.')

    with reraise_warnings():
        symbolic = {
            tuple(offset): sympify(value, locals)
            for offset, value in tb_hamiltonian.items()
        }
    if not symbolic:
        raise ValueError("The tight-binding Hamiltonian cannot be empty.")
    if any(len(offset) != len(coords) for offset in symbolic):
        raise ValueError("Hopping offsets must match the discrete dimension.")
    sample = next(iter(symbolic.values()))
    norbs = sample.rows if isinstance(sample, sympy.MatrixBase) else 1
    if isinstance(sample, sympy.MatrixBase) and sample.rows != sample.cols:
        raise ValueError("Hamiltonian matrices must be square.")

    if np.isscalar(grid):
        primitive = float(grid) * np.eye(len(coords))
        grid = lattice.Monatomic(primitive, norbs=norbs)
    if not isinstance(grid, lattice.Monatomic):
        raise ValueError("grid must be a scalar or a Monatomic lattice.")
    primitive = np.asarray(grid.prim_vecs, dtype=float)
    if (
        primitive.shape != (len(coords), len(coords))
        or not np.allclose(primitive, np.diag(np.diag(primitive)))
    ):
        raise ValueError(
            '"grid" has to be an orthogonal lattice '
            'of dimension matching number of "coords".'
        )
    if grid.norbs is not None and grid.norbs != norbs:
        raise ValueError(
            "Number of lattice orbitals does not match the number "
            "of orbitals in the Hamiltonian."
        )

    numeric = {}
    for index, (offset, expression) in enumerate(symbolic.items()):
        onsite = all(value == 0 for value in offset)
        name = "onsite" if onsite else f"hopping_{index}"
        numeric[offset] = _value_function(
            expression,
            coords,
            np.diag(primitive),
            onsite,
            name,
        )
    zero = (0,) * len(coords)
    if zero not in numeric:
        raise ValueError("The tight-binding Hamiltonian requires an onsite term.")
    onsite = numeric.pop(zero)

    system = _DiscretizedBuilder(
        coords,
        grid,
        lattice.TranslationalSymmetry(*primitive),
    )
    origin = grid(*zero)
    system[origin] = onsite
    for offset, value in numeric.items():
        kind = builder.HoppingKind(
            tuple(-component for component in offset),
            grid,
        )
        system[kind] = value
    return system


def discretize(
    hamiltonian,
    coords=None,
    *,
    grid=None,
    locals=None,
    grid_spacing=None,
):
    """Discretize a continuum Hamiltonian and build its lattice template."""
    tight_binding, discrete_coords = discretize_symbolic(
        hamiltonian,
        coords,
        locals=locals,
    )
    return build_discretized(
        tight_binding,
        discrete_coords,
        grid=grid,
        grid_spacing=grid_spacing,
    )


__all__ = [
    "_wf",
    "build_discretized",
    "discretize",
    "discretize_symbolic",
]
