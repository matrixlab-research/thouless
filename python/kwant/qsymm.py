"""Conversion between Kwant builders and the optional qsymm model algebra."""

from __future__ import annotations

import itertools
from collections import OrderedDict, defaultdict

import numpy as np
from scipy import linalg

import qsymm as _qsymm
import sympy
from qsymm.groups import ContinuousGroupGenerator, PointGroupElement
from qsymm.hamiltonian_generator import hamiltonian_from_family
from qsymm.linalg import allclose
from qsymm.model import BlochCoeff, BlochModel, Model
from qsymm.symmetry_finder import bravais_point_group

from . import builder as builder_module
from . import lattice
from ._common import get_parameters


def _orbital_layout(system):
    offsets = {}
    cursor = 0
    for site in sorted(system.sites()):
        orbitals = site.family.norbs
        if orbitals is None:
            raise ValueError("norbs must be provided for every site family")
        offsets[site] = slice(cursor, cursor + int(orbitals))
        cursor += int(orbitals)
    return offsets, cursor


def _matrix_with_block(size, rows, columns, value):
    matrix = np.zeros((size, size), dtype=complex)
    matrix[rows, columns] = value
    return matrix


def _linear_parameter_terms(value, sites, fixed_parameters):
    names = get_parameters(value)[len(sites) :]

    def evaluate(selected):
        arguments = [
            fixed_parameters[name] if name in fixed_parameters else selected.get(name, 0)
            for name in names
        ]
        return np.asarray(value(*sites, *arguments), dtype=complex)

    baseline = evaluate({})
    terms = [
        (name, evaluate({name: 1}) - baseline)
        for name in names
        if name not in fixed_parameters
    ]
    terms.append((sympy.S.One, baseline))
    return terms


def builder_to_model(syst, momenta=None, real_space=True, params=None):
    """Convert a Builder into a qsymm BlochModel."""
    if not isinstance(syst, builder_module.Builder):
        raise TypeError("expected a Kwant Builder")
    sites = list(syst.sites())
    if not sites:
        raise ValueError("cannot convert an empty Builder")
    fixed_parameters = {} if params is None else dict(params)
    periods = np.asarray(syst.symmetry.periods, dtype=float)
    periodic_dimension = len(periods)
    if momenta is None:
        momenta = ["k_x", "k_y", "k_z"][:periodic_dimension]
    momenta = tuple(momenta)
    if len(momenta) != periodic_dimension:
        raise ValueError("the number of momentum names must match the symmetry")

    spatial_dimension = len(np.asarray(sites[0].pos))
    if periodic_dimension == 0:
        projection = np.empty((0, spatial_dimension))
    elif periodic_dimension < spatial_dimension:
        orthogonal, triangular = linalg.qr(periods.T, mode="economic")
        orientation = np.diag(np.sign(np.diag(triangular)))
        projection = orientation @ orthogonal.T
    else:
        projection = np.eye(periodic_dimension)

    slices, size = _orbital_layout(syst)
    canonical = syst.symmetry.to_fd
    result = BlochModel(
        {},
        momenta=momenta,
        shape=(size, size),
        format=np.ndarray,
    )

    def add_term(displacement, coefficient, matrix):
        nonlocal result
        if allclose(matrix, 0):
            return
        key = BlochCoeff(np.asarray(displacement, dtype=float), _qsymm.sympify(coefficient))
        result += BlochModel({key: matrix}, momenta=momenta)

    for site, value in syst.site_value_pairs():
        rows = slices[canonical(site)]
        terms = (
            _linear_parameter_terms(value, (site,), fixed_parameters)
            if callable(value)
            else [(sympy.S.One, value)]
        )
        for coefficient, block in terms:
            add_term(
                np.zeros(periodic_dimension),
                coefficient,
                _matrix_with_block(size, rows, rows, block),
            )

    for (first, second), value in syst.hopping_value_pairs():
        rows = slices[canonical(first)]
        columns = slices[canonical(second)]
        displacement = (
            projection @ (np.asarray(second.pos) - np.asarray(first.pos))
            if real_space
            else np.asarray(syst.symmetry.which(second), dtype=float)
        )
        terms = (
            _linear_parameter_terms(value, (first, second), fixed_parameters)
            if callable(value)
            else [(sympy.S.One, value)]
        )
        for coefficient, block in terms:
            matrix = _matrix_with_block(size, rows, columns, block)
            direct = BlochModel(
                {BlochCoeff(displacement, _qsymm.sympify(coefficient)): matrix},
                momenta=momenta,
            )
            result += direct + direct.T().conj()
    return result


def _as_integer_vector(vector):
    rounded = np.rint(vector).astype(int)
    return tuple(rounded) if allclose(vector, rounded) else None


def model_to_builder(model, norbs, lat_vecs, atom_coords, *, coeffs=None):
    """Convert qsymm Model objects or Hamiltonian families into a Builder."""
    if isinstance(model, Model):
        bloch_model = model if isinstance(model, BlochModel) else BlochModel(model)
    else:
        bloch_model = BlochModel(
            hamiltonian_from_family(
                model,
                coeffs=coeffs,
                nsimplify=False,
                tosympy=False,
            )
        )
    lattice_vectors = np.asarray(lat_vecs, dtype=float)
    if lattice_vectors.ndim != 2:
        raise ValueError("lat_vecs must be a two-dimensional array")
    if len(bloch_model.momenta) != len(lattice_vectors):
        raise ValueError("lattice dimension and momentum count do not match")
    if not isinstance(norbs, (OrderedDict, list, tuple)):
        raise ValueError("norbs must be an ordered mapping or ordered pairs")
    orbital_counts = OrderedDict(norbs)
    atoms = tuple(orbital_counts)
    if len(atoms) != len(atom_coords):
        raise ValueError("atom_coords and norbs must describe the same atoms")

    slices = {}
    cursor = 0
    for atom, count in orbital_counts.items():
        slices[atom] = slice(cursor, cursor + int(count))
        cursor += int(count)
    if tuple(bloch_model.shape) != (cursor, cursor):
        raise ValueError("model matrix shape does not match norbs")

    crystal = lattice.general(
        lattice_vectors,
        atom_coords,
        norbs=tuple(orbital_counts.values()),
    )
    sublattices = dict(zip(atoms, crystal.sublattices, strict=True))
    coordinates = {
        atom: np.asarray(position, dtype=float)
        for atom, position in zip(atoms, atom_coords, strict=True)
    }
    onsite_terms = defaultdict(lambda: 0)
    hopping_terms = defaultdict(lambda: 0)
    zero_translation = (0,) * len(lattice_vectors)

    for key, matrix in bloch_model.items():
        displacement, coefficient = key
        displacement = np.asarray(displacement, dtype=float)
        for first_atom, second_atom in itertools.product(atoms, repeat=2):
            block = np.asarray(matrix[slices[first_atom], slices[second_atom]])
            if allclose(block, 0):
                continue
            term = Model(
                {coefficient: block},
                momenta=bloch_model.momenta,
            )
            zero_displacement = allclose(displacement, 0)
            if zero_displacement:
                if first_atom == second_atom:
                    onsite_terms[first_atom] += term
                    continue
                if not allclose(coordinates[first_atom], coordinates[second_atom]):
                    raise ValueError("site positions are incompatible with the model")
            real_displacement = (
                displacement
                + coordinates[first_atom]
                - coordinates[second_atom]
            )
            lattice_displacement = np.linalg.lstsq(
                lattice_vectors.T,
                real_displacement,
                rcond=None,
            )[0]
            integer_displacement = _as_integer_vector(lattice_displacement)
            if integer_displacement is None:
                raise RuntimeError(
                    "a nonzero model block does not match a lattice translation"
                )
            if all(value == 0 for value in integer_displacement):
                integer_displacement = zero_translation
            hopping = builder_module.HoppingKind(
                tuple(-value for value in integer_displacement),
                sublattices[first_atom],
                sublattices[second_atom],
            )
            hopping_terms[hopping] += term

    zero_tag = (0,) * len(lattice_vectors)
    for atom in atoms:
        if atom not in onsite_terms:
            count = int(orbital_counts[atom])
            onsite_terms[atom] = Model(
                {sympy.S.One: np.zeros((count, count))},
                momenta=bloch_model.momenta,
            )
    system = builder_module.Builder(
        lattice.TranslationalSymmetry(*lattice_vectors)
    )
    for atom, onsite in onsite_terms.items():
        system[sublattices[atom](*zero_tag)] = onsite.lambdify(onsite=True)
    for hopping, value in hopping_terms.items():
        system[hopping] = value.lambdify(hopping=True)
    return system


def _get_builder_symmetries(system):
    """Translate Builder-declared onsite symmetries into qsymm objects."""
    if not isinstance(system, builder_module.Builder):
        raise TypeError("expected a Kwant Builder")
    dimension = len(np.asarray(system.symmetry.periods))
    identity = np.eye(dimension)
    result = {}
    if system.time_reversal is not None:
        result["time_reversal"] = PointGroupElement(
            identity, True, False, system.time_reversal
        )
    if system.particle_hole is not None:
        result["particle_hole"] = PointGroupElement(
            identity, True, True, system.particle_hole
        )
    if system.chiral is not None:
        result["chiral"] = PointGroupElement(
            identity, False, True, system.chiral
        )
    if system.conservation_law is not None:
        result["conservation_law"] = ContinuousGroupGenerator(
            R=None,
            U=system.conservation_law,
        )
    return result


def find_builder_symmetries(
    builder,
    momenta=None,
    params=None,
    spatial_symmetries=True,
    prettify=True,
    sparse=None,
):
    """Find spatial and onsite symmetries of a Builder with qsymm."""
    model = builder_to_model(
        builder,
        momenta=momenta,
        real_space=spatial_symmetries,
        params=params,
    )
    if sparse is None:
        sparse = next(iter(model.values())).shape[0] > 20
    dimension = len(np.asarray(builder.symmetry.periods))
    if spatial_symmetries:
        candidates = bravais_point_group(
            builder.symmetry.periods,
            tr=True,
            ph=True,
            generators=False,
            verbose=False,
        )
    else:
        candidates = [
            PointGroupElement(np.eye(dimension), True, False, None),
            PointGroupElement(np.eye(dimension), True, True, None),
            PointGroupElement(np.eye(dimension), False, True, None),
        ]
    discrete, continuous = _qsymm.symmetries(
        model,
        candidates,
        prettify=prettify,
        continuous_rotations=False,
        sparse_linalg=sparse,
    )
    return [*discrete, *continuous]


__all__ = [
    "_get_builder_symmetries",
    "builder_to_model",
    "find_builder_symmetries",
    "model_to_builder",
]
