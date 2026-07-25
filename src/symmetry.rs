//! Discrete symmetries and conservation-law subspaces.

use std::f64::consts::PI;
use std::fmt;

use nalgebra::DMatrix;

use crate::decomposition::schur;
use crate::{Complex64, ComplexMatrix};

const UNITARY_TOLERANCE: f64 = 1.0e-10;
const CANONICAL_TOLERANCE: f64 = 1.0e-7;
const VALIDATION_TOLERANCE: f64 = 1.0e-8;

/// A declared conservation law or discrete symmetry violated by an operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymmetryViolation {
    ConservationLaw,
    TimeReversal,
    ParticleHole,
    Chiral,
}

impl SymmetryViolation {
    /// Human-readable compatibility label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ConservationLaw => "Conservation law",
            Self::TimeReversal => "Time reversal",
            Self::ParticleHole => "Particle-hole",
            Self::Chiral => "Chiral",
        }
    }
}

/// Invalid symmetry declarations or validation inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SymmetryError {
    EmptyProjectors,
    EmptyWaveFunctionBasis,
    InconsistentDimensions,
    ProjectorsNotComplete,
    ProductNotIdentity,
    OperatorNotUnitary { name: &'static str },
    InvalidOperatorSquare { name: &'static str },
    NonCanonicalProjectors { name: &'static str },
    WaveFunctionBasisNotClosed,
    OddParticleHoleSubspace,
    BasisConstructionFailed,
    ValidationMatrixTooWide,
    MatrixConstruction(String),
}

impl fmt::Display for SymmetryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyProjectors => write!(formatter, "projector list cannot be empty"),
            Self::EmptyWaveFunctionBasis => {
                write!(formatter, "wave-function basis cannot be empty")
            }
            Self::InconsistentDimensions => {
                write!(
                    formatter,
                    "symmetry and projector dimensions are inconsistent"
                )
            }
            Self::ProjectorsNotComplete => {
                write!(formatter, "projectors do not resolve the identity")
            }
            Self::ProductNotIdentity => {
                write!(
                    formatter,
                    "the product of all three symmetries is not identity"
                )
            }
            Self::OperatorNotUnitary { name } => {
                write!(formatter, "{name} symmetry is not unitary")
            }
            Self::InvalidOperatorSquare { name } => {
                write!(formatter, "{name} symmetry has an invalid square")
            }
            Self::NonCanonicalProjectors { name } => {
                write!(
                    formatter,
                    "{name} symmetry is not canonical in the projector basis"
                )
            }
            Self::WaveFunctionBasisNotClosed => write!(
                formatter,
                "wave functions are not orthonormal or not closed under particle-hole symmetry"
            ),
            Self::OddParticleHoleSubspace => write!(
                formatter,
                "a particle-hole subspace with square -1 must have even dimension"
            ),
            Self::BasisConstructionFailed => {
                write!(
                    formatter,
                    "particle-hole-symmetric basis construction failed"
                )
            }
            Self::ValidationMatrixTooWide => {
                write!(formatter, "a validation matrix cannot be wider than square")
            }
            Self::MatrixConstruction(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SymmetryError {}

/// A particle-hole-adapted orthonormal basis and its stable partner ordering.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleHoleBasis {
    wave_functions: ComplexMatrix,
    ordering: Vec<usize>,
}

impl ParticleHoleBasis {
    /// Wave functions stored as orthonormal columns.
    #[must_use]
    pub const fn wave_functions(&self) -> &ComplexMatrix {
        &self.wave_functions
    }

    /// Sorting keys that keep square-minus-one particle-hole partners adjacent.
    #[must_use]
    pub fn ordering(&self) -> &[usize] {
        &self.ordering
    }
}

/// Construct an orthonormal basis adapted to an antiunitary particle-hole
/// symmetry inside a closed wave-function subspace.
///
/// For `P² = +1`, every returned column is invariant under `P`. For
/// `P² = -1`, returned columns form adjacent pairs `[ψ, Pψ]`.
pub fn particle_hole_symmetric_basis(
    wave_functions: &ComplexMatrix,
    particle_hole: &ComplexMatrix,
) -> Result<ParticleHoleBasis, SymmetryError> {
    let ambient_dimension = wave_functions.rows();
    let subspace_dimension = wave_functions.columns();
    if subspace_dimension == 0 {
        return Err(SymmetryError::EmptyWaveFunctionBasis);
    }
    if particle_hole.shape() != (ambient_dimension, ambient_dimension) {
        return Err(SymmetryError::InconsistentDimensions);
    }

    let wave_functions = to_dense(wave_functions);
    let particle_hole = to_dense(particle_hole);
    let restricted = wave_functions.adjoint() * &particle_hole * conjugate(&wave_functions);
    if !almost_identity_with_tolerance(
        &(&restricted * restricted.adjoint()),
        1.0,
        VALIDATION_TOLERANCE,
    ) {
        return Err(SymmetryError::WaveFunctionBasisNotClosed);
    }

    let square = &restricted * conjugate(&restricted);
    let square_sign = if almost_identity_with_tolerance(&square, 1.0, VALIDATION_TOLERANCE) {
        1
    } else if almost_identity_with_tolerance(&square, -1.0, VALIDATION_TOLERANCE) {
        -1
    } else {
        return Err(SymmetryError::InvalidOperatorSquare {
            name: "Particle-hole",
        });
    };

    let (adapted, ordering) = if square_sign == 1 {
        if maximum_entry_norm(&(&restricted - restricted.transpose())) > VALIDATION_TOLERANCE {
            return Err(SymmetryError::InvalidOperatorSquare {
                name: "Particle-hole",
            });
        }
        let restricted_matrix = matrix_from_dense(restricted)?;
        let decomposition =
            schur(&restricted_matrix).map_err(|_| SymmetryError::BasisConstructionFailed)?;
        let mut phases = decomposition
            .eigenvalues()
            .iter()
            .map(|value| value.arg())
            .collect::<Vec<_>>();
        phases.sort_by(f64::total_cmp);
        let (gap_index, gap_size) = phases
            .iter()
            .enumerate()
            .map(|(index, &phase)| {
                let next = if index + 1 < phases.len() {
                    phases[index + 1]
                } else {
                    phases[0] + 2.0 * PI
                };
                (index, next - phase)
            })
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .ok_or(SymmetryError::BasisConstructionFailed)?;
        let shift = -PI - (phases[gap_index] + gap_size / 2.0);
        let phase_shift = Complex64::from_polar(1.0, shift);
        let root_unshift = Complex64::from_polar(1.0, -shift / 2.0);
        let roots = decomposition
            .eigenvalues()
            .iter()
            .map(|&value| (value * phase_shift).sqrt() * root_unshift)
            .collect::<Vec<_>>();
        let vectors = to_dense(decomposition.vectors());
        let root_diagonal = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(roots));
        let square_root = &vectors * root_diagonal * vectors.adjoint();
        (&wave_functions * square_root, vec![0; subspace_dimension])
    } else {
        if subspace_dimension % 2 != 0 {
            return Err(SymmetryError::OddParticleHoleSubspace);
        }
        let mut basis = Vec::<Vec<Complex64>>::with_capacity(subspace_dimension);
        while basis.len() < subspace_dimension {
            let candidate = (0..subspace_dimension)
                .map(|column| {
                    let source = wave_functions.column(column);
                    let mut residual = source.iter().copied().collect::<Vec<_>>();
                    orthogonalize(&mut residual, &basis);
                    let norm = vector_norm(&residual);
                    (residual, norm)
                })
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .ok_or(SymmetryError::BasisConstructionFailed)?;
            if candidate.1 <= VALIDATION_TOLERANCE {
                return Err(SymmetryError::BasisConstructionFailed);
            }
            let wave_function = normalized(candidate.0, candidate.1);
            let conjugated = DMatrix::from_column_slice(
                ambient_dimension,
                1,
                &wave_function
                    .iter()
                    .map(|value| value.conj())
                    .collect::<Vec<_>>(),
            );
            let partner_matrix = &particle_hole * conjugated;
            let partner = partner_matrix.column(0).iter().copied().collect::<Vec<_>>();
            if basis
                .iter()
                .chain(std::iter::once(&wave_function))
                .any(|existing| inner_product(existing, &partner).norm() > VALIDATION_TOLERANCE)
            {
                return Err(SymmetryError::BasisConstructionFailed);
            }
            let partner_norm = vector_norm(&partner);
            if (partner_norm - 1.0).abs() > VALIDATION_TOLERANCE {
                return Err(SymmetryError::BasisConstructionFailed);
            }
            basis.push(wave_function);
            basis.push(normalized(partner, partner_norm));
        }
        let adapted = DMatrix::from_fn(ambient_dimension, subspace_dimension, |row, column| {
            basis[column][row]
        });
        (adapted, (0..subspace_dimension).collect())
    };

    let gram = adapted.adjoint() * &adapted;
    if !almost_identity_with_tolerance(&gram, 1.0, VALIDATION_TOLERANCE) {
        return Err(SymmetryError::BasisConstructionFailed);
    }
    let transformed = &particle_hole * conjugate(&adapted);
    let relation_is_valid = if square_sign == 1 {
        maximum_entry_norm(&(&adapted - transformed)) <= VALIDATION_TOLERANCE
    } else {
        (0..subspace_dimension).step_by(2).all(|column| {
            (0..ambient_dimension).all(|row| {
                (adapted[(row, column + 1)] - transformed[(row, column)]).norm()
                    <= VALIDATION_TOLERANCE
            })
        })
    };
    if !relation_is_valid {
        return Err(SymmetryError::BasisConstructionFailed);
    }

    Ok(ParticleHoleBasis {
        wave_functions: matrix_from_dense(adapted)?,
        ordering,
    })
}

/// A validated collection of conservation-law projectors and symmetries.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteSymmetry {
    projectors: Option<Vec<ComplexMatrix>>,
    time_reversal: Option<ComplexMatrix>,
    particle_hole: Option<ComplexMatrix>,
    chiral: Option<ComplexMatrix>,
}

impl DiscreteSymmetry {
    /// Validates declarations, removes empty projector columns, and derives a
    /// missing third symmetry when two are supplied.
    pub fn new(
        projectors: Option<Vec<ComplexMatrix>>,
        time_reversal: Option<ComplexMatrix>,
        particle_hole: Option<ComplexMatrix>,
        chiral: Option<ComplexMatrix>,
    ) -> Result<Self, SymmetryError> {
        let mut projectors = projectors.map(trim_projectors).transpose()?;
        let mut time_reversal = time_reversal.map(|matrix| to_dense(&matrix));
        let mut particle_hole = particle_hole.map(|matrix| to_dense(&matrix));
        let mut chiral = chiral.map(|matrix| to_dense(&matrix));

        let declared_dimension = [
            time_reversal.as_ref(),
            particle_hole.as_ref(),
            chiral.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(DMatrix::nrows)
        .next()
        .or_else(|| {
            projectors
                .as_ref()
                .and_then(|values| values.first())
                .map(ComplexMatrix::rows)
        });
        for operator in [
            time_reversal.as_ref(),
            particle_hole.as_ref(),
            chiral.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if operator.nrows() != operator.ncols() || Some(operator.nrows()) != declared_dimension
            {
                return Err(SymmetryError::InconsistentDimensions);
            }
        }
        if let Some(values) = &projectors {
            if values
                .iter()
                .any(|projector| Some(projector.rows()) != declared_dimension)
            {
                return Err(SymmetryError::InconsistentDimensions);
            }
        }

        if let (Some(time), Some(particle), Some(chiral_operator)) =
            (&time_reversal, &particle_hole, &chiral)
        {
            let product = chiral_operator * particle * conjugate(time);
            if !almost_identity(&product, 1.0) {
                return Err(SymmetryError::ProductNotIdentity);
            }
        }

        match (&time_reversal, &particle_hole, &chiral) {
            (Some(time), Some(particle), None) => {
                chiral = Some(particle * conjugate(time));
            }
            (None, Some(particle), Some(chiral_operator)) => {
                time_reversal =
                    Some(particle * conjugate(particle) * particle * conjugate(chiral_operator));
            }
            (Some(time), None, Some(chiral_operator)) => {
                particle_hole = Some(time * conjugate(time) * chiral_operator * time);
            }
            _ => {}
        }

        let symmetries = [
            (time_reversal.as_ref(), true, "Time reversal"),
            (particle_hole.as_ref(), true, "Particle-hole"),
            (chiral.as_ref(), false, "Chiral"),
        ];
        for (operator, antiunitary, name) in symmetries {
            let Some(operator) = operator else {
                continue;
            };
            if !almost_identity(&(operator.adjoint() * operator), 1.0) {
                return Err(SymmetryError::OperatorNotUnitary { name });
            }
            let square = if antiunitary {
                operator * conjugate(operator)
            } else {
                operator * operator
            };
            if !almost_identity(&square, 1.0) && !(antiunitary && almost_identity(&square, -1.0)) {
                return Err(SymmetryError::InvalidOperatorSquare { name });
            }
        }

        if let Some(values) = &projectors {
            let dimension = values[0].rows();
            let mut resolution = DMatrix::zeros(dimension, dimension);
            for projector in values {
                let projector = to_dense(projector);
                resolution += &projector * projector.adjoint();
            }
            if !almost_identity(&resolution, 1.0) {
                return Err(SymmetryError::ProjectorsNotComplete);
            }

            for (operator, antiunitary, name) in symmetries {
                let Some(operator) = operator else {
                    continue;
                };
                for target in values {
                    let target = to_dense(target);
                    let mut nonzero_blocks = 0;
                    for source in values {
                        let source = to_dense(source);
                        let source = if antiunitary {
                            conjugate(&source)
                        } else {
                            source
                        };
                        let block = target.adjoint() * operator * source;
                        if maximum_entry_norm(&block) > CANONICAL_TOLERANCE {
                            nonzero_blocks += 1;
                        }
                    }
                    if nonzero_blocks > 1 {
                        return Err(SymmetryError::NonCanonicalProjectors { name });
                    }
                }
            }
        }

        if let Some(values) = &mut projectors {
            for projector in values {
                *projector = matrix_from_dense(to_dense(projector))?;
            }
        }
        Ok(Self {
            projectors,
            time_reversal: time_reversal.map(matrix_from_dense).transpose()?,
            particle_hole: particle_hole.map(matrix_from_dense).transpose()?,
            chiral: chiral.map(matrix_from_dense).transpose()?,
        })
    }

    #[must_use]
    pub fn projectors(&self) -> Option<&[ComplexMatrix]> {
        self.projectors.as_deref()
    }

    #[must_use]
    pub fn time_reversal(&self) -> Option<&ComplexMatrix> {
        self.time_reversal.as_ref()
    }

    #[must_use]
    pub fn particle_hole(&self) -> Option<&ComplexMatrix> {
        self.particle_hole.as_ref()
    }

    #[must_use]
    pub fn chiral(&self) -> Option<&ComplexMatrix> {
        self.chiral.as_ref()
    }

    /// Validates a square matrix or a left-aligned rectangular hopping block.
    pub fn validate(
        &self,
        matrix: &ComplexMatrix,
    ) -> Result<Vec<SymmetryViolation>, SymmetryError> {
        if matrix.columns() > matrix.rows() {
            return Err(SymmetryError::ValidationMatrixTooWide);
        }
        let dimension = matrix.rows();
        if self
            .projectors
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| value.rows() != dimension))
            || [
                self.time_reversal.as_ref(),
                self.particle_hole.as_ref(),
                self.chiral.as_ref(),
            ]
            .into_iter()
            .flatten()
            .any(|value| value.rows() != dimension)
        {
            return Err(SymmetryError::InconsistentDimensions);
        }
        let mut padded = DMatrix::zeros(dimension, dimension);
        for row in 0..matrix.rows() {
            for column in 0..matrix.columns() {
                padded[(row, column)] = matrix.as_slice()[row * matrix.columns() + column];
            }
        }
        let mut violations = Vec::new();
        if let Some(projectors) = &self.projectors {
            if projectors.iter().any(|projector| {
                let projector = to_dense(projector);
                let full = &projector * projector.adjoint();
                frobenius_norm(&(&full * &padded - &padded * &full)) > VALIDATION_TOLERANCE
            }) {
                violations.push(SymmetryViolation::ConservationLaw);
            }
        }
        let checks = [
            (
                self.time_reversal.as_ref(),
                true,
                1.0,
                SymmetryViolation::TimeReversal,
            ),
            (
                self.particle_hole.as_ref(),
                true,
                -1.0,
                SymmetryViolation::ParticleHole,
            ),
            (self.chiral.as_ref(), false, -1.0, SymmetryViolation::Chiral),
        ];
        for (operator, antiunitary, sign, violation) in checks {
            let Some(operator) = operator else {
                continue;
            };
            let operator = to_dense(operator);
            let transformed = operator.adjoint() * &padded * operator;
            let expected = if antiunitary {
                conjugate(&padded) * Complex64::new(sign, 0.0)
            } else {
                &padded * Complex64::new(sign, 0.0)
            };
            if frobenius_norm(&(transformed - expected)) > VALIDATION_TOLERANCE {
                violations.push(violation);
            }
        }
        Ok(violations)
    }
}

fn trim_projectors(projectors: Vec<ComplexMatrix>) -> Result<Vec<ComplexMatrix>, SymmetryError> {
    if projectors.is_empty() {
        return Err(SymmetryError::EmptyProjectors);
    }
    projectors
        .into_iter()
        .map(|projector| {
            let kept = (0..projector.columns())
                .filter(|&column| {
                    (0..projector.rows())
                        .map(|row| projector.as_slice()[row * projector.columns() + column].norm())
                        .sum::<f64>()
                        > UNITARY_TOLERANCE
                })
                .collect::<Vec<_>>();
            let dense = DMatrix::from_fn(projector.rows(), kept.len(), |row, column| {
                projector.as_slice()[row * projector.columns() + kept[column]]
            });
            matrix_from_dense(dense)
        })
        .collect()
}

fn to_dense(matrix: &ComplexMatrix) -> DMatrix<Complex64> {
    DMatrix::from_row_slice(matrix.rows(), matrix.columns(), matrix.as_slice())
}

fn matrix_from_dense(matrix: DMatrix<Complex64>) -> Result<ComplexMatrix, SymmetryError> {
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    let matrix = &matrix;
    let values = (0..rows)
        .flat_map(|row| (0..columns).map(move |column| matrix[(row, column)]))
        .collect();
    ComplexMatrix::new(rows, columns, values)
        .map_err(|error| SymmetryError::MatrixConstruction(error.to_string()))
}

fn conjugate(matrix: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    matrix.map(|value| value.conj())
}

fn almost_identity(matrix: &DMatrix<Complex64>, sign: f64) -> bool {
    almost_identity_with_tolerance(matrix, sign, UNITARY_TOLERANCE)
}

fn almost_identity_with_tolerance(matrix: &DMatrix<Complex64>, sign: f64, tolerance: f64) -> bool {
    if matrix.nrows() != matrix.ncols() {
        return false;
    }
    (0..matrix.nrows()).all(|row| {
        (0..matrix.ncols()).all(|column| {
            let expected = if row == column { sign } else { 0.0 };
            (matrix[(row, column)] - Complex64::new(expected, 0.0)).norm() < tolerance
        })
    })
}

fn inner_product(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| left.conj() * right)
        .sum()
}

fn orthogonalize(vector: &mut [Complex64], basis: &[Vec<Complex64>]) {
    for existing in basis {
        let overlap = inner_product(existing, vector);
        for (value, &basis_value) in vector.iter_mut().zip(existing) {
            *value -= basis_value * overlap;
        }
    }
}

fn vector_norm(vector: &[Complex64]) -> f64 {
    vector
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn normalized(mut vector: Vec<Complex64>, norm: f64) -> Vec<Complex64> {
    for value in &mut vector {
        *value /= norm;
    }
    vector
}

fn frobenius_norm(matrix: &DMatrix<Complex64>) -> f64 {
    matrix
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn maximum_entry_norm(matrix: &DMatrix<Complex64>) -> f64 {
    matrix.iter().map(|value| value.norm()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, columns: usize, values: &[f64]) -> ComplexMatrix {
        ComplexMatrix::new(
            rows,
            columns,
            values
                .iter()
                .map(|&value| Complex64::new(value, 0.0))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn complementary_projectors_detect_mixing() {
        let symmetry = DiscreteSymmetry::new(
            Some(vec![matrix(2, 1, &[1.0, 0.0]), matrix(2, 1, &[0.0, 1.0])]),
            None,
            None,
            None,
        )
        .unwrap();
        assert!(symmetry
            .validate(&ComplexMatrix::identity(2))
            .unwrap()
            .is_empty());
        assert_eq!(
            symmetry
                .validate(&matrix(2, 2, &[0.0, 1.0, 1.0, 0.0]))
                .unwrap(),
            vec![SymmetryViolation::ConservationLaw]
        );
    }

    #[test]
    fn two_antiunitary_symmetries_determine_chiral_symmetry() {
        let identity = ComplexMatrix::identity(2);
        let particle = matrix(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let symmetry =
            DiscreteSymmetry::new(None, Some(identity), Some(particle.clone()), None).unwrap();
        assert_eq!(symmetry.chiral(), Some(&particle));
    }

    #[test]
    fn square_plus_one_basis_is_particle_hole_invariant() {
        let inverse_sqrt_two = 2.0_f64.sqrt().recip();
        let wave_functions = ComplexMatrix::new(
            2,
            2,
            vec![
                Complex64::new(inverse_sqrt_two, 0.0),
                Complex64::new(0.0, inverse_sqrt_two),
                Complex64::new(0.0, inverse_sqrt_two),
                Complex64::new(inverse_sqrt_two, 0.0),
            ],
        )
        .unwrap();
        let basis =
            particle_hole_symmetric_basis(&wave_functions, &ComplexMatrix::identity(2)).unwrap();
        assert_eq!(basis.ordering(), &[0, 0]);
        let adapted = to_dense(basis.wave_functions());
        assert!(almost_identity_with_tolerance(
            &(adapted.adjoint() * &adapted),
            1.0,
            VALIDATION_TOLERANCE,
        ));
        assert!(maximum_entry_norm(&(&adapted - conjugate(&adapted))) <= VALIDATION_TOLERANCE);
    }

    #[test]
    fn square_minus_one_basis_forms_adjacent_kramers_pairs() {
        let particle_hole = matrix(
            4,
            4,
            &[
                0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0,
            ],
        );
        let basis =
            particle_hole_symmetric_basis(&ComplexMatrix::identity(4), &particle_hole).unwrap();
        assert_eq!(basis.ordering(), &[0, 1, 2, 3]);
        let adapted = to_dense(basis.wave_functions());
        let transformed = to_dense(&particle_hole) * conjugate(&adapted);
        for column in (0..4).step_by(2) {
            for row in 0..4 {
                assert!(
                    (adapted[(row, column + 1)] - transformed[(row, column)]).norm()
                        <= VALIDATION_TOLERANCE
                );
            }
        }
        assert!(almost_identity_with_tolerance(
            &(adapted.adjoint() * adapted),
            1.0,
            VALIDATION_TOLERANCE,
        ));
    }
}
