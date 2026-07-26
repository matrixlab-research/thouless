//! Embedded Bravais geometry, neighbor shells, and translation domains.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use nalgebra::{DMatrix, DVector};

use crate::lattice_reduction::{closest_lattice_vectors, LatticeReductionError};

/// A Bravais lattice embedded in Cartesian space with one or more basis sites.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddedLattice {
    primitive_vectors: Vec<Vec<f64>>,
    basis_offsets: Vec<Vec<f64>>,
}

impl EmbeddedLattice {
    /// Validate and construct an embedded lattice.
    ///
    /// Primitive vectors are rows and may span a lower-dimensional subspace
    /// of the ambient Cartesian space.
    pub fn new(
        primitive_vectors: Vec<Vec<f64>>,
        basis_offsets: Vec<Vec<f64>>,
    ) -> Result<Self, LatticeGeometryError> {
        let ambient_dimension = validate_real_basis(&primitive_vectors)?;
        if basis_offsets.is_empty() {
            return Err(LatticeGeometryError::EmptyBasis);
        }
        if basis_offsets
            .iter()
            .any(|offset| offset.len() != ambient_dimension)
        {
            return Err(LatticeGeometryError::InvalidBasisDimension {
                expected: ambient_dimension,
            });
        }
        if basis_offsets
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(LatticeGeometryError::NonFiniteValue);
        }
        Ok(Self {
            primitive_vectors,
            basis_offsets,
        })
    }

    /// Number of independent lattice coordinates.
    #[must_use]
    pub fn lattice_dimension(&self) -> usize {
        self.primitive_vectors.len()
    }

    /// Number of Cartesian coordinates.
    #[must_use]
    pub fn ambient_dimension(&self) -> usize {
        self.primitive_vectors[0].len()
    }

    /// Primitive vectors as Cartesian rows.
    #[must_use]
    pub fn primitive_vectors(&self) -> &[Vec<f64>] {
        &self.primitive_vectors
    }

    /// Cartesian positions of the basis sites in the home cell.
    #[must_use]
    pub fn basis_offsets(&self) -> &[Vec<f64>] {
        &self.basis_offsets
    }

    /// Cartesian lattice vector represented by an integer tag.
    pub fn vector(&self, tag: &[i64]) -> Result<Vec<f64>, LatticeGeometryError> {
        if tag.len() != self.lattice_dimension() {
            return Err(LatticeGeometryError::InvalidTagDimension {
                expected: self.lattice_dimension(),
                actual: tag.len(),
            });
        }
        Ok((0..self.ambient_dimension())
            .map(|component| {
                tag.iter()
                    .zip(&self.primitive_vectors)
                    .map(|(&coefficient, vector)| coefficient as f64 * vector[component])
                    .sum()
            })
            .collect())
    }

    /// Cartesian position of a basis site in an integer-tagged cell.
    pub fn position(
        &self,
        basis_site: usize,
        tag: &[i64],
    ) -> Result<Vec<f64>, LatticeGeometryError> {
        let offset = self.basis_offsets.get(basis_site).ok_or(
            LatticeGeometryError::BasisSiteOutOfBounds {
                site: basis_site,
                site_count: self.basis_offsets.len(),
            },
        )?;
        let mut position = self.vector(tag)?;
        for (component, offset) in position.iter_mut().zip(offset) {
            *component += offset;
        }
        Ok(position)
    }

    /// Enumerate one global distance shell of basis-to-basis lattice relations.
    ///
    /// Order zero is the zero-distance onsite shell. For a relation between
    /// the same basis site, only one of a displacement and its negative is
    /// retained. Enumeration uses proven nearest-lattice-vector bounds rather
    /// than a fixed integer search box.
    pub fn neighbor_shell(
        &self,
        order: usize,
        relative_tolerance: f64,
    ) -> Result<Vec<NeighborRelation>, LatticeGeometryError> {
        if !relative_tolerance.is_finite() || relative_tolerance < 0.0 {
            return Err(LatticeGeometryError::InvalidTolerance);
        }
        let required_local_groups = order
            .checked_add(2)
            .ok_or(LatticeGeometryError::DimensionOverflow)?;
        let length_scale = self
            .primitive_vectors
            .iter()
            .map(|vector| squared_norm(vector).sqrt())
            .fold(f64::INFINITY, f64::min);
        let absolute_tolerance = relative_tolerance * length_scale;
        let mut candidates = Vec::new();
        for first in 0..self.basis_offsets.len() {
            for second in first..self.basis_offsets.len() {
                let target = self.basis_offsets[second]
                    .iter()
                    .zip(&self.basis_offsets[first])
                    .map(|(second, first)| second - first)
                    .collect::<Vec<_>>();
                let mut neighbor_count = required_local_groups.max(8);
                loop {
                    let displacements = closest_lattice_vectors(
                        &target,
                        &self.primitive_vectors,
                        neighbor_count,
                        false,
                        0.0,
                    )?;
                    let mut local = displacements
                        .into_iter()
                        .map(|displacement| {
                            let vector = self.vector(&displacement)?;
                            let distance = vector
                                .iter()
                                .zip(&self.basis_offsets[first])
                                .zip(&self.basis_offsets[second])
                                .map(|((translation, first), second)| {
                                    (translation + first - second).powi(2)
                                })
                                .sum::<f64>()
                                .sqrt();
                            Ok((distance, displacement, first, second))
                        })
                        .collect::<Result<Vec<_>, LatticeGeometryError>>()?;
                    local.sort_by(|left, right| {
                        left.0
                            .total_cmp(&right.0)
                            .then_with(|| left.1.cmp(&right.1))
                    });
                    let mut local_groups = 0usize;
                    let mut previous = None;
                    for (distance, _, _, _) in &local {
                        if previous.map_or(true, |previous: f64| {
                            (distance - previous).abs() > absolute_tolerance
                        }) {
                            local_groups += 1;
                            previous = Some(*distance);
                        }
                    }
                    if local_groups >= required_local_groups {
                        candidates.extend(local);
                        break;
                    }
                    neighbor_count = neighbor_count
                        .checked_mul(2)
                        .ok_or(LatticeGeometryError::DimensionOverflow)?;
                }
            }
        }
        let mut distances = candidates
            .iter()
            .map(|candidate| candidate.0)
            .collect::<Vec<_>>();
        distances.sort_by(f64::total_cmp);
        let mut shells = Vec::new();
        for distance in distances {
            if shells.last().map_or(true, |previous: &f64| {
                (distance - previous).abs() > absolute_tolerance
            }) {
                shells.push(distance);
            }
        }
        let Some(&target_distance) = shells.get(order) else {
            return Ok(Vec::new());
        };

        let mut unique = BTreeSet::new();
        for (distance, mut displacement, first, second) in candidates {
            if (distance - target_distance).abs() > absolute_tolerance {
                continue;
            }
            if first == second {
                let opposite = displacement.iter().map(|value| -*value).collect::<Vec<_>>();
                if opposite > displacement {
                    displacement = opposite;
                }
            }
            unique.insert((first, second, displacement));
        }
        Ok(unique
            .into_iter()
            .map(
                |(first_basis_site, second_basis_site, displacement)| NeighborRelation {
                    displacement,
                    first_basis_site,
                    second_basis_site,
                },
            )
            .collect())
    }

    /// Construct exact integer data for a Cartesian translation group.
    pub fn translation_domain(
        &self,
        cartesian_periods: &[Vec<f64>],
        other_vectors: &[Vec<i64>],
        tolerance: f64,
    ) -> Result<TranslationFundamentalDomain, LatticeGeometryError> {
        if cartesian_periods.is_empty() {
            return Err(LatticeGeometryError::EmptyTranslationGroup);
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(LatticeGeometryError::InvalidTolerance);
        }
        if cartesian_periods.iter().any(|period| {
            period.len() != self.ambient_dimension()
                || period.iter().any(|value| !value.is_finite())
        }) {
            return Err(LatticeGeometryError::InvalidPeriodDimension {
                expected: self.ambient_dimension(),
            });
        }
        if cartesian_periods.len() > self.lattice_dimension() {
            return Err(LatticeGeometryError::DependentTranslations);
        }

        let primitive = DMatrix::from_fn(
            self.lattice_dimension(),
            self.ambient_dimension(),
            |row, column| self.primitive_vectors[row][column],
        );
        let gram = &primitive * primitive.transpose();
        let inverse = gram
            .try_inverse()
            .ok_or(LatticeGeometryError::DependentPrimitiveVectors)?;
        let mut period_vectors = Vec::with_capacity(cartesian_periods.len());
        for (period_index, period) in cartesian_periods.iter().enumerate() {
            let period_column = DVector::from_row_slice(period);
            let coordinates = &inverse * &primitive * &period_column;
            let integer = coordinates
                .iter()
                .map(|value| rounded_i64(*value))
                .collect::<Result<Vec<_>, _>>()?;
            let reconstructed = self.vector(&integer)?;
            let commensurate =
                coordinates.iter().zip(&integer).all(|(value, integer)| {
                    approximately_equal(*value, *integer as f64, tolerance)
                }) && reconstructed
                    .iter()
                    .zip(period)
                    .all(|(actual, expected)| approximately_equal(*actual, *expected, tolerance));
            if !commensurate {
                return Err(LatticeGeometryError::IncommensuratePeriod {
                    period: period_index,
                });
            }
            period_vectors.push(integer);
        }

        let mut columns = period_vectors.clone();
        for vector in other_vectors {
            if vector.len() != self.lattice_dimension() {
                return Err(LatticeGeometryError::InvalidOtherVectorDimension {
                    expected: self.lattice_dimension(),
                });
            }
            columns.push(vector.clone());
        }
        if columns.len() > self.lattice_dimension() || !integer_columns_independent(&columns)? {
            return Err(LatticeGeometryError::DependentTranslations);
        }
        for axis in 0..self.lattice_dimension() {
            if columns.len() == self.lattice_dimension() {
                break;
            }
            let mut candidate = vec![0; self.lattice_dimension()];
            candidate[axis] = 1;
            let mut trial = columns.clone();
            trial.push(candidate.clone());
            if integer_columns_independent(&trial)? {
                columns.push(candidate);
            }
        }
        if columns.len() != self.lattice_dimension() {
            return Err(LatticeGeometryError::DependentTranslations);
        }

        let dimension = self.lattice_dimension();
        let basis = (0..dimension)
            .map(|row| {
                (0..dimension)
                    .map(|column| i128::from(columns[column][row]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let signed_determinant = exact_determinant(&basis)?;
        if signed_determinant == 0 {
            return Err(LatticeGeometryError::DependentTranslations);
        }
        let mut adjugate = exact_adjugate(&basis)?;
        let determinant = if signed_determinant < 0 {
            for value in adjugate.iter_mut().flatten() {
                *value = value
                    .checked_neg()
                    .ok_or(LatticeGeometryError::IntegerOverflow)?;
            }
            signed_determinant
                .checked_neg()
                .ok_or(LatticeGeometryError::IntegerOverflow)?
        } else {
            signed_determinant
        };
        let determinant =
            i64::try_from(determinant).map_err(|_| LatticeGeometryError::IntegerOverflow)?;
        let direction_count = period_vectors.len();
        let adjugate_rows = adjugate
            .into_iter()
            .take(direction_count)
            .map(|row| {
                row.into_iter()
                    .map(|value| {
                        i64::try_from(value).map_err(|_| LatticeGeometryError::IntegerOverflow)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TranslationFundamentalDomain {
            period_vectors,
            adjugate_rows,
            determinant,
        })
    }
}

/// One canonical hopping relation in a lattice distance shell.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NeighborRelation {
    displacement: Vec<i64>,
    first_basis_site: usize,
    second_basis_site: usize,
}

impl NeighborRelation {
    /// Cell displacement supplied to the hopping relation.
    #[must_use]
    pub fn displacement(&self) -> &[i64] {
        &self.displacement
    }

    /// First basis-site index.
    #[must_use]
    pub fn first_basis_site(&self) -> usize {
        self.first_basis_site
    }

    /// Second basis-site index.
    #[must_use]
    pub fn second_basis_site(&self) -> usize {
        self.second_basis_site
    }
}

/// Exact arithmetic for one translation-group fundamental domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranslationFundamentalDomain {
    period_vectors: Vec<Vec<i64>>,
    adjugate_rows: Vec<Vec<i64>>,
    determinant: i64,
}

impl TranslationFundamentalDomain {
    /// Integer lattice vectors corresponding to translation generators.
    #[must_use]
    pub fn period_vectors(&self) -> &[Vec<i64>] {
        &self.period_vectors
    }

    /// Rows of the completed-basis adjugate used to obtain group coordinates.
    #[must_use]
    pub fn adjugate_rows(&self) -> &[Vec<i64>] {
        &self.adjugate_rows
    }

    /// Positive volume of the completed integer basis.
    #[must_use]
    pub fn determinant(&self) -> i64 {
        self.determinant
    }

    /// Translation-group coordinates of an integer lattice tag.
    pub fn which(&self, tag: &[i64]) -> Result<Vec<i64>, LatticeGeometryError> {
        translation_coordinates(&self.adjugate_rows, self.determinant, tag)
    }

    /// Integer tag displacement generated by a group element.
    pub fn shift(&self, element: &[i64]) -> Result<Vec<i64>, LatticeGeometryError> {
        translation_shift(&self.period_vectors, element)
    }

    /// Canonical representative of an integer tag.
    pub fn to_fundamental_domain(&self, tag: &[i64]) -> Result<Vec<i64>, LatticeGeometryError> {
        let coordinates = self.which(tag)?;
        let inverse = coordinates
            .iter()
            .map(|value| {
                value
                    .checked_neg()
                    .ok_or(LatticeGeometryError::IntegerOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shift = self.shift(&inverse)?;
        tag.iter()
            .zip(shift)
            .map(|(&tag, shift)| {
                tag.checked_add(shift)
                    .ok_or(LatticeGeometryError::IntegerOverflow)
            })
            .collect()
    }
}

/// Compute exact group coordinates from a fundamental-domain adjugate.
pub fn translation_coordinates(
    adjugate_rows: &[Vec<i64>],
    determinant: i64,
    tag: &[i64],
) -> Result<Vec<i64>, LatticeGeometryError> {
    if determinant <= 0 {
        return Err(LatticeGeometryError::InvalidDeterminant);
    }
    if adjugate_rows.iter().any(|row| row.len() != tag.len()) {
        return Err(LatticeGeometryError::InvalidTagDimension {
            expected: adjugate_rows.first().map_or(tag.len(), Vec::len),
            actual: tag.len(),
        });
    }
    adjugate_rows
        .iter()
        .map(|row| {
            let numerator = checked_integer_dot(row, tag)?;
            let coordinate = numerator.div_euclid(i128::from(determinant));
            i64::try_from(coordinate).map_err(|_| LatticeGeometryError::IntegerOverflow)
        })
        .collect()
}

/// Compute an exact integer-tag displacement from translation generators.
pub fn translation_shift(
    period_vectors: &[Vec<i64>],
    element: &[i64],
) -> Result<Vec<i64>, LatticeGeometryError> {
    if element.len() != period_vectors.len() {
        return Err(LatticeGeometryError::InvalidGroupElementDimension {
            expected: period_vectors.len(),
            actual: element.len(),
        });
    }
    let dimension = period_vectors.first().map_or(0, Vec::len);
    if period_vectors
        .iter()
        .any(|period| period.len() != dimension)
    {
        return Err(LatticeGeometryError::InvalidPeriodDimension {
            expected: dimension,
        });
    }
    (0..dimension)
        .map(|axis| {
            let value = period_vectors.iter().zip(element).try_fold(
                0i128,
                |sum, (period, &coefficient)| {
                    let product = i128::from(period[axis])
                        .checked_mul(i128::from(coefficient))
                        .ok_or(LatticeGeometryError::IntegerOverflow)?;
                    sum.checked_add(product)
                        .ok_or(LatticeGeometryError::IntegerOverflow)
                },
            )?;
            i64::try_from(value).map_err(|_| LatticeGeometryError::IntegerOverflow)
        })
        .collect()
}

/// Return whether every `other_periods` row is an integer combination of
/// `group_periods`.
pub fn contains_translation_subgroup(
    group_periods: &[Vec<f64>],
    other_periods: &[Vec<f64>],
    tolerance: f64,
) -> Result<bool, LatticeGeometryError> {
    let ambient_dimension = validate_real_basis(group_periods)?;
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(LatticeGeometryError::InvalidTolerance);
    }
    if other_periods.iter().any(|period| {
        period.len() != ambient_dimension || period.iter().any(|value| !value.is_finite())
    }) {
        return Err(LatticeGeometryError::InvalidPeriodDimension {
            expected: ambient_dimension,
        });
    }
    let group = DMatrix::from_fn(group_periods.len(), ambient_dimension, |row, column| {
        group_periods[row][column]
    });
    let gram = &group * group.transpose();
    let inverse = gram
        .try_inverse()
        .ok_or(LatticeGeometryError::DependentTranslations)?;
    for period in other_periods {
        let period = DVector::from_row_slice(period);
        let coordinates = &inverse * &group * &period;
        let integers = coordinates
            .iter()
            .map(|value| rounded_i64(*value))
            .collect::<Result<Vec<_>, _>>()?;
        if !coordinates
            .iter()
            .zip(&integers)
            .all(|(value, integer)| approximately_equal(*value, *integer as f64, tolerance))
        {
            return Ok(false);
        }
        let reconstructed = (0..ambient_dimension)
            .map(|component| {
                integers
                    .iter()
                    .zip(group_periods)
                    .map(|(&coefficient, vector)| coefficient as f64 * vector[component])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        if !reconstructed
            .iter()
            .zip(period.iter())
            .all(|(actual, expected)| approximately_equal(*actual, *expected, tolerance))
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Invalid embedded-lattice geometry or exact integer-domain arithmetic.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LatticeGeometryError {
    EmptyPrimitiveVectors,
    EmptyBasis,
    RaggedPrimitiveVectors,
    TooManyPrimitiveVectors,
    DependentPrimitiveVectors,
    InvalidBasisDimension { expected: usize },
    NonFiniteValue,
    BasisSiteOutOfBounds { site: usize, site_count: usize },
    InvalidTagDimension { expected: usize, actual: usize },
    InvalidGroupElementDimension { expected: usize, actual: usize },
    InvalidPeriodDimension { expected: usize },
    InvalidOtherVectorDimension { expected: usize },
    InvalidDeterminant,
    EmptyTranslationGroup,
    DependentTranslations,
    IncommensuratePeriod { period: usize },
    InvalidTolerance,
    DimensionOverflow,
    IntegerOverflow,
    LatticeReduction(LatticeReductionError),
}

impl fmt::Display for LatticeGeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPrimitiveVectors => write!(formatter, "primitive vectors cannot be empty"),
            Self::EmptyBasis => write!(formatter, "basis offsets cannot be empty"),
            Self::RaggedPrimitiveVectors => {
                write!(
                    formatter,
                    "primitive vectors must have equal nonzero dimensions"
                )
            }
            Self::TooManyPrimitiveVectors => {
                write!(
                    formatter,
                    "there are more primitive vectors than Cartesian dimensions"
                )
            }
            Self::DependentPrimitiveVectors => {
                write!(formatter, "primitive vectors are linearly dependent")
            }
            Self::InvalidBasisDimension { expected } => {
                write!(
                    formatter,
                    "basis offsets must have {expected} Cartesian components"
                )
            }
            Self::NonFiniteValue => write!(formatter, "lattice geometry must be finite"),
            Self::BasisSiteOutOfBounds { site, site_count } => {
                write!(formatter, "basis site {site} is outside {site_count} sites")
            }
            Self::InvalidTagDimension { expected, actual } => {
                write!(
                    formatter,
                    "tag has {actual} components; expected {expected}"
                )
            }
            Self::InvalidGroupElementDimension { expected, actual } => write!(
                formatter,
                "group element has {actual} components; expected {expected}"
            ),
            Self::InvalidPeriodDimension { expected } => {
                write!(
                    formatter,
                    "translation periods must have {expected} components"
                )
            }
            Self::InvalidOtherVectorDimension { expected } => {
                write!(
                    formatter,
                    "other vectors must have {expected} integer components"
                )
            }
            Self::InvalidDeterminant => {
                write!(formatter, "fundamental-domain determinant must be positive")
            }
            Self::EmptyTranslationGroup => {
                write!(formatter, "at least one translation period is required")
            }
            Self::DependentTranslations => {
                write!(
                    formatter,
                    "translation and completion vectors must be independent"
                )
            }
            Self::IncommensuratePeriod { period } => {
                write!(
                    formatter,
                    "translation period {period} is not a lattice vector"
                )
            }
            Self::InvalidTolerance => write!(formatter, "tolerance must be finite and nonnegative"),
            Self::DimensionOverflow => write!(formatter, "lattice dimension overflowed"),
            Self::IntegerOverflow => write!(formatter, "exact lattice arithmetic overflowed"),
            Self::LatticeReduction(error) => error.fmt(formatter),
        }
    }
}

impl Error for LatticeGeometryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::LatticeReduction(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LatticeReductionError> for LatticeGeometryError {
    fn from(error: LatticeReductionError) -> Self {
        Self::LatticeReduction(error)
    }
}

fn validate_real_basis(basis: &[Vec<f64>]) -> Result<usize, LatticeGeometryError> {
    let ambient_dimension = basis
        .first()
        .map(Vec::len)
        .ok_or(LatticeGeometryError::EmptyPrimitiveVectors)?;
    if ambient_dimension == 0 || basis.iter().any(|vector| vector.len() != ambient_dimension) {
        return Err(LatticeGeometryError::RaggedPrimitiveVectors);
    }
    if basis.len() > ambient_dimension {
        return Err(LatticeGeometryError::TooManyPrimitiveVectors);
    }
    if basis.iter().flatten().any(|value| !value.is_finite()) {
        return Err(LatticeGeometryError::NonFiniteValue);
    }
    let matrix = DMatrix::from_fn(basis.len(), ambient_dimension, |row, column| {
        basis[row][column]
    });
    let singular_values = matrix.svd(false, false).singular_values;
    let maximum = singular_values.iter().copied().fold(0.0, f64::max);
    let tolerance = f64::EPSILON * basis.len().max(ambient_dimension) as f64 * maximum;
    if maximum == 0.0 || singular_values.iter().any(|value| *value <= tolerance) {
        return Err(LatticeGeometryError::DependentPrimitiveVectors);
    }
    Ok(ambient_dimension)
}

fn approximately_equal(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance * left.abs().max(right.abs()).max(1.0)
}

fn rounded_i64(value: f64) -> Result<i64, LatticeGeometryError> {
    let rounded = value.round_ties_even();
    if !rounded.is_finite() || rounded < i64::MIN as f64 || rounded >= -(i64::MIN as f64) {
        return Err(LatticeGeometryError::IntegerOverflow);
    }
    Ok(rounded as i64)
}

fn checked_integer_dot(left: &[i64], right: &[i64]) -> Result<i128, LatticeGeometryError> {
    left.iter()
        .zip(right)
        .try_fold(0i128, |sum, (&left, &right)| {
            let product = i128::from(left)
                .checked_mul(i128::from(right))
                .ok_or(LatticeGeometryError::IntegerOverflow)?;
            sum.checked_add(product)
                .ok_or(LatticeGeometryError::IntegerOverflow)
        })
}

fn integer_columns_independent(columns: &[Vec<i64>]) -> Result<bool, LatticeGeometryError> {
    if columns.is_empty() {
        return Ok(true);
    }
    let dimension = columns[0].len();
    if columns.iter().any(|column| column.len() != dimension) {
        return Err(LatticeGeometryError::InvalidOtherVectorDimension {
            expected: dimension,
        });
    }
    if columns.len() > dimension {
        return Ok(false);
    }
    let mut gram = vec![vec![0i128; columns.len()]; columns.len()];
    for row in 0..columns.len() {
        for column in 0..columns.len() {
            gram[row][column] = columns[row].iter().zip(&columns[column]).try_fold(
                0i128,
                |sum, (&left, &right)| {
                    let product = i128::from(left)
                        .checked_mul(i128::from(right))
                        .ok_or(LatticeGeometryError::IntegerOverflow)?;
                    sum.checked_add(product)
                        .ok_or(LatticeGeometryError::IntegerOverflow)
                },
            )?;
        }
    }
    Ok(exact_determinant(&gram)? != 0)
}

fn exact_determinant(matrix: &[Vec<i128>]) -> Result<i128, LatticeGeometryError> {
    let dimension = matrix.len();
    if matrix.iter().any(|row| row.len() != dimension) {
        return Err(LatticeGeometryError::DimensionOverflow);
    }
    if dimension == 0 {
        return Ok(1);
    }
    let mut values = matrix.to_vec();
    let mut sign = 1i128;
    let mut denominator = 1i128;
    for pivot_index in 0..dimension.saturating_sub(1) {
        let Some(pivot_row) = (pivot_index..dimension).find(|&row| values[row][pivot_index] != 0)
        else {
            return Ok(0);
        };
        if pivot_row != pivot_index {
            values.swap(pivot_row, pivot_index);
            sign = -sign;
        }
        let pivot = values[pivot_index][pivot_index];
        for row in pivot_index + 1..dimension {
            for column in pivot_index + 1..dimension {
                let diagonal = values[row][column]
                    .checked_mul(pivot)
                    .ok_or(LatticeGeometryError::IntegerOverflow)?;
                let cross = values[row][pivot_index]
                    .checked_mul(values[pivot_index][column])
                    .ok_or(LatticeGeometryError::IntegerOverflow)?;
                let numerator = diagonal
                    .checked_sub(cross)
                    .ok_or(LatticeGeometryError::IntegerOverflow)?;
                if numerator % denominator != 0 {
                    return Err(LatticeGeometryError::IntegerOverflow);
                }
                values[row][column] = numerator / denominator;
            }
            values[row][pivot_index] = 0;
        }
        denominator = pivot;
    }
    sign.checked_mul(values[dimension - 1][dimension - 1])
        .ok_or(LatticeGeometryError::IntegerOverflow)
}

fn exact_adjugate(matrix: &[Vec<i128>]) -> Result<Vec<Vec<i128>>, LatticeGeometryError> {
    let dimension = matrix.len();
    let mut result = vec![vec![0i128; dimension]; dimension];
    for (row, result_row) in result.iter_mut().enumerate() {
        for (column, result_value) in result_row.iter_mut().enumerate() {
            let minor = matrix
                .iter()
                .enumerate()
                .filter(|(source_row, _)| *source_row != column)
                .map(|(_, values)| {
                    values
                        .iter()
                        .enumerate()
                        .filter_map(|(source_column, &value)| {
                            (source_column != row).then_some(value)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut cofactor = exact_determinant(&minor)?;
            if (row + column) % 2 == 1 {
                cofactor = cofactor
                    .checked_neg()
                    .ok_or(LatticeGeometryError::IntegerOverflow)?;
            }
            *result_value = cofactor;
        }
    }
    Ok(result)
}

fn squared_norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn honeycomb() -> EmbeddedLattice {
        let root_three = 3.0f64.sqrt();
        EmbeddedLattice::new(
            vec![vec![1.0, 0.0], vec![0.5, root_three / 2.0]],
            vec![vec![0.0, 0.0], vec![0.0, 1.0 / root_three]],
        )
        .unwrap()
    }

    #[test]
    fn position_supports_lower_dimensional_embeddings() {
        let lattice =
            EmbeddedLattice::new(vec![vec![2.0, 1.0, 0.0]], vec![vec![0.25, 0.5, 3.0]]).unwrap();
        assert_eq!(lattice.vector(&[3]).unwrap(), [6.0, 3.0, 0.0]);
        assert_eq!(lattice.position(0, &[-2]).unwrap(), [-3.75, -1.5, 3.0]);
    }

    #[test]
    fn neighbor_shells_are_metric_complete_without_a_fixed_box() {
        let lattice = honeycomb();
        let counts = (0..5)
            .map(|order| lattice.neighbor_shell(order, 1.0e-8).unwrap().len())
            .collect::<Vec<_>>();
        assert_eq!(counts, [2, 3, 6, 3, 6]);

        let embedded = EmbeddedLattice::new(
            vec![vec![0.0, 1.0e8, 0.0, 0.0], vec![0.0, 0.0, 1.0e8, 0.0]],
            vec![vec![0.0; 4]],
        )
        .unwrap();
        let counts = (0..5)
            .map(|order| embedded.neighbor_shell(order, 1.0e-8).unwrap().len())
            .collect::<Vec<_>>();
        assert_eq!(counts, [1, 2, 2, 2, 4]);
    }

    #[test]
    fn nonunimodular_translation_domains_are_exact() {
        let lattice =
            EmbeddedLattice::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![vec![0.0, 0.0]])
                .unwrap();
        let domain = lattice
            .translation_domain(&[vec![10.0, 0.0], vec![7.0, 7.0]], &[], 1.0e-8)
            .unwrap();
        assert_eq!(domain.determinant(), 70);
        assert_eq!(domain.which(&[31, -20]).unwrap(), [5, -3]);
        assert_eq!(domain.to_fundamental_domain(&[31, -20]).unwrap(), [2, 1]);

        let one_direction = lattice
            .translation_domain(&[vec![10.0, 0.0]], &[vec![7, 7]], 1.0e-8)
            .unwrap();
        assert_eq!(one_direction.which(&[31, -20]).unwrap(), [5]);
    }

    #[test]
    fn generated_integer_domains_round_trip_group_elements() {
        let period_sets = [
            vec![vec![2]],
            vec![vec![2, 1], vec![1, 3]],
            vec![vec![0, 1], vec![1, 0]],
            vec![vec![2, 1, 0], vec![0, 3, 1], vec![1, 0, 2]],
            vec![vec![0, 1, 0], vec![0, 0, 1], vec![1, 0, 0]],
        ];
        for periods in period_sets {
            let dimension = periods.len();
            let primitive = (0..dimension)
                .map(|row| {
                    (0..dimension)
                        .map(|column| f64::from(row == column))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let lattice = EmbeddedLattice::new(primitive, vec![vec![0.0; dimension]]).unwrap();
            let cartesian_periods = periods
                .iter()
                .map(|period| period.iter().map(|&value| value as f64).collect())
                .collect::<Vec<_>>();
            let domain = lattice
                .translation_domain(&cartesian_periods, &[], 1.0e-12)
                .unwrap();
            for seed in -20..=20 {
                let element = (0..dimension)
                    .map(|axis| i64::from(seed + axis as i32 * 7))
                    .collect::<Vec<_>>();
                let tag = domain.shift(&element).unwrap();
                assert_eq!(domain.which(&tag).unwrap(), element);
                assert_eq!(
                    domain.to_fundamental_domain(&tag).unwrap(),
                    vec![0; dimension]
                );
            }
        }
    }

    #[test]
    fn subgroup_containment_requires_integer_coefficients() {
        let group = vec![
            vec![1.0, 0.2, -0.4],
            vec![0.3, 1.1, 0.5],
            vec![0.7, -0.2, 0.9],
        ];
        let subgroup = vec![
            (0..3).map(|component| 2.0 * group[0][component]).collect(),
            (0..3)
                .map(|component| 3.0 * group[1][component] + 4.0 * group[2][component])
                .collect(),
        ];
        assert!(contains_translation_subgroup(&group, &subgroup, 1.0e-8).unwrap());
        let scaled = group
            .iter()
            .map(|period| period.iter().map(|value| 0.8 * value).collect())
            .collect::<Vec<_>>();
        assert!(!contains_translation_subgroup(&scaled, &group, 1.0e-8).unwrap());
    }
}
