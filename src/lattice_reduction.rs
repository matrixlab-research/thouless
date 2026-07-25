//! Lattice-basis reduction and nearest lattice-vector geometry.

use std::fmt;

use nalgebra::{DMatrix, DVector};

/// A reduced real-space basis and its exact integer row transformation.
#[derive(Clone, Debug, PartialEq)]
pub struct ReducedBasis {
    vectors: Vec<Vec<f64>>,
    transformation: Vec<Vec<i64>>,
}

impl ReducedBasis {
    /// Returns reduced basis vectors as rows.
    #[must_use]
    pub fn vectors(&self) -> &[Vec<f64>] {
        &self.vectors
    }

    /// Returns integer coefficients mapping original rows to reduced rows.
    #[must_use]
    pub fn transformation(&self) -> &[Vec<i64>] {
        &self.transformation
    }
}

/// Invalid lattice geometry or an unstable reduction.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LatticeReductionError {
    EmptyBasis,
    RaggedBasis,
    TooManyBasisVectors,
    DependentBasis,
    InvalidReductionParameter,
    InvalidTargetDimension { expected: usize, actual: usize },
    InvalidNeighborCount,
    InvalidTolerance,
    DimensionTooLarge,
    NonFiniteValue,
    UnstableReduction,
}

impl fmt::Display for LatticeReductionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBasis => write!(formatter, "basis cannot be empty"),
            Self::RaggedBasis => write!(formatter, "basis vectors must have equal dimensions"),
            Self::TooManyBasisVectors => {
                write!(formatter, "basis has more vectors than ambient dimensions")
            }
            Self::DependentBasis => write!(formatter, "basis vectors are linearly dependent"),
            Self::InvalidReductionParameter => {
                write!(formatter, "reduction parameter must exceed four thirds")
            }
            Self::InvalidTargetDimension { expected, actual } => write!(
                formatter,
                "target has {actual} components; expected {expected}"
            ),
            Self::InvalidNeighborCount => {
                write!(formatter, "neighbor count must be positive")
            }
            Self::InvalidTolerance => {
                write!(
                    formatter,
                    "relative tolerance must be finite and non-negative"
                )
            }
            Self::DimensionTooLarge => {
                write!(formatter, "lattice dimension is too large to enumerate")
            }
            Self::NonFiniteValue => write!(formatter, "lattice inputs must be finite"),
            Self::UnstableReduction => {
                write!(
                    formatter,
                    "lattice reduction did not converge to a reduced basis"
                )
            }
        }
    }
}

impl std::error::Error for LatticeReductionError {}

/// Returns the Gram--Schmidt coefficient of `vector` along `reference`.
pub fn gram_schmidt_coefficient(
    vector: &[f64],
    reference: &[f64],
) -> Result<f64, LatticeReductionError> {
    if vector.len() != reference.len() {
        return Err(LatticeReductionError::RaggedBasis);
    }
    if vector
        .iter()
        .chain(reference)
        .any(|value| !value.is_finite())
    {
        return Err(LatticeReductionError::NonFiniteValue);
    }
    let denominator = dot(reference, reference);
    if denominator <= f64::EPSILON {
        return Err(LatticeReductionError::DependentBasis);
    }
    Ok(dot(vector, reference) / denominator)
}

/// Computes row-wise Gram--Schmidt vectors without normalizing them.
pub fn gram_schmidt(basis: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, LatticeReductionError> {
    validate_basis(basis)?;
    let mut orthogonal = basis.to_vec();
    for row in 0..orthogonal.len() {
        for previous in 0..row {
            let coefficient = gram_schmidt_coefficient(&orthogonal[row], &orthogonal[previous])?;
            let previous = orthogonal[previous].clone();
            for (component, previous) in orthogonal[row].iter_mut().zip(previous) {
                *component -= coefficient * previous;
            }
        }
        if squared_norm(&orthogonal[row]) <= f64::EPSILON {
            return Err(LatticeReductionError::DependentBasis);
        }
    }
    Ok(orthogonal)
}

/// Returns whether successive Gram--Schmidt lengths satisfy the reduction bound.
pub fn is_c_reduced(
    basis: &[Vec<f64>],
    reduction_parameter: f64,
) -> Result<bool, LatticeReductionError> {
    if !reduction_parameter.is_finite() {
        return Err(LatticeReductionError::InvalidReductionParameter);
    }
    let orthogonal = gram_schmidt(basis)?;
    Ok(orthogonal
        .windows(2)
        .all(|pair| squared_norm(&pair[0]) / squared_norm(&pair[1]) < reduction_parameter))
}

/// Reduces a row basis with the Lenstra--Lenstra--Lovász algorithm.
pub fn lll_reduce(
    basis: &[Vec<f64>],
    reduction_parameter: f64,
) -> Result<ReducedBasis, LatticeReductionError> {
    let (vector_count, _) = validate_basis(basis)?;
    if !reduction_parameter.is_finite() || reduction_parameter <= 4.0 / 3.0 {
        return Err(LatticeReductionError::InvalidReductionParameter);
    }
    let mut vectors = basis.to_vec();
    let mut orthogonal = basis.to_vec();
    let mut coefficients = (0..vector_count)
        .map(|row| {
            (0..vector_count)
                .map(|column| f64::from(row == column))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut transformation = (0..vector_count)
        .map(|row| {
            (0..vector_count)
                .map(|column| i64::from(row == column))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    for row in 0..vector_count {
        for previous in 0..row {
            coefficients[row][previous] =
                gram_schmidt_coefficient(&vectors[row], &orthogonal[previous])?;
            let previous_vector = orthogonal[previous].clone();
            for (component, previous_component) in orthogonal[row].iter_mut().zip(previous_vector) {
                *component -= coefficients[row][previous] * previous_component;
            }
        }
        if squared_norm(&orthogonal[row]) <= f64::EPSILON {
            return Err(LatticeReductionError::DependentBasis);
        }
        size_reduce(row, &mut vectors, &mut coefficients, &mut transformation)?;
    }

    let mut index = 0;
    let mut iterations = 0usize;
    let iteration_limit = vector_count
        .saturating_mul(vector_count)
        .saturating_mul(100_000)
        .max(100_000);
    while index + 1 < vector_count {
        iterations += 1;
        if iterations > iteration_limit {
            return Err(LatticeReductionError::UnstableReduction);
        }
        if squared_norm(&orthogonal[index])
            < reduction_parameter * squared_norm(&orthogonal[index + 1])
        {
            index += 1;
            continue;
        }

        let coupling = coefficients[index + 1][index];
        let earlier = orthogonal[index].clone();
        for (component, earlier_component) in orthogonal[index + 1].iter_mut().zip(earlier) {
            *component += coupling * earlier_component;
        }
        coefficients[index][index] =
            gram_schmidt_coefficient(&vectors[index], &orthogonal[index + 1])?;
        coefficients[index][index + 1] = 1.0;
        coefficients[index + 1][index] = 1.0;
        coefficients[index + 1][index + 1] = 0.0;
        let projection = coefficients[index][index];
        let later = orthogonal[index + 1].clone();
        for (component, later_component) in orthogonal[index].iter_mut().zip(later) {
            *component -= projection * later_component;
        }
        vectors.swap(index, index + 1);
        orthogonal.swap(index, index + 1);
        coefficients.swap(index, index + 1);
        transformation.swap(index, index + 1);
        for row in index + 2..vector_count {
            coefficients[row][index] = gram_schmidt_coefficient(&vectors[row], &orthogonal[index])?;
            coefficients[row][index + 1] =
                gram_schmidt_coefficient(&vectors[row], &orthogonal[index + 1])?;
        }
        if coefficients[index + 1][index].abs() > 0.5 {
            size_reduce(
                index + 1,
                &mut vectors,
                &mut coefficients,
                &mut transformation,
            )?;
        }
        index = index.saturating_sub(1);
    }
    if !is_c_reduced(&vectors, reduction_parameter)? {
        return Err(LatticeReductionError::UnstableReduction);
    }
    Ok(ReducedBasis {
        vectors,
        transformation,
    })
}

/// Returns coefficients of the nearest lattice vectors to `target`.
pub fn closest_lattice_vectors(
    target: &[f64],
    basis: &[Vec<f64>],
    neighbor_count: usize,
    group_by_length: bool,
    relative_tolerance: f64,
) -> Result<Vec<Vec<i64>>, LatticeReductionError> {
    let (rank, dimension) = validate_basis(basis)?;
    if target.len() != dimension {
        return Err(LatticeReductionError::InvalidTargetDimension {
            expected: dimension,
            actual: target.len(),
        });
    }
    if neighbor_count == 0 {
        return Err(LatticeReductionError::InvalidNeighborCount);
    }
    if !relative_tolerance.is_finite() || relative_tolerance < 0.0 {
        return Err(LatticeReductionError::InvalidTolerance);
    }
    if target.iter().any(|value| !value.is_finite()) {
        return Err(LatticeReductionError::NonFiniteValue);
    }

    let basis_matrix = DMatrix::from_fn(rank, dimension, |row, column| basis[row][column]);
    let gram = &basis_matrix * basis_matrix.transpose();
    let inverse = gram
        .try_inverse()
        .ok_or(LatticeReductionError::DependentBasis)?;
    let target = DVector::from_row_slice(target);
    let coordinates = &inverse * &basis_matrix * target;
    let center = coordinates
        .iter()
        .map(|value| round_ties_even(*value) as i64)
        .collect::<Vec<_>>();
    let projected = coordinates.transpose() * &basis_matrix;
    let projected = projected.row(0).iter().copied().collect::<Vec<_>>();
    let radius = 0.5
        / inverse
            .diagonal()
            .iter()
            .copied()
            .fold(0.0, f64::max)
            .sqrt();

    let mut layer = 1i64;
    loop {
        let points = integer_box(&center, layer)?;
        if points.len() < neighbor_count {
            layer += 1;
            continue;
        }
        let mut ranked = points
            .into_iter()
            .map(|point| {
                let distance = (0..dimension)
                    .map(|component| {
                        let position = (0..rank)
                            .map(|row| point[row] as f64 * basis[row][component])
                            .sum::<f64>();
                        (position - projected[component]).powi(2)
                    })
                    .sum::<f64>()
                    .sqrt();
                (distance, point)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let distances = ranked.iter().map(|item| item.0).collect::<Vec<_>>();

        let (distance_limit, keep_count, effective_tolerance) = if group_by_length {
            let boundaries = distances
                .windows(2)
                .enumerate()
                .filter_map(|(index, pair)| {
                    (pair[1] - pair[0] > relative_tolerance * radius).then_some(index)
                })
                .collect::<Vec<_>>();
            if boundaries.len() + 1 < neighbor_count {
                layer += 1;
                continue;
            }
            if boundaries.len() == neighbor_count - 1 {
                (*distances.last().unwrap(), ranked.len(), relative_tolerance)
            } else {
                let boundary = boundaries[neighbor_count - 1];
                (distances[boundary], boundary + 1, relative_tolerance)
            }
        } else {
            (distances[neighbor_count - 1], neighbor_count, 0.0)
        };
        if distance_limit < (2.0 * layer as f64 - 1.0 - effective_tolerance) * radius {
            return Ok(ranked
                .into_iter()
                .take(keep_count)
                .map(|item| item.1)
                .collect());
        }
        layer += 1;
    }
}

/// Returns integer lattice vectors whose bisectors bound the Voronoi cell.
pub fn voronoi_neighbors(
    basis: &[Vec<f64>],
    reduced: bool,
    relative_tolerance: f64,
) -> Result<Vec<Vec<i64>>, LatticeReductionError> {
    let (rank, _) = validate_basis(basis)?;
    if !relative_tolerance.is_finite() || relative_tolerance < 0.0 {
        return Err(LatticeReductionError::InvalidTolerance);
    }
    if rank >= usize::BITS as usize {
        return Err(LatticeReductionError::DimensionTooLarge);
    }
    let mut vertices = Vec::with_capacity((1usize << rank) - 1);
    for mask in 1usize..(1usize << rank) {
        let displacement = (0..rank)
            .map(|index| {
                if mask & (1usize << index) == 0 {
                    0.0
                } else {
                    0.5
                }
            })
            .collect::<Vec<_>>();
        let target = (0..basis[0].len())
            .map(|component| {
                (0..rank)
                    .map(|row| displacement[row] * basis[row][component])
                    .sum()
            })
            .collect::<Vec<_>>();
        let closest = closest_lattice_vectors(&target, basis, 1, false, relative_tolerance)?;
        vertices.push(
            (0..rank)
                .map(|index| {
                    round_ties_even((closest[0][index] as f64 - displacement[index]) * 2.0) as i64
                })
                .collect::<Vec<_>>(),
        );
    }

    if reduced {
        let basis_matrix = DMatrix::from_fn(rank, basis[0].len(), |row, column| basis[row][column]);
        let gram = &basis_matrix * basis_matrix.transpose();
        let vertex_matrix = DMatrix::from_fn(vertices.len(), rank, |row, column| {
            vertices[row][column] as f64
        });
        let products = &vertex_matrix * gram * vertex_matrix.transpose();
        let mut keep = vec![true; vertices.len()];
        for candidate in 0..vertices.len() {
            let relevant = (0..vertices.len())
                .filter(|&other| other != candidate && keep[other])
                .all(|other| {
                    let denominator = products[(other, other)];
                    denominator > 0.0
                        && (0.5 * products[(candidate, other)] / denominator).abs()
                            < 0.5 - relative_tolerance
                });
            if !relevant {
                keep[candidate] = false;
            }
        }
        vertices = vertices
            .into_iter()
            .zip(keep)
            .filter_map(|(vertex, keep)| keep.then_some(vertex))
            .collect();
    }
    let negatives = vertices
        .iter()
        .map(|vertex| vertex.iter().map(|value| -*value).collect())
        .collect::<Vec<_>>();
    vertices.extend(negatives);
    Ok(vertices)
}

fn validate_basis(basis: &[Vec<f64>]) -> Result<(usize, usize), LatticeReductionError> {
    let dimension = basis
        .first()
        .map(Vec::len)
        .ok_or(LatticeReductionError::EmptyBasis)?;
    if dimension == 0 {
        return Err(LatticeReductionError::EmptyBasis);
    }
    if basis.iter().any(|vector| vector.len() != dimension) {
        return Err(LatticeReductionError::RaggedBasis);
    }
    if basis.len() > dimension {
        return Err(LatticeReductionError::TooManyBasisVectors);
    }
    if basis.iter().flatten().any(|value| !value.is_finite()) {
        return Err(LatticeReductionError::NonFiniteValue);
    }
    Ok((basis.len(), dimension))
}

fn size_reduce(
    row: usize,
    vectors: &mut [Vec<f64>],
    coefficients: &mut [Vec<f64>],
    transformation: &mut [Vec<i64>],
) -> Result<(), LatticeReductionError> {
    for previous in (0..row).rev() {
        let multiple = round_ties_even(coefficients[row][previous]);
        if multiple.abs() > i64::MAX as f64 {
            return Err(LatticeReductionError::UnstableReduction);
        }
        let multiple_integer = multiple as i64;
        for component in 0..vectors[row].len() {
            vectors[row][component] -= multiple * vectors[previous][component];
        }
        for column in 0..transformation[row].len() {
            transformation[row][column] = transformation[row][column]
                .checked_sub(
                    multiple_integer
                        .checked_mul(transformation[previous][column])
                        .ok_or(LatticeReductionError::UnstableReduction)?,
                )
                .ok_or(LatticeReductionError::UnstableReduction)?;
        }
        let previous_coefficients = coefficients[previous].clone();
        for column in 0..coefficients[row].len() {
            coefficients[row][column] -= multiple * previous_coefficients[column];
        }
    }
    Ok(())
}

fn integer_box(center: &[i64], radius: i64) -> Result<Vec<Vec<i64>>, LatticeReductionError> {
    let side =
        usize::try_from(2 * radius + 1).map_err(|_| LatticeReductionError::DimensionTooLarge)?;
    let count = side
        .checked_pow(
            u32::try_from(center.len()).map_err(|_| LatticeReductionError::DimensionTooLarge)?,
        )
        .ok_or(LatticeReductionError::DimensionTooLarge)?;
    let mut points = Vec::with_capacity(count);
    for mut encoded in 0..count {
        let mut point = Vec::with_capacity(center.len());
        for &coordinate in center {
            let offset = i64::try_from(encoded % side)
                .map_err(|_| LatticeReductionError::DimensionTooLarge)?
                - radius;
            point.push(
                coordinate
                    .checked_add(offset)
                    .ok_or(LatticeReductionError::DimensionTooLarge)?,
            );
            encoded /= side;
        }
        points.push(point);
    }
    Ok(points)
}

fn round_ties_even(value: f64) -> f64 {
    value.round_ties_even()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn squared_norm(vector: &[f64]) -> f64 {
    dot(vector, vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduction_preserves_the_integer_lattice() {
        let basis = vec![vec![1.0, 1.0, 1.0], vec![-1.0, 0.0, 2.0]];
        let reduced = lll_reduce(&basis, 1.34).unwrap();
        assert!(is_c_reduced(reduced.vectors(), 1.34).unwrap());
        for (row, coefficients) in reduced.transformation().iter().enumerate() {
            for component in 0..3 {
                let reconstructed = coefficients
                    .iter()
                    .zip(&basis)
                    .map(|(&coefficient, vector)| coefficient as f64 * vector[component])
                    .sum::<f64>();
                assert!((reconstructed - reduced.vectors()[row][component]).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn cubic_half_offset_has_eight_equidistant_neighbors() {
        let points = closest_lattice_vectors(
            &[0.5, 0.5, 0.5],
            &[
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            1,
            true,
            1.0e-9,
        )
        .unwrap();
        assert_eq!(points.len(), 8);
    }
}
