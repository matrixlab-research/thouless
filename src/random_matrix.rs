//! Random-matrix ensemble projections for the ten Altland--Zirnbauer classes.
//!
//! Random-number generation is deliberately outside this module. Callers
//! provide independent standard-normal components and optional random bits;
//! the scientific core performs the symmetry projection and matrix
//! factorization deterministically.

use std::fmt;

use nalgebra::DMatrix;

use crate::{Complex64, ComplexMatrix};

/// One of the ten Altland--Zirnbauer symmetry classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymmetryClass {
    A,
    Ai,
    Aii,
    Aiii,
    Bdi,
    Cii,
    D,
    Diii,
    C,
    Ci,
}

impl SymmetryClass {
    /// Returns the square of the antiunitary time-reversal operation.
    #[must_use]
    pub const fn time_reversal_square(self) -> i8 {
        match self {
            Self::Ai | Self::Bdi | Self::Ci => 1,
            Self::Aii | Self::Cii | Self::Diii => -1,
            _ => 0,
        }
    }

    /// Returns the square of the antiunitary particle-hole operation.
    #[must_use]
    pub const fn particle_hole_square(self) -> i8 {
        match self {
            Self::D | Self::Diii | Self::Bdi => 1,
            Self::C | Self::Ci | Self::Cii => -1,
            _ => 0,
        }
    }

    /// Returns whether the class has chiral symmetry.
    #[must_use]
    pub const fn has_chiral_symmetry(self) -> bool {
        matches!(
            self,
            Self::Aiii | Self::Bdi | Self::Cii | Self::Diii | Self::Ci
        )
    }

    const fn needs_even_dimension(self) -> bool {
        self.has_chiral_symmetry()
            || self.time_reversal_square() == -1
            || self.particle_hole_square() == -1
    }
}

/// Failures while projecting a random seed onto a symmetry ensemble.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RandomMatrixError {
    /// A symmetry class requires an even matrix dimension.
    EvenDimensionRequired,
    /// Class CII requires a dimension divisible by four.
    MultipleOfFourRequired,
    /// A standard-normal component array has the wrong length.
    InvalidComponentCount {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A component is NaN or infinite.
    NonFiniteComponent,
    /// A topological sector is outside the class-specific range.
    InvalidTopologicalSector,
    /// LAPACK failed while constructing the circular ensemble.
    FactorizationFailure(String),
    /// The backend QR decomposition did not preserve symplectic pairing.
    SymplecticFactorizationFailure,
}

impl fmt::Display for RandomMatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EvenDimensionRequired => {
                write!(
                    formatter,
                    "the selected symmetry class requires an even dimension"
                )
            }
            Self::MultipleOfFourRequired => {
                write!(
                    formatter,
                    "class CII requires a dimension divisible by four"
                )
            }
            Self::InvalidComponentCount {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} contains {actual} components; expected {expected}"
            ),
            Self::NonFiniteComponent => {
                write!(formatter, "random-matrix components must be finite")
            }
            Self::InvalidTopologicalSector => {
                write!(
                    formatter,
                    "invalid topological sector for the symmetry class"
                )
            }
            Self::FactorizationFailure(error) => error.fmt(formatter),
            Self::SymplecticFactorizationFailure => write!(
                formatter,
                "QR factorization did not preserve the symplectic pair structure"
            ),
        }
    }
}

impl std::error::Error for RandomMatrixError {}

/// Projects standard-normal components onto a Gaussian Hamiltonian ensemble.
pub fn gaussian_from_components(
    dimension: usize,
    symmetry: SymmetryClass,
    variance: f64,
    real: &[f64],
    imaginary: &[f64],
) -> Result<ComplexMatrix, RandomMatrixError> {
    validate_dimension(dimension, symmetry, true)?;
    let count = entry_count(dimension)?;
    validate_components("real components", count, real)?;
    validate_components("imaginary components", count, imaginary)?;

    let mut matrix = real
        .iter()
        .zip(imaginary)
        .map(|(&real, &imaginary)| match symmetry {
            SymmetryClass::Ai => Complex64::new(real, 0.0),
            SymmetryClass::D | SymmetryClass::Bdi => Complex64::new(0.0, real),
            _ => Complex64::new(real, imaginary),
        })
        .collect::<Vec<_>>();
    hermitize(&mut matrix, dimension);
    let mut scale = variance / 2.0_f64.sqrt();

    if symmetry.has_chiral_symmetry() {
        for row in 0..dimension {
            let row_sign = alternating_sign(row);
            for column in 0..dimension {
                let index = row * dimension + column;
                let value = matrix[index];
                matrix[index] -= value * row_sign * alternating_sign(column);
            }
        }
        scale *= 0.5;
    }

    match symmetry {
        SymmetryClass::Aii | SymmetryClass::Diii => {
            apply_paired_conjugation(&mut matrix, dimension, 1.0);
            scale /= 2.0_f64.sqrt();
        }
        SymmetryClass::C | SymmetryClass::Ci => {
            apply_paired_conjugation(&mut matrix, dimension, -1.0);
            scale /= 2.0_f64.sqrt();
        }
        SymmetryClass::Cii => {
            let source = matrix.clone();
            for row in 0..dimension {
                let row_sign = spin_sign(row);
                let paired_row = spin_partner(row);
                for column in 0..dimension {
                    let paired_column = spin_partner(column);
                    matrix[row * dimension + column] +=
                        source[paired_row * dimension + paired_column].conj()
                            * row_sign
                            * spin_sign(column);
                }
            }
            scale /= 2.0_f64.sqrt();
        }
        _ => {}
    }
    for value in &mut matrix {
        *value *= scale;
    }
    matrix_result(dimension, matrix)
}

/// Projects standard-normal components onto a circular ensemble.
///
/// `random_bits` is consumed only for a chiral class when
/// `topological_sector` is `None`. It contains one bit per channel, or one
/// bit per Kramers pair for class CII.
pub fn circular_from_components(
    dimension: usize,
    symmetry: SymmetryClass,
    topological_sector: Option<i32>,
    real: &[f64],
    imaginary: &[f64],
    random_bits: &[bool],
) -> Result<ComplexMatrix, RandomMatrixError> {
    validate_dimension(dimension, symmetry, false)?;
    let count = entry_count(dimension)?;
    validate_components("real components", count, real)?;
    validate_components("imaginary components", count, imaginary)?;

    let mut seed = real
        .iter()
        .zip(imaginary)
        .map(|(&real, &imaginary)| {
            if symmetry.particle_hole_square() == 1 {
                Complex64::new(real, 0.0)
            } else {
                Complex64::new(real, imaginary)
            }
        })
        .collect::<Vec<_>>();
    if symmetry.particle_hole_square() == -1 {
        let source = seed.clone();
        for row in 0..dimension {
            let row_sign = alternating_sign(row);
            for column in 0..dimension {
                seed[row * dimension + column] -= source[paired(row) * dimension + paired(column)]
                    .conj()
                    * row_sign
                    * alternating_sign(column);
                seed[row * dimension + column] *= Complex64::new(0.0, 1.0);
            }
        }
    }

    let decomposition = thouless_lapack::complex_qr(dimension, &seed)
        .map_err(|error| RandomMatrixError::FactorizationFailure(error.to_string()))?;
    if symmetry.particle_hole_square() == -1
        && decomposition
            .first_superdiagonal()
            .iter()
            .step_by(2)
            .any(|value| value.norm() > 1.0e-8)
    {
        return Err(RandomMatrixError::SymplecticFactorizationFailure);
    }
    let mut matrix = decomposition.unitary_row_major().to_vec();
    for column in 0..dimension {
        let diagonal = decomposition.diagonal()[column];
        let phase = if diagonal.norm() > 0.0 {
            diagonal / diagonal.norm()
        } else {
            Complex64::new(1.0, 0.0)
        };
        for row in 0..dimension {
            matrix[row * dimension + column] *= phase;
        }
    }

    if matches!(symmetry, SymmetryClass::D | SymmetryClass::Diii) {
        if let Some(sector) = topological_sector {
            if !matches!(sector, -1 | 1) {
                return Err(RandomMatrixError::InvalidTopologicalSector);
            }
            let dense = DMatrix::from_row_slice(dimension, dimension, &matrix);
            let mut determinant = dense.determinant().re;
            if symmetry == SymmetryClass::Diii && dimension / 2 % 2 == 1 {
                determinant = -determinant;
            }
            if (sector > 0) != (determinant > 0.0) {
                if dimension < 2 {
                    return Err(RandomMatrixError::InvalidTopologicalSector);
                }
                for column in 0..dimension {
                    matrix.swap(
                        (dimension - 2) * dimension + column,
                        (dimension - 1) * dimension + column,
                    );
                }
            }
        }
    }

    matrix = match symmetry {
        SymmetryClass::Ai | SymmetryClass::Ci => {
            let mut projected = multiply(&transpose(&matrix, dimension), &matrix, dimension);
            if symmetry == SymmetryClass::Ci {
                let source = projected.clone();
                for row in 0..dimension {
                    for column in 0..dimension {
                        projected[row * dimension + column] = Complex64::new(0.0, 1.0)
                            * alternating_sign(column)
                            * source[row * dimension + paired(column)];
                    }
                }
            }
            projected
        }
        SymmetryClass::Aii | SymmetryClass::Diii => {
            let transposed = transpose(&matrix, dimension);
            let mut projected = vec![Complex64::new(0.0, 0.0); count];
            for row in 0..dimension {
                for column in 0..dimension {
                    projected[row * dimension + column] = (0..dimension)
                        .map(|inner| {
                            Complex64::new(0.0, 1.0)
                                * transposed[row * dimension + inner]
                                * alternating_sign(inner)
                                * matrix[paired(inner) * dimension + column]
                        })
                        .sum();
                }
            }
            projected
        }
        SymmetryClass::Aiii | SymmetryClass::Bdi | SymmetryClass::Cii => {
            let diagonal =
                topological_diagonal(dimension, symmetry, topological_sector, random_bits)?;
            let adjoint = adjoint(&matrix, dimension);
            let mut weighted = adjoint;
            for row in 0..dimension {
                for column in 0..dimension {
                    weighted[row * dimension + column] *= diagonal[column];
                }
            }
            multiply(&weighted, &matrix, dimension)
        }
        _ => matrix,
    };
    matrix_result(dimension, matrix)
}

fn entry_count(dimension: usize) -> Result<usize, RandomMatrixError> {
    dimension
        .checked_mul(dimension)
        .ok_or(RandomMatrixError::InvalidComponentCount {
            name: "matrix",
            expected: usize::MAX,
            actual: 0,
        })
}

fn validate_components(
    name: &'static str,
    expected: usize,
    values: &[f64],
) -> Result<(), RandomMatrixError> {
    if values.len() != expected {
        return Err(RandomMatrixError::InvalidComponentCount {
            name,
            expected,
            actual: values.len(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(RandomMatrixError::NonFiniteComponent);
    }
    Ok(())
}

fn validate_dimension(
    dimension: usize,
    symmetry: SymmetryClass,
    gaussian: bool,
) -> Result<(), RandomMatrixError> {
    let needs_even = if gaussian {
        symmetry.needs_even_dimension()
    } else {
        symmetry.time_reversal_square() == -1 || symmetry.particle_hole_square() == -1
    };
    if needs_even && dimension % 2 == 1 {
        return Err(RandomMatrixError::EvenDimensionRequired);
    }
    if gaussian && symmetry == SymmetryClass::Cii && dimension % 4 != 0 {
        return Err(RandomMatrixError::MultipleOfFourRequired);
    }
    Ok(())
}

fn hermitize(matrix: &mut [Complex64], dimension: usize) {
    let source = matrix.to_vec();
    for row in 0..dimension {
        for column in 0..dimension {
            matrix[row * dimension + column] =
                source[row * dimension + column] + source[column * dimension + row].conj();
        }
    }
}

fn apply_paired_conjugation(matrix: &mut [Complex64], dimension: usize, sign: f64) {
    let source = matrix.to_vec();
    for row in 0..dimension {
        for column in 0..dimension {
            matrix[row * dimension + column] += source[paired(row) * dimension + paired(column)]
                .conj()
                * alternating_sign(row)
                * alternating_sign(column)
                * sign;
        }
    }
}

const fn alternating_sign(index: usize) -> f64 {
    if index % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

const fn paired(index: usize) -> usize {
    if index % 2 == 0 {
        index + 1
    } else {
        index - 1
    }
}

const fn spin_sign(index: usize) -> f64 {
    if index % 4 < 2 {
        1.0
    } else {
        -1.0
    }
}

const fn spin_partner(index: usize) -> usize {
    if index % 4 < 2 {
        index + 2
    } else {
        index - 2
    }
}

fn topological_diagonal(
    dimension: usize,
    symmetry: SymmetryClass,
    sector: Option<i32>,
    random_bits: &[bool],
) -> Result<Vec<f64>, RandomMatrixError> {
    let pair_count = if symmetry == SymmetryClass::Cii {
        dimension / 2
    } else {
        dimension
    };
    if let Some(sector) = sector {
        let sector =
            usize::try_from(sector).map_err(|_| RandomMatrixError::InvalidTopologicalSector)?;
        if sector > pair_count {
            return Err(RandomMatrixError::InvalidTopologicalSector);
        }
        let negative = if symmetry == SymmetryClass::Cii {
            2 * sector
        } else {
            sector
        };
        return Ok((0..dimension)
            .map(|index| if index < negative { -1.0 } else { 1.0 })
            .collect());
    }
    if random_bits.len() != pair_count {
        return Err(RandomMatrixError::InvalidComponentCount {
            name: "topological random bits",
            expected: pair_count,
            actual: random_bits.len(),
        });
    }
    if symmetry == SymmetryClass::Cii {
        Ok(random_bits
            .iter()
            .flat_map(|&bit| [if bit { 1.0 } else { -1.0 }; 2])
            .collect())
    } else {
        Ok(random_bits
            .iter()
            .map(|&bit| if bit { 1.0 } else { -1.0 })
            .collect())
    }
}

fn transpose(matrix: &[Complex64], dimension: usize) -> Vec<Complex64> {
    (0..dimension)
        .flat_map(|row| (0..dimension).map(move |column| matrix[column * dimension + row]))
        .collect()
}

fn adjoint(matrix: &[Complex64], dimension: usize) -> Vec<Complex64> {
    (0..dimension)
        .flat_map(|row| (0..dimension).map(move |column| matrix[column * dimension + row].conj()))
        .collect()
}

fn multiply(left: &[Complex64], right: &[Complex64], dimension: usize) -> Vec<Complex64> {
    (0..dimension)
        .flat_map(|row| {
            (0..dimension).map(move |column| {
                (0..dimension)
                    .map(|inner| left[row * dimension + inner] * right[inner * dimension + column])
                    .sum()
            })
        })
        .collect()
}

fn matrix_result(
    dimension: usize,
    values: Vec<Complex64>,
) -> Result<ComplexMatrix, RandomMatrixError> {
    ComplexMatrix::new(dimension, dimension, values)
        .map_err(|error| RandomMatrixError::FactorizationFailure(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn components(dimension: usize) -> (Vec<f64>, Vec<f64>) {
        let count = dimension * dimension;
        (
            (0..count)
                .map(|index| ((index * 17 + 3) as f64).sin())
                .collect(),
            (0..count)
                .map(|index| ((index * 11 + 5) as f64).cos())
                .collect(),
        )
    }

    #[test]
    fn gaussian_projection_is_hermitian_for_every_class() {
        let classes = [
            SymmetryClass::A,
            SymmetryClass::Ai,
            SymmetryClass::Aii,
            SymmetryClass::Aiii,
            SymmetryClass::Bdi,
            SymmetryClass::Cii,
            SymmetryClass::D,
            SymmetryClass::Diii,
            SymmetryClass::C,
            SymmetryClass::Ci,
        ];
        let (real, imaginary) = components(8);
        for symmetry in classes {
            let matrix = gaussian_from_components(8, symmetry, 1.0, &real, &imaginary).unwrap();
            assert!(matrix.is_hermitian(1.0e-12).unwrap(), "{symmetry:?}");
        }
    }

    #[test]
    fn circular_projection_is_unitary_across_dimensions_and_classes() {
        let classes = [
            SymmetryClass::A,
            SymmetryClass::Ai,
            SymmetryClass::Aii,
            SymmetryClass::Aiii,
            SymmetryClass::Bdi,
            SymmetryClass::Cii,
            SymmetryClass::D,
            SymmetryClass::Diii,
            SymmetryClass::C,
            SymmetryClass::Ci,
        ];
        let dimension = 8;
        let (real, imaginary) = components(dimension);
        for symmetry in classes {
            let bit_count = if symmetry == SymmetryClass::Cii {
                dimension / 2
            } else if matches!(symmetry, SymmetryClass::Aiii | SymmetryClass::Bdi) {
                dimension
            } else {
                0
            };
            let matrix = circular_from_components(
                dimension,
                symmetry,
                None,
                &real,
                &imaginary,
                &vec![true; bit_count],
            )
            .unwrap();
            let product = multiply(matrix.as_slice(), &matrix.adjoint().into_vec(), dimension);
            for row in 0..dimension {
                for column in 0..dimension {
                    let expected = if row == column { 1.0 } else { 0.0 };
                    assert!(
                        (product[row * dimension + column] - Complex64::new(expected, 0.0)).norm()
                            < 1.0e-10,
                        "{symmetry:?} ({row}, {column})"
                    );
                }
            }
        }
    }
}
