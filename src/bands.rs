//! Band energies and momentum derivatives of one-dimensional periodic leads.

use std::fmt;

use crate::spectrum::hermitian_eigensystem;
use crate::{Complex64, ComplexMatrix, MatrixError, SpectrumError};

const HERMITIAN_TOLERANCE: f64 = 1.0e-10;

/// A principal-cell Hamiltonian and hopping to the neighboring cell.
#[derive(Clone, Debug, PartialEq)]
pub struct PeriodicBands {
    cell_hamiltonian: ComplexMatrix,
    inter_cell_hopping: ComplexMatrix,
}

/// Energies and requested momentum derivatives at one momentum.
#[derive(Clone, Debug, PartialEq)]
pub struct BandEvaluation {
    energies: Vec<f64>,
    first_derivatives: Option<Vec<f64>>,
    second_derivatives: Option<Vec<f64>>,
    eigenvectors: Option<ComplexMatrix>,
}

impl BandEvaluation {
    /// Band energies in ascending order.
    #[must_use]
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// First momentum derivatives, when requested.
    #[must_use]
    pub fn first_derivatives(&self) -> Option<&[f64]> {
        self.first_derivatives.as_deref()
    }

    /// Second momentum derivatives, when requested.
    #[must_use]
    pub fn second_derivatives(&self) -> Option<&[f64]> {
        self.second_derivatives.as_deref()
    }

    /// Column eigenvectors, when requested.
    #[must_use]
    pub const fn eigenvectors(&self) -> Option<&ComplexMatrix> {
        self.eigenvectors.as_ref()
    }
}

/// Failures raised by periodic-band evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BandError {
    /// Cell and hopping matrices do not have one common nonzero square shape.
    InvalidShape,
    /// The cell Hamiltonian is not Hermitian.
    NonHermitianCell,
    /// Momentum is NaN or infinite.
    InvalidMomentum,
    /// Only energy, velocity, and curvature are implemented.
    UnsupportedDerivativeOrder {
        /// Requested derivative order.
        order: usize,
    },
    /// Dense eigendecomposition failed.
    EigensystemFailure,
    /// Matrix construction failed.
    Matrix(MatrixError),
}

impl fmt::Display for BandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => write!(
                formatter,
                "cell Hamiltonian and hopping must share a nonzero square shape"
            ),
            Self::NonHermitianCell => write!(formatter, "cell Hamiltonian is not Hermitian"),
            Self::InvalidMomentum => write!(formatter, "momentum must be finite"),
            Self::UnsupportedDerivativeOrder { order } => {
                write!(formatter, "derivative order {order} is not supported")
            }
            Self::EigensystemFailure => write!(formatter, "band eigensystem failed"),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for BandError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatrixError> for BandError {
    fn from(error: MatrixError) -> Self {
        Self::Matrix(error)
    }
}

impl PeriodicBands {
    /// Validate and store one periodic principal cell.
    pub fn new(
        cell_hamiltonian: ComplexMatrix,
        inter_cell_hopping: ComplexMatrix,
    ) -> Result<Self, BandError> {
        let dimension = cell_hamiltonian.rows();
        if dimension == 0
            || cell_hamiltonian.columns() != dimension
            || inter_cell_hopping.shape() != (dimension, dimension)
        {
            return Err(BandError::InvalidShape);
        }
        if !cell_hamiltonian.is_hermitian(HERMITIAN_TOLERANCE)? {
            return Err(BandError::NonHermitianCell);
        }
        Ok(Self {
            cell_hamiltonian,
            inter_cell_hopping,
        })
    }

    /// Evaluate energies and derivatives through the requested order.
    pub fn evaluate(
        &self,
        momentum: f64,
        derivative_order: usize,
        return_eigenvectors: bool,
    ) -> Result<BandEvaluation, BandError> {
        if !momentum.is_finite() {
            return Err(BandError::InvalidMomentum);
        }
        if derivative_order > 2 {
            return Err(BandError::UnsupportedDerivativeOrder {
                order: derivative_order,
            });
        }

        let dimension = self.cell_hamiltonian.rows();
        let phase = Complex64::from_polar(1.0, -momentum);
        let mut hamiltonian = self.cell_hamiltonian.clone();
        let mut first_operator = ComplexMatrix::zeros(dimension, dimension);
        let mut second_operator = ComplexMatrix::zeros(dimension, dimension);
        for row in 0..dimension {
            for column in 0..dimension {
                let forward = self.inter_cell_hopping.as_slice()[row * dimension + column] * phase;
                let backward = self.inter_cell_hopping.as_slice()[column * dimension + row].conj()
                    * phase.conj();
                hamiltonian.add_entry(row, column, forward + backward)?;
                if derivative_order >= 1 {
                    first_operator.set(
                        row,
                        column,
                        Complex64::new(0.0, 1.0) * (-forward + backward),
                    )?;
                }
                if derivative_order >= 2 {
                    second_operator.set(row, column, -forward - backward)?;
                }
            }
        }

        let eigensystem = hermitian_eigensystem(&hamiltonian, HERMITIAN_TOLERANCE).map_err(
            |error| match error {
                SpectrumError::Matrix(matrix_error) => BandError::Matrix(matrix_error),
                _ => BandError::EigensystemFailure,
            },
        )?;
        let energies = eigensystem.eigenvalues().to_vec();
        let vectors = eigensystem.eigenvectors();
        let transformed_first =
            (derivative_order >= 1).then(|| transform_operator(&first_operator, vectors));
        let first_derivatives = transformed_first.as_ref().map(|operator| {
            (0..dimension)
                .map(|band| operator[band * dimension + band].re)
                .collect()
        });
        let second_derivatives = if derivative_order >= 2 {
            let transformed_second = transform_operator(&second_operator, vectors);
            let transformed_first = transformed_first
                .as_ref()
                .expect("first derivative is computed with second derivative");
            Some(
                (0..dimension)
                    .map(|band| {
                        let mixing = (0..dimension)
                            .filter(|other| energies[*other] != energies[band])
                            .map(|other| {
                                transformed_first[other * dimension + band].norm_sqr()
                                    / (energies[other] - energies[band])
                            })
                            .sum::<f64>();
                        transformed_second[band * dimension + band].re - 2.0 * mixing
                    })
                    .collect(),
            )
        } else {
            None
        };

        Ok(BandEvaluation {
            energies,
            first_derivatives,
            second_derivatives,
            eigenvectors: return_eigenvectors.then(|| vectors.clone()),
        })
    }
}

fn transform_operator(operator: &ComplexMatrix, vectors: &ComplexMatrix) -> Vec<Complex64> {
    let dimension = operator.rows();
    let mut transformed = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for left_band in 0..dimension {
        for right_band in 0..dimension {
            transformed[left_band * dimension + right_band] = (0..dimension)
                .flat_map(|left_basis| {
                    (0..dimension).map(move |right_basis| {
                        vectors.as_slice()[left_basis * dimension + left_band].conj()
                            * operator.as_slice()[left_basis * dimension + right_basis]
                            * vectors.as_slice()[right_basis * dimension + right_band]
                    })
                })
                .sum();
        }
    }
    transformed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_chain_derivatives_match_the_cosine_band() {
        let onsite = ComplexMatrix::scalar(Complex64::new(0.3, 0.0));
        let hopping = ComplexMatrix::scalar(Complex64::new(-1.2, 0.0));
        let bands = PeriodicBands::new(onsite, hopping).unwrap();
        let momentum = 0.7;
        let result = bands.evaluate(momentum, 2, true).unwrap();
        assert!((result.energies()[0] - (0.3 - 2.4 * momentum.cos())).abs() < 1.0e-12);
        assert!((result.first_derivatives().unwrap()[0] - 2.4 * momentum.sin()).abs() < 1.0e-12);
        assert!((result.second_derivatives().unwrap()[0] - 2.4 * momentum.cos()).abs() < 1.0e-12);
        assert_eq!(result.eigenvectors().unwrap().shape(), (1, 1));
    }
}
