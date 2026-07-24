//! Hermitian spectral algorithms.

use nalgebra::linalg::SymmetricEigen;
use nalgebra::DMatrix;

use crate::{Complex64, ComplexMatrix, SpectrumError};

/// Eigenvalues and column eigenvectors of a Hermitian matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Eigensystem {
    eigenvalues: Vec<f64>,
    eigenvectors: ComplexMatrix,
}

impl Eigensystem {
    /// Returns eigenvalues in ascending order.
    #[must_use]
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Returns eigenvectors as columns in eigenvalue order.
    #[must_use]
    pub const fn eigenvectors(&self) -> &ComplexMatrix {
        &self.eigenvectors
    }
}

/// Diagonalizes a Hermitian matrix.
pub fn hermitian_eigensystem(
    matrix: &ComplexMatrix,
    tolerance: f64,
) -> Result<Eigensystem, SpectrumError> {
    if matrix.rows() != matrix.columns() {
        return Err(SpectrumError::NotSquare {
            rows: matrix.rows(),
            columns: matrix.columns(),
        });
    }
    if !matrix.is_hermitian(tolerance)? {
        return Err(SpectrumError::NonHermitian);
    }

    let dimension = matrix.rows();
    let backend = DMatrix::from_row_slice(dimension, dimension, matrix.as_slice());
    let decomposition = SymmetricEigen::new(backend);

    let mut order: Vec<usize> = (0..dimension).collect();
    order.sort_by(|left, right| {
        decomposition.eigenvalues[*left].total_cmp(&decomposition.eigenvalues[*right])
    });

    let eigenvalues = order
        .iter()
        .map(|index| decomposition.eigenvalues[*index])
        .collect();
    let mut eigenvectors = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for (new_column, old_column) in order.iter().enumerate() {
        for row in 0..dimension {
            eigenvectors[row * dimension + new_column] =
                decomposition.eigenvectors[(row, *old_column)];
        }
    }

    Ok(Eigensystem {
        eigenvalues,
        eigenvectors: ComplexMatrix::new(dimension, dimension, eigenvectors)?,
    })
}
