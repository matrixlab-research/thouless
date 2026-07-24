//! Hermitian spectral algorithms.

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
    let decomposition = thouless_lapack::hermitian_eigensystem(dimension, matrix.as_slice())
        .map_err(|_| SpectrumError::DecompositionFailure)?;
    let eigenvalues = decomposition.eigenvalues().to_vec();
    let mut eigenvectors = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for column in 0..dimension {
        for row in 0..dimension {
            eigenvectors[row * dimension + column] =
                decomposition.eigenvectors_column_major()[row + column * dimension];
        }
    }

    Ok(Eigensystem {
        eigenvalues,
        eigenvectors: ComplexMatrix::new(dimension, dimension, eigenvectors)?,
    })
}
