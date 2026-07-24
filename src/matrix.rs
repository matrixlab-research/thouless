//! Small owned dense matrices used at the public scientific boundary.

use crate::{Complex64, MatrixError};

/// An owned row-major dense complex matrix.
///
/// The type deliberately exposes a backend-independent public representation.
/// Numerical algorithms may convert it to optimized internal backends.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexMatrix {
    rows: usize,
    columns: usize,
    data: Vec<Complex64>,
}

impl ComplexMatrix {
    /// Creates a matrix from row-major entries.
    pub fn new(rows: usize, columns: usize, data: Vec<Complex64>) -> Result<Self, MatrixError> {
        if data.len() != rows.saturating_mul(columns) {
            return Err(MatrixError::InvalidDataLength {
                rows,
                columns,
                actual: data.len(),
            });
        }
        if data
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(MatrixError::NonFiniteValue);
        }
        Ok(Self {
            rows,
            columns,
            data,
        })
    }

    /// Creates a zero matrix.
    #[must_use]
    pub fn zeros(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            data: vec![Complex64::new(0.0, 0.0); rows.saturating_mul(columns)],
        }
    }

    /// Creates an identity matrix.
    #[must_use]
    pub fn identity(dimension: usize) -> Self {
        let mut matrix = Self::zeros(dimension, dimension);
        for index in 0..dimension {
            matrix.data[index * dimension + index] = Complex64::new(1.0, 0.0);
        }
        matrix
    }

    /// Creates a one-by-one matrix.
    #[must_use]
    pub fn scalar(value: Complex64) -> Self {
        Self {
            rows: 1,
            columns: 1,
            data: vec![value],
        }
    }

    /// Returns the number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the matrix shape.
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    /// Returns the row-major entries.
    #[must_use]
    pub fn as_slice(&self) -> &[Complex64] {
        &self.data
    }

    /// Consumes the matrix and returns its row-major entries.
    #[must_use]
    pub fn into_vec(self) -> Vec<Complex64> {
        self.data
    }

    /// Returns one entry.
    pub fn get(&self, row: usize, column: usize) -> Result<Complex64, MatrixError> {
        let index = self.index(row, column)?;
        Ok(self.data[index])
    }

    /// Replaces one entry.
    pub fn set(&mut self, row: usize, column: usize, value: Complex64) -> Result<(), MatrixError> {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(MatrixError::NonFiniteValue);
        }
        let index = self.index(row, column)?;
        self.data[index] = value;
        Ok(())
    }

    /// Adds a value to one entry.
    pub fn add_entry(
        &mut self,
        row: usize,
        column: usize,
        value: Complex64,
    ) -> Result<(), MatrixError> {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(MatrixError::NonFiniteValue);
        }
        let index = self.index(row, column)?;
        self.data[index] += value;
        Ok(())
    }

    /// Returns the conjugate transpose.
    #[must_use]
    pub fn adjoint(&self) -> Self {
        let mut result = Self::zeros(self.columns, self.rows);
        for row in 0..self.rows {
            for column in 0..self.columns {
                result.data[column * self.rows + row] =
                    self.data[row * self.columns + column].conj();
            }
        }
        result
    }

    /// Returns whether the matrix is Hermitian within an absolute tolerance.
    pub fn is_hermitian(&self, tolerance: f64) -> Result<bool, MatrixError> {
        if self.rows != self.columns {
            return Err(MatrixError::NotSquare {
                rows: self.rows,
                columns: self.columns,
            });
        }
        for row in 0..self.rows {
            for column in 0..row {
                let left = self.data[row * self.columns + column];
                let right = self.data[column * self.columns + row].conj();
                if (left - right).norm() > tolerance {
                    return Ok(false);
                }
            }
            if self.data[row * self.columns + row].im.abs() > tolerance {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn index(&self, row: usize, column: usize) -> Result<usize, MatrixError> {
        if row >= self.rows || column >= self.columns {
            return Err(MatrixError::IndexOutOfBounds {
                row,
                column,
                rows: self.rows,
                columns: self.columns,
            });
        }
        Ok(row * self.columns + column)
    }
}
