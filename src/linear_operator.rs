//! Matrix-free linear operators and a canonical CSR implementation.

use std::error::Error;
use std::fmt;

use crate::{Complex64, ComplexMatrix};

/// Errors raised while constructing or applying a linear operator.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinearOperatorError {
    /// A matrix must have at least one row and one column.
    EmptyShape {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// An operation requires a square matrix.
    NonSquare {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// CSR storage requires one row offset per row plus a terminal offset.
    InvalidRowOffsetCount {
        /// Required number of offsets.
        expected: usize,
        /// Supplied number of offsets.
        actual: usize,
    },
    /// The first CSR row offset must be zero.
    NonzeroFirstRowOffset {
        /// Supplied first offset.
        actual: usize,
    },
    /// CSR row offsets must be monotone.
    NonmonotoneRowOffsets {
        /// Row whose terminal offset is smaller than its initial offset.
        row: usize,
    },
    /// The terminal CSR offset must equal the number of stored entries.
    InvalidTerminalRowOffset {
        /// Number of stored entries.
        expected: usize,
        /// Supplied terminal offset.
        actual: usize,
    },
    /// CSR column indices and values must have the same length.
    InvalidStoredEntryCount {
        /// Number of column indices.
        indices: usize,
        /// Number of values.
        values: usize,
    },
    /// A CSR column index lies outside the matrix.
    ColumnOutOfBounds {
        /// Row containing the invalid index.
        row: usize,
        /// Supplied column index.
        column: usize,
        /// Number of matrix columns.
        columns: usize,
    },
    /// Columns within each CSR row must be strictly increasing.
    NoncanonicalRow {
        /// Row containing an unsorted or duplicate column.
        row: usize,
        /// Previous column index.
        previous: usize,
        /// Current column index.
        current: usize,
    },
    /// A matrix or vector value is NaN or infinity.
    NonFiniteValue,
    /// An input vector has an incompatible length.
    InputDimension {
        /// Required input length.
        expected: usize,
        /// Supplied input length.
        actual: usize,
    },
    /// An output vector has an incompatible length.
    OutputDimension {
        /// Required output length.
        expected: usize,
        /// Supplied output length.
        actual: usize,
    },
    /// A tolerance is negative or non-finite.
    InvalidTolerance,
    /// Explicit dense materialization would overflow the addressable size.
    DenseSizeOverflow {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
}

impl fmt::Display for LinearOperatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyShape { rows, columns } => {
                write!(formatter, "linear operator shape {rows}x{columns} is empty")
            }
            Self::NonSquare { rows, columns } => {
                write!(
                    formatter,
                    "linear operator shape {rows}x{columns} is not square"
                )
            }
            Self::InvalidRowOffsetCount { expected, actual } => write!(
                formatter,
                "CSR storage has {actual} row offsets; expected {expected}"
            ),
            Self::NonzeroFirstRowOffset { actual } => {
                write!(formatter, "first CSR row offset is {actual}; expected zero")
            }
            Self::NonmonotoneRowOffsets { row } => {
                write!(formatter, "CSR row offsets decrease at row {row}")
            }
            Self::InvalidTerminalRowOffset { expected, actual } => write!(
                formatter,
                "terminal CSR row offset is {actual}; expected {expected}"
            ),
            Self::InvalidStoredEntryCount { indices, values } => write!(
                formatter,
                "CSR storage has {indices} column indices but {values} values"
            ),
            Self::ColumnOutOfBounds {
                row,
                column,
                columns,
            } => write!(
                formatter,
                "CSR entry ({row}, {column}) is outside a matrix with {columns} columns"
            ),
            Self::NoncanonicalRow {
                row,
                previous,
                current,
            } => write!(
                formatter,
                "CSR row {row} has non-increasing columns {previous}, {current}"
            ),
            Self::NonFiniteValue => {
                write!(formatter, "linear operator contains a non-finite value")
            }
            Self::InputDimension { expected, actual } => write!(
                formatter,
                "linear-operator input has length {actual}; expected {expected}"
            ),
            Self::OutputDimension { expected, actual } => write!(
                formatter,
                "linear-operator output has length {actual}; expected {expected}"
            ),
            Self::InvalidTolerance => write!(
                formatter,
                "operator tolerance must be finite and nonnegative"
            ),
            Self::DenseSizeOverflow { rows, columns } => write!(
                formatter,
                "cannot materialize a {rows}x{columns} linear operator"
            ),
        }
    }
}

impl Error for LinearOperatorError {}

/// A matrix-free complex linear operator.
///
/// Implementations write into caller-owned output storage so iterative
/// algorithms can reuse allocations.
pub trait LinearOperator {
    /// Number of output components.
    fn rows(&self) -> usize;

    /// Number of input components.
    fn columns(&self) -> usize;

    /// Applies the operator to `input`, replacing all entries of `output`.
    fn apply_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), LinearOperatorError>;

    /// Applies the operator and returns a newly allocated result.
    fn apply(&self, input: &[Complex64]) -> Result<Vec<Complex64>, LinearOperatorError> {
        let mut output = vec![Complex64::new(0.0, 0.0); self.rows()];
        self.apply_into(input, &mut output)?;
        Ok(output)
    }

    /// Returns the matrix shape.
    fn shape(&self) -> (usize, usize) {
        (self.rows(), self.columns())
    }
}

impl LinearOperator for ComplexMatrix {
    fn rows(&self) -> usize {
        self.rows()
    }

    fn columns(&self) -> usize {
        self.columns()
    }

    fn apply_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), LinearOperatorError> {
        validate_vectors(self.rows(), self.columns(), input, output)?;
        for (row, result) in output.iter_mut().enumerate() {
            *result = (0..self.columns())
                .map(|column| self.as_slice()[row * self.columns() + column] * input[column])
                .sum();
        }
        validate_finite_output(output)
    }
}

/// An owned canonical compressed-sparse-row complex matrix.
///
/// Column indices in every row are strictly increasing. This makes structural
/// validation, Hermiticity checks, and deterministic multiplication possible
/// without hidden normalization work.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix {
    rows: usize,
    columns: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<Complex64>,
}

impl CsrMatrix {
    /// Creates a canonical CSR matrix.
    pub fn new(
        rows: usize,
        columns: usize,
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<Complex64>,
    ) -> Result<Self, LinearOperatorError> {
        if rows == 0 || columns == 0 {
            return Err(LinearOperatorError::EmptyShape { rows, columns });
        }
        let expected_offsets = rows
            .checked_add(1)
            .ok_or(LinearOperatorError::DenseSizeOverflow { rows, columns })?;
        if row_offsets.len() != expected_offsets {
            return Err(LinearOperatorError::InvalidRowOffsetCount {
                expected: expected_offsets,
                actual: row_offsets.len(),
            });
        }
        if row_offsets[0] != 0 {
            return Err(LinearOperatorError::NonzeroFirstRowOffset {
                actual: row_offsets[0],
            });
        }
        if column_indices.len() != values.len() {
            return Err(LinearOperatorError::InvalidStoredEntryCount {
                indices: column_indices.len(),
                values: values.len(),
            });
        }
        for row in 0..rows {
            if row_offsets[row] > row_offsets[row + 1] {
                return Err(LinearOperatorError::NonmonotoneRowOffsets { row });
            }
        }
        if row_offsets[rows] != values.len() {
            return Err(LinearOperatorError::InvalidTerminalRowOffset {
                expected: values.len(),
                actual: row_offsets[rows],
            });
        }
        if values
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(LinearOperatorError::NonFiniteValue);
        }
        for row in 0..rows {
            let range = row_offsets[row]..row_offsets[row + 1];
            let mut previous = None;
            for &column in &column_indices[range] {
                if column >= columns {
                    return Err(LinearOperatorError::ColumnOutOfBounds {
                        row,
                        column,
                        columns,
                    });
                }
                if let Some(previous) = previous {
                    if column <= previous {
                        return Err(LinearOperatorError::NoncanonicalRow {
                            row,
                            previous,
                            current: column,
                        });
                    }
                }
                previous = Some(column);
            }
        }
        Ok(Self {
            rows,
            columns,
            row_offsets,
            column_indices,
            values,
        })
    }

    /// Converts a dense matrix, dropping entries no larger than `zero_tolerance`.
    pub fn from_dense(
        matrix: &ComplexMatrix,
        zero_tolerance: f64,
    ) -> Result<Self, LinearOperatorError> {
        if !zero_tolerance.is_finite() || zero_tolerance < 0.0 {
            return Err(LinearOperatorError::InvalidTolerance);
        }
        let mut row_offsets = Vec::with_capacity(matrix.rows() + 1);
        let mut column_indices = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(0);
        for row in 0..matrix.rows() {
            for column in 0..matrix.columns() {
                let value = matrix.as_slice()[row * matrix.columns() + column];
                if value.norm() > zero_tolerance {
                    column_indices.push(column);
                    values.push(value);
                }
            }
            row_offsets.push(values.len());
        }
        Self::new(
            matrix.rows(),
            matrix.columns(),
            row_offsets,
            column_indices,
            values,
        )
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Number of explicitly stored entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// CSR row offsets.
    #[must_use]
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    /// CSR column indices.
    #[must_use]
    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    /// Explicitly stored values.
    #[must_use]
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    /// Tests Hermiticity without dense materialization.
    pub fn is_hermitian(&self, tolerance: f64) -> Result<bool, LinearOperatorError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(LinearOperatorError::InvalidTolerance);
        }
        if self.rows != self.columns {
            return Ok(false);
        }
        for row in 0..self.rows {
            for entry in self.row_offsets[row]..self.row_offsets[row + 1] {
                let column = self.column_indices[entry];
                let value = self.values[entry];
                if row == column {
                    if value.im.abs() > tolerance {
                        return Ok(false);
                    }
                    continue;
                }
                let reverse = self.value_at(column, row);
                if (value - reverse.conj()).norm() > tolerance {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Returns conservative Hermitian spectral bounds from Gershgorin discs.
    pub fn gershgorin_bounds(&self) -> Result<(f64, f64), LinearOperatorError> {
        if self.rows != self.columns {
            return Err(LinearOperatorError::NonSquare {
                rows: self.rows,
                columns: self.columns,
            });
        }
        let mut lower = f64::INFINITY;
        let mut upper = f64::NEG_INFINITY;
        for row in 0..self.rows {
            let mut center = 0.0;
            let mut radius = 0.0;
            for entry in self.row_offsets[row]..self.row_offsets[row + 1] {
                let column = self.column_indices[entry];
                let value = self.values[entry];
                if row == column {
                    center = value.re;
                } else {
                    radius += value.norm();
                }
            }
            lower = lower.min(center - radius);
            upper = upper.max(center + radius);
        }
        if lower.is_finite() && upper.is_finite() {
            Ok((lower, upper))
        } else {
            Err(LinearOperatorError::NonFiniteValue)
        }
    }

    /// Explicitly materializes the sparse matrix.
    pub fn to_dense(&self) -> Result<ComplexMatrix, LinearOperatorError> {
        let entries =
            self.rows
                .checked_mul(self.columns)
                .ok_or(LinearOperatorError::DenseSizeOverflow {
                    rows: self.rows,
                    columns: self.columns,
                })?;
        let mut data = vec![Complex64::new(0.0, 0.0); entries];
        for row in 0..self.rows {
            for entry in self.row_offsets[row]..self.row_offsets[row + 1] {
                data[row * self.columns + self.column_indices[entry]] = self.values[entry];
            }
        }
        ComplexMatrix::new(self.rows, self.columns, data)
            .map_err(|_| LinearOperatorError::NonFiniteValue)
    }

    fn value_at(&self, row: usize, column: usize) -> Complex64 {
        let range = self.row_offsets[row]..self.row_offsets[row + 1];
        self.column_indices[range.clone()]
            .binary_search(&column)
            .map_or(Complex64::new(0.0, 0.0), |index| {
                self.values[range.start + index]
            })
    }
}

impl LinearOperator for CsrMatrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn apply_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), LinearOperatorError> {
        validate_vectors(self.rows, self.columns, input, output)?;
        for (row, result) in output.iter_mut().enumerate() {
            *result = (self.row_offsets[row]..self.row_offsets[row + 1])
                .map(|entry| self.values[entry] * input[self.column_indices[entry]])
                .sum();
        }
        validate_finite_output(output)
    }
}

fn validate_vectors(
    rows: usize,
    columns: usize,
    input: &[Complex64],
    output: &[Complex64],
) -> Result<(), LinearOperatorError> {
    if input.len() != columns {
        return Err(LinearOperatorError::InputDimension {
            expected: columns,
            actual: input.len(),
        });
    }
    if output.len() != rows {
        return Err(LinearOperatorError::OutputDimension {
            expected: rows,
            actual: output.len(),
        });
    }
    if input
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(LinearOperatorError::NonFiniteValue);
    }
    Ok(())
}

fn validate_finite_output(output: &[Complex64]) -> Result<(), LinearOperatorError> {
    if output
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        Err(LinearOperatorError::NonFiniteValue)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_and_dense_operators_apply_identically() {
        let dense = ComplexMatrix::new(
            3,
            3,
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-0.5, 0.0),
                Complex64::new(0.3, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.3, 0.0),
                Complex64::new(0.7, 0.0),
            ],
        )
        .unwrap();
        let sparse = CsrMatrix::from_dense(&dense, 0.0).unwrap();
        let vector = [
            Complex64::new(0.4, -0.2),
            Complex64::new(-0.1, 0.8),
            Complex64::new(0.6, 0.3),
        ];
        assert_eq!(
            sparse.apply(&vector).unwrap(),
            dense.apply(&vector).unwrap()
        );
        assert!(sparse.is_hermitian(1.0e-12).unwrap());
        assert_eq!(sparse.to_dense().unwrap(), dense);
    }

    #[test]
    fn canonical_csr_structure_is_enforced() {
        assert_eq!(
            CsrMatrix::new(
                1,
                2,
                vec![0, 2],
                vec![1, 1],
                vec![Complex64::new(1.0, 0.0); 2],
            ),
            Err(LinearOperatorError::NoncanonicalRow {
                row: 0,
                previous: 1,
                current: 1,
            })
        );
    }
}
