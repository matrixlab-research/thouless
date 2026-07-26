//! Reusable sparse direct factorization and Schur complements.
//!
//! The public boundary accepts Thouless's canonical CSR representation.  The
//! implementation performs fill-reducing symbolic analysis separately from
//! numeric LU factorization so callers can reuse structure across matrices
//! with changing values.

use std::error::Error;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};

use faer::linalg::solvers::SpSolver;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::SparseColMat;
use faer::Mat;

use crate::linear_operator::{CsrMatrix, LinearOperatorError};
use crate::{Complex64, ComplexMatrix, MatrixError};

/// Failures raised by sparse direct-solver workflows.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SparseDirectError {
    /// Direct factorization requires a square matrix.
    NonSquare {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// A matrix does not have the structure used during symbolic analysis.
    StructureMismatch,
    /// The sparse numerical factorization failed.
    FactorizationFailed,
    /// A right-hand side has the wrong row count.
    RightHandSideRows {
        /// Required rows.
        expected: usize,
        /// Supplied rows.
        actual: usize,
    },
    /// A selected Schur-complement index lies outside the matrix.
    SelectionOutOfBounds {
        /// Invalid index.
        index: usize,
        /// Matrix dimension.
        dimension: usize,
    },
    /// A Schur-complement index occurs more than once.
    DuplicateSelection {
        /// Duplicated index.
        index: usize,
    },
    /// Canonical sparse-matrix validation failed.
    Operator(LinearOperatorError),
    /// Dense right-hand-side or result construction failed.
    Matrix(MatrixError),
}

impl fmt::Display for SparseDirectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonSquare { rows, columns } => {
                write!(
                    formatter,
                    "sparse matrix shape {rows}x{columns} is not square"
                )
            }
            Self::StructureMismatch => write!(
                formatter,
                "matrix sparsity differs from the symbolic LU analysis"
            ),
            Self::FactorizationFailed => {
                write!(formatter, "sparse numerical LU factorization failed")
            }
            Self::RightHandSideRows { expected, actual } => write!(
                formatter,
                "right-hand side has {actual} rows; expected {expected}"
            ),
            Self::SelectionOutOfBounds { index, dimension } => write!(
                formatter,
                "Schur-complement index {index} is outside dimension {dimension}"
            ),
            Self::DuplicateSelection { index } => {
                write!(formatter, "Schur-complement index {index} is duplicated")
            }
            Self::Operator(error) => error.fmt(formatter),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl Error for SparseDirectError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Operator(error) => Some(error),
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LinearOperatorError> for SparseDirectError {
    fn from(error: LinearOperatorError) -> Self {
        Self::Operator(error)
    }
}

impl From<MatrixError> for SparseDirectError {
    fn from(error: MatrixError) -> Self {
        Self::Matrix(error)
    }
}

/// Fill-reducing symbolic LU analysis for one canonical sparsity pattern.
#[derive(Clone, Debug)]
pub struct SparseLuAnalysis {
    dimension: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    symbolic: SymbolicLu<usize>,
}

impl SparseLuAnalysis {
    /// Analyzes the sparsity pattern of a square matrix.
    pub fn analyze(matrix: &CsrMatrix) -> Result<Self, SparseDirectError> {
        validate_square(matrix)?;
        let backend = backend_matrix(matrix)?;
        let symbolic = SymbolicLu::try_new(backend.symbolic())
            .map_err(|_| SparseDirectError::FactorizationFailed)?;
        Ok(Self {
            dimension: matrix.rows(),
            row_offsets: matrix.row_offsets().to_vec(),
            column_indices: matrix.column_indices().to_vec(),
            symbolic,
        })
    }

    /// Numerically factors a matrix with this analyzed structure.
    pub fn factor(&self, matrix: &CsrMatrix) -> Result<SparseLuFactorization, SparseDirectError> {
        validate_square(matrix)?;
        if matrix.rows() != self.dimension
            || matrix.row_offsets() != self.row_offsets
            || matrix.column_indices() != self.column_indices
        {
            return Err(SparseDirectError::StructureMismatch);
        }
        let backend = backend_matrix(matrix)?;
        let factor = catch_unwind(AssertUnwindSafe(|| {
            Lu::try_new_with_symbolic(self.symbolic.clone(), backend.as_ref())
        }))
        .map_err(|_| SparseDirectError::FactorizationFailed)?
        .map_err(|_| SparseDirectError::FactorizationFailed)?;
        Ok(SparseLuFactorization {
            dimension: self.dimension,
            input_nonzeros: matrix.nnz(),
            factor,
        })
    }

    /// Matrix dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of entries in the analyzed input pattern.
    #[must_use]
    pub fn input_nonzeros(&self) -> usize {
        self.column_indices.len()
    }
}

/// Reusable numerical sparse LU factorization.
#[derive(Clone, Debug)]
pub struct SparseLuFactorization {
    dimension: usize,
    input_nonzeros: usize,
    factor: Lu<usize, Complex64>,
}

impl SparseLuFactorization {
    /// Analyzes and factors a square sparse matrix.
    pub fn factor(matrix: &CsrMatrix) -> Result<Self, SparseDirectError> {
        SparseLuAnalysis::analyze(matrix)?.factor(matrix)
    }

    /// Solves for one or more dense right-hand-side columns.
    pub fn solve(
        &self,
        right_hand_side: &ComplexMatrix,
    ) -> Result<ComplexMatrix, SparseDirectError> {
        if right_hand_side.rows() != self.dimension {
            return Err(SparseDirectError::RightHandSideRows {
                expected: self.dimension,
                actual: right_hand_side.rows(),
            });
        }
        let backend = Mat::from_fn(
            right_hand_side.rows(),
            right_hand_side.columns(),
            |row, column| right_hand_side.as_slice()[row * right_hand_side.columns() + column],
        );
        let solution = self.factor.solve(backend.as_ref());
        let rows = solution.nrows();
        let columns = solution.ncols();
        let mut values = Vec::with_capacity(rows * columns);
        for row in 0..rows {
            for column in 0..columns {
                values.push(solution.read(row, column));
            }
        }
        ComplexMatrix::new(rows, columns, values).map_err(Into::into)
    }

    /// Matrix dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of entries in the factored input matrix.
    #[must_use]
    pub const fn input_nonzeros(&self) -> usize {
        self.input_nonzeros
    }
}

/// Computes the principal Schur complement retained on `selected` indices.
///
/// For the partition `A = [[Aee, Aes], [Ase, Ass]]`, where `s` denotes
/// selected indices, this returns `Ass - Ase Aee⁻¹ Aes` in the caller's
/// selected-index order.
pub fn schur_complement(
    matrix: &CsrMatrix,
    selected: &[usize],
) -> Result<ComplexMatrix, SparseDirectError> {
    validate_square(matrix)?;
    let dimension = matrix.rows();
    let mut is_selected = vec![false; dimension];
    for &index in selected {
        if index >= dimension {
            return Err(SparseDirectError::SelectionOutOfBounds { index, dimension });
        }
        if std::mem::replace(&mut is_selected[index], true) {
            return Err(SparseDirectError::DuplicateSelection { index });
        }
    }
    if selected.is_empty() {
        return ComplexMatrix::new(0, 0, Vec::new()).map_err(Into::into);
    }

    let eliminated = (0..dimension)
        .filter(|index| !is_selected[*index])
        .collect::<Vec<_>>();
    let mut result = selected_dense_block(matrix, selected, selected)?;
    if eliminated.is_empty() {
        return Ok(result);
    }

    let eliminated_matrix = principal_sparse_block(matrix, &eliminated)?;
    let coupling = selected_dense_block(matrix, &eliminated, selected)?;
    let solved = SparseLuFactorization::factor(&eliminated_matrix)?.solve(&coupling)?;
    for (selected_row, &source_row) in selected.iter().enumerate() {
        for selected_column in 0..selected.len() {
            let correction = eliminated
                .iter()
                .enumerate()
                .map(|(inner, source_column)| {
                    csr_value(matrix, source_row, *source_column)
                        * solved.as_slice()[inner * selected.len() + selected_column]
                })
                .sum::<Complex64>();
            let index = selected_row * selected.len() + selected_column;
            let value = result.as_slice()[index] - correction;
            result.set(selected_row, selected_column, value)?;
        }
    }
    Ok(result)
}

fn validate_square(matrix: &CsrMatrix) -> Result<(), SparseDirectError> {
    if matrix.rows() != matrix.columns() {
        return Err(SparseDirectError::NonSquare {
            rows: matrix.rows(),
            columns: matrix.columns(),
        });
    }
    Ok(())
}

fn backend_matrix(matrix: &CsrMatrix) -> Result<SparseColMat<usize, Complex64>, SparseDirectError> {
    let mut triplets = Vec::with_capacity(matrix.nnz());
    for row in 0..matrix.rows() {
        for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
            triplets.push((
                isize::try_from(row).map_err(|_| SparseDirectError::FactorizationFailed)?,
                isize::try_from(matrix.column_indices()[entry])
                    .map_err(|_| SparseDirectError::FactorizationFailed)?,
                matrix.values()[entry],
            ));
        }
    }
    SparseColMat::<usize, Complex64>::try_new_from_nonnegative_triplets(
        matrix.rows(),
        matrix.columns(),
        &triplets,
    )
    .map_err(|_| SparseDirectError::FactorizationFailed)
}

fn principal_sparse_block(
    matrix: &CsrMatrix,
    indices: &[usize],
) -> Result<CsrMatrix, SparseDirectError> {
    let mut local_index = vec![None; matrix.rows()];
    for (local, &source) in indices.iter().enumerate() {
        local_index[source] = Some(local);
    }
    let mut row_offsets = Vec::with_capacity(indices.len() + 1);
    let mut column_indices = Vec::new();
    let mut values = Vec::new();
    row_offsets.push(0);
    for &source_row in indices {
        for entry in matrix.row_offsets()[source_row]..matrix.row_offsets()[source_row + 1] {
            if let Some(column) = local_index[matrix.column_indices()[entry]] {
                column_indices.push(column);
                values.push(matrix.values()[entry]);
            }
        }
        row_offsets.push(values.len());
    }
    CsrMatrix::new(
        indices.len(),
        indices.len(),
        row_offsets,
        column_indices,
        values,
    )
    .map_err(Into::into)
}

fn selected_dense_block(
    matrix: &CsrMatrix,
    rows: &[usize],
    columns: &[usize],
) -> Result<ComplexMatrix, SparseDirectError> {
    let values = rows
        .iter()
        .flat_map(|row| {
            columns
                .iter()
                .map(|column| csr_value(matrix, *row, *column))
        })
        .collect();
    ComplexMatrix::new(rows.len(), columns.len(), values).map_err(Into::into)
}

fn csr_value(matrix: &CsrMatrix, row: usize, column: usize) -> Complex64 {
    let range = matrix.row_offsets()[row]..matrix.row_offsets()[row + 1];
    matrix.column_indices()[range.clone()]
        .binary_search(&column)
        .map_or(Complex64::new(0.0, 0.0), |index| {
            matrix.values()[range.start + index]
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_operator::LinearOperator;

    fn residual_norm(
        matrix: &CsrMatrix,
        solution: &ComplexMatrix,
        right_hand_side: &ComplexMatrix,
    ) -> f64 {
        let mut norm = 0.0_f64;
        for column in 0..solution.columns() {
            let vector = (0..solution.rows())
                .map(|row| solution.as_slice()[row * solution.columns() + column])
                .collect::<Vec<_>>();
            let applied = matrix.apply(&vector).unwrap();
            for (row, applied_value) in applied.iter().enumerate() {
                norm += (*applied_value
                    - right_hand_side.as_slice()[row * solution.columns() + column])
                    .norm_sqr();
            }
        }
        norm.sqrt()
    }

    #[test]
    fn sparse_lu_solves_multiple_complex_right_hand_sides() {
        let matrix = CsrMatrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![
                Complex64::new(3.0, 0.5),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.2, -0.1),
                Complex64::new(2.5, 0.0),
                Complex64::new(0.4, 0.3),
                Complex64::new(-0.6, 0.0),
                Complex64::new(1.8, -0.2),
            ],
        )
        .unwrap();
        let right_hand_side = ComplexMatrix::new(
            3,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(2.0, -0.5),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.3, 0.2),
                Complex64::new(1.5, -0.7),
            ],
        )
        .unwrap();
        let factorization = SparseLuFactorization::factor(&matrix).unwrap();
        let solution = factorization.solve(&right_hand_side).unwrap();
        assert!(residual_norm(&matrix, &solution, &right_hand_side) < 1.0e-12);
    }

    #[test]
    fn symbolic_analysis_is_reusable_but_structure_checked() {
        let matrix = CsrMatrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![2.0.into(), 0.5.into(), 0.25.into(), 3.0.into()],
        )
        .unwrap();
        let changed = CsrMatrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![4.0.into(), 0.5.into(), 0.25.into(), 6.0.into()],
        )
        .unwrap();
        let analysis = SparseLuAnalysis::analyze(&matrix).unwrap();
        let factorization = analysis.factor(&changed).unwrap();
        let right_hand_side = ComplexMatrix::new(2, 1, vec![Complex64::new(1.0, 0.0); 2]).unwrap();
        let solution = factorization.solve(&right_hand_side).unwrap();
        assert!(residual_norm(&changed, &solution, &right_hand_side) < 1.0e-12);

        let diagonal = CsrMatrix::new(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            vec![1.0.into(), 1.0.into()],
        )
        .unwrap();
        assert_eq!(
            analysis.factor(&diagonal).unwrap_err(),
            SparseDirectError::StructureMismatch
        );
    }

    #[test]
    fn numerical_singularity_is_a_recoverable_error() {
        let singular = CsrMatrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0.into(), 2.0.into(), 2.0.into(), 4.0.into()],
        )
        .unwrap();
        assert_eq!(
            SparseLuFactorization::factor(&singular).unwrap_err(),
            SparseDirectError::FactorizationFailed
        );
    }

    #[test]
    fn sparse_lu_handles_a_large_tridiagonal_system() {
        let dimension = 4096;
        let mut row_offsets = Vec::with_capacity(dimension + 1);
        let mut column_indices = Vec::with_capacity(3 * dimension - 2);
        let mut values = Vec::with_capacity(3 * dimension - 2);
        row_offsets.push(0);
        for row in 0..dimension {
            if row > 0 {
                column_indices.push(row - 1);
                values.push(Complex64::new(-1.0, 0.05));
            }
            column_indices.push(row);
            values.push(Complex64::new(4.0, 0.1));
            if row + 1 < dimension {
                column_indices.push(row + 1);
                values.push(Complex64::new(-1.0, -0.05));
            }
            row_offsets.push(values.len());
        }
        let matrix =
            CsrMatrix::new(dimension, dimension, row_offsets, column_indices, values).unwrap();
        let expected = (0..dimension)
            .map(|index| {
                let argument = index as f64 * 0.017;
                Complex64::new(argument.sin(), argument.cos())
            })
            .collect::<Vec<_>>();
        let right_hand_side =
            ComplexMatrix::new(dimension, 1, matrix.apply(&expected).unwrap()).unwrap();
        let solution = SparseLuFactorization::factor(&matrix)
            .unwrap()
            .solve(&right_hand_side)
            .unwrap();
        let relative_error = solution
            .as_slice()
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr())
            .sum::<f64>()
            .sqrt()
            / expected
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt();
        assert!(relative_error < 1.0e-12);
    }

    #[test]
    fn schur_complement_obeys_the_block_elimination_identity() {
        let matrix = CsrMatrix::from_dense(
            &ComplexMatrix::new(
                4,
                4,
                vec![
                    4.0.into(),
                    1.0.into(),
                    0.5.into(),
                    0.0.into(),
                    0.2.into(),
                    3.0.into(),
                    (-0.4).into(),
                    0.1.into(),
                    1.0.into(),
                    0.0.into(),
                    2.0.into(),
                    0.3.into(),
                    0.0.into(),
                    0.5.into(),
                    (-0.2).into(),
                    2.5.into(),
                ],
            )
            .unwrap(),
            0.0,
        )
        .unwrap();
        let complement = schur_complement(&matrix, &[2, 0]).unwrap();

        let eliminated =
            ComplexMatrix::new(2, 2, vec![3.0.into(), 0.1.into(), 0.5.into(), 2.5.into()]).unwrap();
        let coupling = ComplexMatrix::new(
            2,
            2,
            vec![(-0.4).into(), 0.2.into(), (-0.2).into(), 0.0.into()],
        )
        .unwrap();
        let solved =
            SparseLuFactorization::factor(&CsrMatrix::from_dense(&eliminated, 0.0).unwrap())
                .unwrap()
                .solve(&coupling)
                .unwrap();
        let expected = [
            Complex64::new(2.0, 0.0)
                - Complex64::new(0.0, 0.0) * solved.as_slice()[0]
                - Complex64::new(0.3, 0.0) * solved.as_slice()[2],
            Complex64::new(1.0, 0.0)
                - Complex64::new(0.0, 0.0) * solved.as_slice()[1]
                - Complex64::new(0.3, 0.0) * solved.as_slice()[3],
            Complex64::new(0.5, 0.0) - Complex64::new(1.0, 0.0) * solved.as_slice()[0],
            Complex64::new(4.0, 0.0) - Complex64::new(1.0, 0.0) * solved.as_slice()[1],
        ];
        assert!(complement
            .as_slice()
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (*actual - expected).norm() < 1.0e-12));
    }
}
