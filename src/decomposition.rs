//! Dense Schur and generalized Schur decompositions.
//!
//! The native API uses one complex representation for real and complex input.
//! This avoids quasi-triangular special cases while retaining unitary Schur
//! vectors and stable eigenvalue reordering.

use std::fmt;

use crate::{Complex64, ComplexMatrix, MatrixError};

/// Failures raised by dense decomposition workflows.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecompositionError {
    /// A matrix is not square.
    NotSquare {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// Matrices in one decomposition do not share a dimension.
    DimensionMismatch,
    /// A selection mask does not contain one value per eigenvalue.
    InvalidSelectionLength {
        /// Required number of values.
        expected: usize,
        /// Supplied number of values.
        actual: usize,
    },
    /// The numerical backend failed.
    BackendFailure,
    /// Matrix construction failed.
    Matrix(MatrixError),
}

impl fmt::Display for DecompositionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSquare { rows, columns } => {
                write!(formatter, "matrix shape ({rows}, {columns}) is not square")
            }
            Self::DimensionMismatch => {
                write!(
                    formatter,
                    "decomposition matrices have incompatible dimensions"
                )
            }
            Self::InvalidSelectionLength { expected, actual } => write!(
                formatter,
                "selection has {actual} values; expected {expected}"
            ),
            Self::BackendFailure => write!(formatter, "dense decomposition backend failed"),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DecompositionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatrixError> for DecompositionError {
    fn from(error: MatrixError) -> Self {
        Self::Matrix(error)
    }
}

/// Complex Schur representation `A = Q T Qᴴ`.
#[derive(Clone, Debug, PartialEq)]
pub struct SchurDecomposition {
    form: ComplexMatrix,
    vectors: ComplexMatrix,
    eigenvalues: Vec<Complex64>,
}

impl SchurDecomposition {
    /// Upper-triangular Schur form.
    #[must_use]
    pub const fn form(&self) -> &ComplexMatrix {
        &self.form
    }

    /// Unitary Schur vectors.
    #[must_use]
    pub const fn vectors(&self) -> &ComplexMatrix {
        &self.vectors
    }

    /// Eigenvalues in diagonal order.
    #[must_use]
    pub fn eigenvalues(&self) -> &[Complex64] {
        &self.eigenvalues
    }
}

/// Complex generalized Schur representation
/// `A = Q S Zᴴ`, `B = Q T Zᴴ`.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralizedSchurDecomposition {
    left_form: ComplexMatrix,
    right_form: ComplexMatrix,
    left_vectors: ComplexMatrix,
    right_vectors: ComplexMatrix,
    alpha: Vec<Complex64>,
    beta: Vec<Complex64>,
}

impl GeneralizedSchurDecomposition {
    /// First triangular form `S`.
    #[must_use]
    pub const fn left_form(&self) -> &ComplexMatrix {
        &self.left_form
    }

    /// Second triangular form `T`.
    #[must_use]
    pub const fn right_form(&self) -> &ComplexMatrix {
        &self.right_form
    }

    /// Left unitary vectors `Q`.
    #[must_use]
    pub const fn left_vectors(&self) -> &ComplexMatrix {
        &self.left_vectors
    }

    /// Right unitary vectors `Z`.
    #[must_use]
    pub const fn right_vectors(&self) -> &ComplexMatrix {
        &self.right_vectors
    }

    /// Generalized eigenvalue numerators.
    #[must_use]
    pub fn alpha(&self) -> &[Complex64] {
        &self.alpha
    }

    /// Generalized eigenvalue denominators.
    #[must_use]
    pub fn beta(&self) -> &[Complex64] {
        &self.beta
    }
}

/// Selected left and right eigenvectors.
#[derive(Clone, Debug, PartialEq)]
pub struct EigenvectorSet {
    left: Option<ComplexMatrix>,
    right: Option<ComplexMatrix>,
}

impl EigenvectorSet {
    /// Left eigenvectors as selected columns.
    #[must_use]
    pub const fn left(&self) -> Option<&ComplexMatrix> {
        self.left.as_ref()
    }

    /// Right eigenvectors as selected columns.
    #[must_use]
    pub const fn right(&self) -> Option<&ComplexMatrix> {
        self.right.as_ref()
    }
}

/// Compute a complex Schur decomposition.
pub fn schur(matrix: &ComplexMatrix) -> Result<SchurDecomposition, DecompositionError> {
    let dimension = square_dimension(matrix)?;
    let result = thouless_lapack::complex_schur(dimension, matrix.as_slice())
        .map_err(|_| DecompositionError::BackendFailure)?;
    schur_from_backend(dimension, result)
}

/// Reorder a Schur decomposition so selected eigenvalues occur first.
pub fn reorder_schur(
    form: &ComplexMatrix,
    vectors: &ComplexMatrix,
    selected: &[bool],
) -> Result<SchurDecomposition, DecompositionError> {
    let dimension = matching_square_dimensions(&[form, vectors])?;
    validate_selection(dimension, selected)?;
    let result = thouless_lapack::reorder_complex_schur(
        dimension,
        form.as_slice(),
        vectors.as_slice(),
        selected,
    )
    .map_err(|_| DecompositionError::BackendFailure)?;
    schur_from_backend(dimension, result)
}

/// Compute selected left and/or right eigenvectors from a Schur form.
pub fn eigenvectors_from_schur(
    form: &ComplexMatrix,
    vectors: &ComplexMatrix,
    selected: &[bool],
    compute_left: bool,
    compute_right: bool,
) -> Result<EigenvectorSet, DecompositionError> {
    let dimension = matching_square_dimensions(&[form, vectors])?;
    validate_selection(dimension, selected)?;
    let result = thouless_lapack::complex_schur_eigenvectors(
        dimension,
        form.as_slice(),
        vectors.as_slice(),
        selected,
        compute_left,
        compute_right,
    )
    .map_err(|_| DecompositionError::BackendFailure)?;
    eigenvectors_from_backend(dimension, result)
}

/// Convert an arbitrary real or complex Schur representation to a fully
/// triangular complex representation by reconstructing the represented matrix
/// and decomposing it through the native backend.
pub fn complexify_schur(
    form: &ComplexMatrix,
    vectors: &ComplexMatrix,
) -> Result<SchurDecomposition, DecompositionError> {
    matching_square_dimensions(&[form, vectors])?;
    let represented = multiply(&multiply(vectors, form)?, &vectors.adjoint())?;
    schur(&represented)
}

/// Compute a complex generalized Schur decomposition of `(left, right)`.
pub fn generalized_schur(
    left: &ComplexMatrix,
    right: &ComplexMatrix,
) -> Result<GeneralizedSchurDecomposition, DecompositionError> {
    let dimension = matching_square_dimensions(&[left, right])?;
    let result =
        thouless_lapack::generalized_complex_schur(dimension, left.as_slice(), right.as_slice())
            .map_err(|_| DecompositionError::BackendFailure)?;
    generalized_schur_from_backend(dimension, result)
}

/// Reorder a generalized Schur form so selected eigenvalues occur first.
pub fn reorder_generalized_schur(
    left_form: &ComplexMatrix,
    right_form: &ComplexMatrix,
    left_vectors: &ComplexMatrix,
    right_vectors: &ComplexMatrix,
    selected: &[bool],
) -> Result<GeneralizedSchurDecomposition, DecompositionError> {
    let dimension =
        matching_square_dimensions(&[left_form, right_form, left_vectors, right_vectors])?;
    validate_selection(dimension, selected)?;
    let result = thouless_lapack::reorder_generalized_complex_schur(
        dimension,
        left_form.as_slice(),
        right_form.as_slice(),
        left_vectors.as_slice(),
        right_vectors.as_slice(),
        selected,
    )
    .map_err(|_| DecompositionError::BackendFailure)?;
    generalized_schur_from_backend(dimension, result)
}

/// Compute selected generalized left and/or right eigenvectors.
pub fn eigenvectors_from_generalized_schur(
    left_form: &ComplexMatrix,
    right_form: &ComplexMatrix,
    left_vectors: &ComplexMatrix,
    right_vectors: &ComplexMatrix,
    selected: &[bool],
    compute_left: bool,
    compute_right: bool,
) -> Result<EigenvectorSet, DecompositionError> {
    let dimension =
        matching_square_dimensions(&[left_form, right_form, left_vectors, right_vectors])?;
    validate_selection(dimension, selected)?;
    let result = thouless_lapack::generalized_complex_schur_eigenvectors(
        dimension,
        left_form.as_slice(),
        right_form.as_slice(),
        left_vectors.as_slice(),
        right_vectors.as_slice(),
        selected,
        match (compute_left, compute_right) {
            (false, false) => thouless_lapack::EigenvectorSides::Neither,
            (true, false) => thouless_lapack::EigenvectorSides::Left,
            (false, true) => thouless_lapack::EigenvectorSides::Right,
            (true, true) => thouless_lapack::EigenvectorSides::Both,
        },
    )
    .map_err(|_| DecompositionError::BackendFailure)?;
    eigenvectors_from_backend(dimension, result)
}

/// Convert an arbitrary real or complex generalized Schur representation into
/// a fully triangular complex representation.
pub fn complexify_generalized_schur(
    left_form: &ComplexMatrix,
    right_form: &ComplexMatrix,
    left_vectors: &ComplexMatrix,
    right_vectors: &ComplexMatrix,
) -> Result<GeneralizedSchurDecomposition, DecompositionError> {
    matching_square_dimensions(&[left_form, right_form, left_vectors, right_vectors])?;
    let right_adjoint = right_vectors.adjoint();
    let left = multiply(&multiply(left_vectors, left_form)?, &right_adjoint)?;
    let right = multiply(&multiply(left_vectors, right_form)?, &right_adjoint)?;
    generalized_schur(&left, &right)
}

fn schur_from_backend(
    dimension: usize,
    result: thouless_lapack::ComplexSchur,
) -> Result<SchurDecomposition, DecompositionError> {
    Ok(SchurDecomposition {
        form: ComplexMatrix::new(dimension, dimension, result.form_row_major().to_vec())?,
        vectors: ComplexMatrix::new(dimension, dimension, result.vectors_row_major().to_vec())?,
        eigenvalues: result.eigenvalues().to_vec(),
    })
}

fn generalized_schur_from_backend(
    dimension: usize,
    result: thouless_lapack::GeneralizedComplexSchur,
) -> Result<GeneralizedSchurDecomposition, DecompositionError> {
    Ok(GeneralizedSchurDecomposition {
        left_form: ComplexMatrix::new(dimension, dimension, result.left_form_row_major().to_vec())?,
        right_form: ComplexMatrix::new(
            dimension,
            dimension,
            result.right_form_row_major().to_vec(),
        )?,
        left_vectors: ComplexMatrix::new(
            dimension,
            dimension,
            result.left_vectors_row_major().to_vec(),
        )?,
        right_vectors: ComplexMatrix::new(
            dimension,
            dimension,
            result.right_vectors_row_major().to_vec(),
        )?,
        alpha: result.alpha().to_vec(),
        beta: result.beta().to_vec(),
    })
}

fn eigenvectors_from_backend(
    dimension: usize,
    result: thouless_lapack::ComplexEigenvectors,
) -> Result<EigenvectorSet, DecompositionError> {
    let columns = result.selected_count();
    Ok(EigenvectorSet {
        left: result
            .left_row_major()
            .map(|values| ComplexMatrix::new(dimension, columns, values.to_vec()))
            .transpose()?,
        right: result
            .right_row_major()
            .map(|values| ComplexMatrix::new(dimension, columns, values.to_vec()))
            .transpose()?,
    })
}

fn square_dimension(matrix: &ComplexMatrix) -> Result<usize, DecompositionError> {
    if matrix.rows() != matrix.columns() {
        return Err(DecompositionError::NotSquare {
            rows: matrix.rows(),
            columns: matrix.columns(),
        });
    }
    Ok(matrix.rows())
}

fn matching_square_dimensions(matrices: &[&ComplexMatrix]) -> Result<usize, DecompositionError> {
    let Some(first) = matrices.first() else {
        return Ok(0);
    };
    let dimension = square_dimension(first)?;
    for matrix in &matrices[1..] {
        if square_dimension(matrix)? != dimension {
            return Err(DecompositionError::DimensionMismatch);
        }
    }
    Ok(dimension)
}

fn validate_selection(dimension: usize, selected: &[bool]) -> Result<(), DecompositionError> {
    if selected.len() != dimension {
        return Err(DecompositionError::InvalidSelectionLength {
            expected: dimension,
            actual: selected.len(),
        });
    }
    Ok(())
}

fn multiply(
    left: &ComplexMatrix,
    right: &ComplexMatrix,
) -> Result<ComplexMatrix, DecompositionError> {
    if left.columns() != right.rows() {
        return Err(DecompositionError::DimensionMismatch);
    }
    let mut values = vec![Complex64::new(0.0, 0.0); left.rows() * right.columns()];
    for row in 0..left.rows() {
        for column in 0..right.columns() {
            values[row * right.columns() + column] = (0..left.columns())
                .map(|inner| {
                    left.as_slice()[row * left.columns() + inner]
                        * right.as_slice()[inner * right.columns() + column]
                })
                .sum();
        }
    }
    ComplexMatrix::new(left.rows(), right.columns(), values).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residual_norm(left: &ComplexMatrix, right: &ComplexMatrix) -> f64 {
        left.as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(left, right)| (*left - *right).norm_sqr())
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn reorder_preserves_a_nonnormal_matrix_and_selected_subspace() {
        let matrix = ComplexMatrix::new(
            4,
            4,
            vec![
                1.0.into(),
                Complex64::new(2.0, 0.5),
                0.0.into(),
                (-1.0).into(),
                0.0.into(),
                3.0.into(),
                2.0.into(),
                0.0.into(),
                0.5.into(),
                0.0.into(),
                (-2.0).into(),
                1.0.into(),
                0.0.into(),
                Complex64::new(0.2, -0.4),
                0.0.into(),
                4.0.into(),
            ],
        )
        .unwrap();
        let initial = schur(&matrix).unwrap();
        let reordered = reorder_schur(
            initial.form(),
            initial.vectors(),
            &[false, true, true, false],
        )
        .unwrap();
        let reconstructed = multiply(
            &multiply(reordered.vectors(), reordered.form()).unwrap(),
            &reordered.vectors().adjoint(),
        )
        .unwrap();
        assert!(residual_norm(&reconstructed, &matrix) < 1.0e-10);
    }

    #[test]
    fn left_and_right_vectors_satisfy_the_eigenvalue_equations() {
        let matrix = ComplexMatrix::new(
            3,
            3,
            vec![
                1.0.into(),
                2.0.into(),
                0.0.into(),
                0.0.into(),
                3.0.into(),
                1.0.into(),
                0.5.into(),
                0.0.into(),
                (-2.0).into(),
            ],
        )
        .unwrap();
        let decomposition = schur(&matrix).unwrap();
        let vectors = eigenvectors_from_schur(
            decomposition.form(),
            decomposition.vectors(),
            &[true, true, true],
            true,
            true,
        )
        .unwrap();
        for (column, eigenvalue) in decomposition.eigenvalues().iter().enumerate() {
            let right = vectors.right().unwrap();
            for row in 0..3 {
                let applied = (0..3)
                    .map(|inner| {
                        matrix.as_slice()[row * 3 + inner] * right.as_slice()[inner * 3 + column]
                    })
                    .sum::<Complex64>();
                assert!(
                    (applied - eigenvalue * right.as_slice()[row * 3 + column]).norm() < 1.0e-10
                );
            }
        }
    }

    #[test]
    fn generalized_reordering_and_eigenvectors_preserve_a_matrix_pencil() {
        let left = ComplexMatrix::new(
            3,
            3,
            vec![
                1.0.into(),
                Complex64::new(2.0, 0.3),
                0.0.into(),
                0.5.into(),
                3.0.into(),
                (-1.0).into(),
                Complex64::new(0.2, -0.1),
                0.0.into(),
                2.0.into(),
            ],
        )
        .unwrap();
        let right = ComplexMatrix::new(
            3,
            3,
            vec![
                2.0.into(),
                0.1.into(),
                0.0.into(),
                0.0.into(),
                1.5.into(),
                Complex64::new(0.0, 0.2),
                0.1.into(),
                0.0.into(),
                1.0.into(),
            ],
        )
        .unwrap();
        let decomposition = generalized_schur(&left, &right).unwrap();
        let reordered = reorder_generalized_schur(
            decomposition.left_form(),
            decomposition.right_form(),
            decomposition.left_vectors(),
            decomposition.right_vectors(),
            &[false, true, true],
        )
        .unwrap();
        let left_reconstructed = multiply(
            &multiply(reordered.left_vectors(), reordered.left_form()).unwrap(),
            &reordered.right_vectors().adjoint(),
        )
        .unwrap();
        let right_reconstructed = multiply(
            &multiply(reordered.left_vectors(), reordered.right_form()).unwrap(),
            &reordered.right_vectors().adjoint(),
        )
        .unwrap();
        assert!(residual_norm(&left_reconstructed, &left) < 1.0e-10);
        assert!(residual_norm(&right_reconstructed, &right) < 1.0e-10);

        let vectors = eigenvectors_from_generalized_schur(
            decomposition.left_form(),
            decomposition.right_form(),
            decomposition.left_vectors(),
            decomposition.right_vectors(),
            &[true, true, true],
            true,
            true,
        )
        .unwrap();
        for column in 0..3 {
            for row in 0..3 {
                let right_vector = vectors.right().unwrap();
                let left_applied = (0..3)
                    .map(|inner| {
                        left.as_slice()[row * 3 + inner]
                            * right_vector.as_slice()[inner * 3 + column]
                    })
                    .sum::<Complex64>()
                    * decomposition.beta()[column];
                let right_applied = (0..3)
                    .map(|inner| {
                        right.as_slice()[row * 3 + inner]
                            * right_vector.as_slice()[inner * 3 + column]
                    })
                    .sum::<Complex64>()
                    * decomposition.alpha()[column];
                assert!((left_applied - right_applied).norm() < 1.0e-10);
            }
            for output_column in 0..3 {
                let left_vector = vectors.left().unwrap();
                let left_applied = (0..3)
                    .map(|inner| {
                        left_vector.as_slice()[inner * 3 + column].conj()
                            * left.as_slice()[inner * 3 + output_column]
                    })
                    .sum::<Complex64>()
                    * decomposition.beta()[column];
                let right_applied = (0..3)
                    .map(|inner| {
                        left_vector.as_slice()[inner * 3 + column].conj()
                            * right.as_slice()[inner * 3 + output_column]
                    })
                    .sum::<Complex64>()
                    * decomposition.alpha()[column];
                assert!((left_applied - right_applied).norm() < 1.0e-10);
            }
        }
    }
}
