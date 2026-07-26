//! Dense Schur and generalized Schur decompositions.
//!
//! Real inputs retain LAPACK's real quasi-triangular representation, including
//! 2-by-2 blocks for complex-conjugate eigenvalue pairs.  Complex inputs use
//! triangular forms.  Explicit conversion routines preserve eigenvalue order
//! when a caller needs to split a real conjugate pair.

use std::fmt;

use crate::{Complex64, ComplexMatrix, MatrixError, RealMatrix};

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
    /// A requested real-form reordering separates a conjugate-pair block.
    SplitConjugatePair {
        /// First index of the 2-by-2 block.
        index: usize,
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
            Self::SplitConjugatePair { index } => write!(
                formatter,
                "selection separates the real Schur block at indices {index} and {}",
                index + 1
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

/// Real Schur representation `A = Q T Qᵀ`.
#[derive(Clone, Debug, PartialEq)]
pub struct RealSchurDecomposition {
    form: RealMatrix,
    vectors: RealMatrix,
    eigenvalues: Vec<Complex64>,
}

impl RealSchurDecomposition {
    /// Quasi-upper-triangular Schur form.
    #[must_use]
    pub const fn form(&self) -> &RealMatrix {
        &self.form
    }

    /// Orthogonal Schur vectors.
    #[must_use]
    pub const fn vectors(&self) -> &RealMatrix {
        &self.vectors
    }

    /// Eigenvalues in Schur block order.
    #[must_use]
    pub fn eigenvalues(&self) -> &[Complex64] {
        &self.eigenvalues
    }
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

/// Real generalized Schur representation
/// `A = Q S Zᵀ`, `B = Q T Zᵀ`.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralizedRealSchurDecomposition {
    left_form: RealMatrix,
    right_form: RealMatrix,
    left_vectors: RealMatrix,
    right_vectors: RealMatrix,
    alpha: Vec<Complex64>,
    beta: Vec<f64>,
}

impl GeneralizedRealSchurDecomposition {
    /// First quasi-upper-triangular form.
    #[must_use]
    pub const fn left_form(&self) -> &RealMatrix {
        &self.left_form
    }

    /// Second upper-triangular form.
    #[must_use]
    pub const fn right_form(&self) -> &RealMatrix {
        &self.right_form
    }

    /// Left orthogonal vectors.
    #[must_use]
    pub const fn left_vectors(&self) -> &RealMatrix {
        &self.left_vectors
    }

    /// Right orthogonal vectors.
    #[must_use]
    pub const fn right_vectors(&self) -> &RealMatrix {
        &self.right_vectors
    }

    /// Generalized eigenvalue numerators.
    #[must_use]
    pub fn alpha(&self) -> &[Complex64] {
        &self.alpha
    }

    /// Real generalized eigenvalue denominators.
    #[must_use]
    pub fn beta(&self) -> &[f64] {
        &self.beta
    }
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

/// Compute a real Schur decomposition.
pub fn real_schur(matrix: &RealMatrix) -> Result<RealSchurDecomposition, DecompositionError> {
    let dimension = real_square_dimension(matrix)?;
    let result = thouless_lapack::real_schur(dimension, matrix.as_slice())
        .map_err(|_| DecompositionError::BackendFailure)?;
    real_schur_from_backend(dimension, result)
}

/// Reorder a real Schur form without separating conjugate-pair blocks.
pub fn reorder_real_schur(
    form: &RealMatrix,
    vectors: &RealMatrix,
    selected: &[bool],
) -> Result<RealSchurDecomposition, DecompositionError> {
    let dimension = matching_real_square_dimensions(&[form, vectors])?;
    validate_selection(dimension, selected)?;
    validate_real_block_selection(form, selected)?;
    let result = thouless_lapack::reorder_real_schur(
        dimension,
        form.as_slice(),
        vectors.as_slice(),
        selected,
    )
    .map_err(|_| DecompositionError::BackendFailure)?;
    real_schur_from_backend(dimension, result)
}

/// Convert a real Schur form to a triangular complex form while preserving the
/// eigenvalue order of every 2-by-2 conjugate-pair block.
pub fn complexify_real_schur(
    form: &RealMatrix,
    vectors: &RealMatrix,
) -> Result<SchurDecomposition, DecompositionError> {
    let dimension = matching_real_square_dimensions(&[form, vectors])?;
    let mut complex_form = form.to_complex().into_vec();
    let mut complex_vectors = vectors.to_complex().into_vec();
    let blocks = real_block_positions(form);

    for index in blocks {
        let a = form.as_slice()[index * dimension + index];
        let b = form.as_slice()[index * dimension + index + 1];
        let c = form.as_slice()[(index + 1) * dimension + index];
        let discriminant = -b * c;
        if discriminant <= 0.0 {
            return Err(DecompositionError::BackendFailure);
        }
        let x = Complex64::new(0.0, discriminant.sqrt());
        let y = Complex64::new(c, 0.0);
        let norm = (discriminant + c * c).sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(DecompositionError::BackendFailure);
        }
        let unitary = [x / norm, -y / norm, y / norm, -x / norm];

        complex_form[index * dimension + index] = Complex64::new(a, 0.0) + x;
        complex_form[(index + 1) * dimension + index] = Complex64::new(0.0, 0.0);
        complex_form[index * dimension + index + 1] = Complex64::new(-b - c, 0.0);
        complex_form[(index + 1) * dimension + index + 1] = Complex64::new(a, 0.0) - x;

        right_multiply_columns(&mut complex_form, dimension, 0..index, index, &unitary);
        left_multiply_rows_adjoint(
            &mut complex_form,
            dimension,
            index,
            (index + 2)..dimension,
            &unitary,
        );
        right_multiply_columns(
            &mut complex_vectors,
            dimension,
            0..dimension,
            index,
            &unitary,
        );
    }

    let form = ComplexMatrix::new(dimension, dimension, complex_form)?;
    let vectors = ComplexMatrix::new(dimension, dimension, complex_vectors)?;
    let eigenvalues = (0..dimension)
        .map(|index| form.as_slice()[index * dimension + index])
        .collect();
    Ok(SchurDecomposition {
        form,
        vectors,
        eigenvalues,
    })
}

/// Compute selected eigenvectors from a real Schur form.
///
/// Eigenvectors are complex because a real 2-by-2 block represents a
/// complex-conjugate pair.
pub fn eigenvectors_from_real_schur(
    form: &RealMatrix,
    vectors: &RealMatrix,
    selected: &[bool],
    compute_left: bool,
    compute_right: bool,
) -> Result<EigenvectorSet, DecompositionError> {
    let dimension = matching_real_square_dimensions(&[form, vectors])?;
    validate_selection(dimension, selected)?;
    let complex = complexify_real_schur(form, vectors)?;
    eigenvectors_from_schur(
        complex.form(),
        complex.vectors(),
        selected,
        compute_left,
        compute_right,
    )
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

/// Compute a real generalized Schur decomposition.
pub fn generalized_real_schur(
    left: &RealMatrix,
    right: &RealMatrix,
) -> Result<GeneralizedRealSchurDecomposition, DecompositionError> {
    let dimension = matching_real_square_dimensions(&[left, right])?;
    let result =
        thouless_lapack::generalized_real_schur(dimension, left.as_slice(), right.as_slice())
            .map_err(|_| DecompositionError::BackendFailure)?;
    generalized_real_schur_from_backend(dimension, result)
}

/// Reorder a real generalized Schur form without separating conjugate-pair
/// blocks.
pub fn reorder_generalized_real_schur(
    left_form: &RealMatrix,
    right_form: &RealMatrix,
    left_vectors: &RealMatrix,
    right_vectors: &RealMatrix,
    selected: &[bool],
) -> Result<GeneralizedRealSchurDecomposition, DecompositionError> {
    let dimension =
        matching_real_square_dimensions(&[left_form, right_form, left_vectors, right_vectors])?;
    validate_selection(dimension, selected)?;
    validate_real_block_selection(left_form, selected)?;
    let result = thouless_lapack::reorder_generalized_real_schur(
        dimension,
        left_form.as_slice(),
        right_form.as_slice(),
        left_vectors.as_slice(),
        right_vectors.as_slice(),
        selected,
    )
    .map_err(|_| DecompositionError::BackendFailure)?;
    generalized_real_schur_from_backend(dimension, result)
}

/// Convert a real generalized Schur form to complex triangular form while
/// preserving conjugate-pair order.
pub fn complexify_real_generalized_schur(
    left_form: &RealMatrix,
    right_form: &RealMatrix,
    left_vectors: &RealMatrix,
    right_vectors: &RealMatrix,
) -> Result<GeneralizedSchurDecomposition, DecompositionError> {
    let dimension =
        matching_real_square_dimensions(&[left_form, right_form, left_vectors, right_vectors])?;
    let blocks = real_block_positions(left_form);
    let mut complex_left = left_form.to_complex().into_vec();
    let mut complex_right = right_form.to_complex().into_vec();
    let mut complex_q = left_vectors.to_complex().into_vec();
    let mut complex_z = right_vectors.to_complex().into_vec();

    for index in blocks {
        let local_left = local_two_by_two(&complex_left, dimension, index);
        let local_right = local_two_by_two(&complex_right, dimension, index);
        let mut local = thouless_lapack::generalized_complex_schur(2, &local_left, &local_right)
            .map_err(|_| DecompositionError::BackendFailure)?;
        let first = local.alpha()[0] / local.beta()[0];
        let second = local.alpha()[1] / local.beta()[1];
        if first.im < second.im {
            local = thouless_lapack::reorder_generalized_complex_schur(
                2,
                local.left_form_row_major(),
                local.right_form_row_major(),
                local.left_vectors_row_major(),
                local.right_vectors_row_major(),
                &[false, true],
            )
            .map_err(|_| DecompositionError::BackendFailure)?;
        }

        let q_block = [
            local.left_vectors_row_major()[0],
            local.left_vectors_row_major()[1],
            local.left_vectors_row_major()[2],
            local.left_vectors_row_major()[3],
        ];
        let z_block = [
            local.right_vectors_row_major()[0],
            local.right_vectors_row_major()[1],
            local.right_vectors_row_major()[2],
            local.right_vectors_row_major()[3],
        ];
        store_two_by_two(
            &mut complex_left,
            dimension,
            index,
            local.left_form_row_major(),
        );
        store_two_by_two(
            &mut complex_right,
            dimension,
            index,
            local.right_form_row_major(),
        );
        right_multiply_columns(&mut complex_left, dimension, 0..index, index, &z_block);
        left_multiply_rows_adjoint(
            &mut complex_left,
            dimension,
            index,
            (index + 2)..dimension,
            &q_block,
        );
        right_multiply_columns(&mut complex_right, dimension, 0..index, index, &z_block);
        left_multiply_rows_adjoint(
            &mut complex_right,
            dimension,
            index,
            (index + 2)..dimension,
            &q_block,
        );
        right_multiply_columns(&mut complex_q, dimension, 0..dimension, index, &q_block);
        right_multiply_columns(&mut complex_z, dimension, 0..dimension, index, &z_block);
    }

    let left_form = ComplexMatrix::new(dimension, dimension, complex_left)?;
    let right_form = ComplexMatrix::new(dimension, dimension, complex_right)?;
    let left_vectors = ComplexMatrix::new(dimension, dimension, complex_q)?;
    let right_vectors = ComplexMatrix::new(dimension, dimension, complex_z)?;
    let alpha = (0..dimension)
        .map(|index| left_form.as_slice()[index * dimension + index])
        .collect();
    let beta = (0..dimension)
        .map(|index| right_form.as_slice()[index * dimension + index])
        .collect();
    Ok(GeneralizedSchurDecomposition {
        left_form,
        right_form,
        left_vectors,
        right_vectors,
        alpha,
        beta,
    })
}

/// Compute selected generalized eigenvectors from a real generalized Schur
/// form.
pub fn eigenvectors_from_generalized_real_schur(
    left_form: &RealMatrix,
    right_form: &RealMatrix,
    left_vectors: &RealMatrix,
    right_vectors: &RealMatrix,
    selected: &[bool],
    compute_left: bool,
    compute_right: bool,
) -> Result<EigenvectorSet, DecompositionError> {
    let dimension =
        matching_real_square_dimensions(&[left_form, right_form, left_vectors, right_vectors])?;
    validate_selection(dimension, selected)?;
    let complex =
        complexify_real_generalized_schur(left_form, right_form, left_vectors, right_vectors)?;
    eigenvectors_from_generalized_schur(
        complex.left_form(),
        complex.right_form(),
        complex.left_vectors(),
        complex.right_vectors(),
        selected,
        compute_left,
        compute_right,
    )
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

fn real_schur_from_backend(
    dimension: usize,
    result: thouless_lapack::RealSchur,
) -> Result<RealSchurDecomposition, DecompositionError> {
    Ok(RealSchurDecomposition {
        form: RealMatrix::new(dimension, dimension, result.form_row_major().to_vec())?,
        vectors: RealMatrix::new(dimension, dimension, result.vectors_row_major().to_vec())?,
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

fn generalized_real_schur_from_backend(
    dimension: usize,
    result: thouless_lapack::GeneralizedRealSchur,
) -> Result<GeneralizedRealSchurDecomposition, DecompositionError> {
    Ok(GeneralizedRealSchurDecomposition {
        left_form: RealMatrix::new(dimension, dimension, result.left_form_row_major().to_vec())?,
        right_form: RealMatrix::new(dimension, dimension, result.right_form_row_major().to_vec())?,
        left_vectors: RealMatrix::new(
            dimension,
            dimension,
            result.left_vectors_row_major().to_vec(),
        )?,
        right_vectors: RealMatrix::new(
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

fn real_square_dimension(matrix: &RealMatrix) -> Result<usize, DecompositionError> {
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

fn matching_real_square_dimensions(matrices: &[&RealMatrix]) -> Result<usize, DecompositionError> {
    let Some(first) = matrices.first() else {
        return Ok(0);
    };
    let dimension = real_square_dimension(first)?;
    for matrix in &matrices[1..] {
        if real_square_dimension(matrix)? != dimension {
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

fn real_block_positions(form: &RealMatrix) -> Vec<usize> {
    let dimension = form.rows();
    (0..dimension.saturating_sub(1))
        .filter(|index| form.as_slice()[(index + 1) * dimension + index] != 0.0)
        .collect()
}

fn validate_real_block_selection(
    form: &RealMatrix,
    selected: &[bool],
) -> Result<(), DecompositionError> {
    for index in real_block_positions(form) {
        if selected[index] != selected[index + 1] {
            return Err(DecompositionError::SplitConjugatePair { index });
        }
    }
    Ok(())
}

fn local_two_by_two(matrix: &[Complex64], dimension: usize, index: usize) -> [Complex64; 4] {
    [
        matrix[index * dimension + index],
        matrix[index * dimension + index + 1],
        matrix[(index + 1) * dimension + index],
        matrix[(index + 1) * dimension + index + 1],
    ]
}

fn store_two_by_two(matrix: &mut [Complex64], dimension: usize, index: usize, block: &[Complex64]) {
    matrix[index * dimension + index] = block[0];
    matrix[index * dimension + index + 1] = block[1];
    matrix[(index + 1) * dimension + index] = block[2];
    matrix[(index + 1) * dimension + index + 1] = block[3];
}

fn right_multiply_columns(
    matrix: &mut [Complex64],
    dimension: usize,
    rows: std::ops::Range<usize>,
    first_column: usize,
    block: &[Complex64; 4],
) {
    for row in rows {
        let first = matrix[row * dimension + first_column];
        let second = matrix[row * dimension + first_column + 1];
        matrix[row * dimension + first_column] = first * block[0] + second * block[2];
        matrix[row * dimension + first_column + 1] = first * block[1] + second * block[3];
    }
}

fn left_multiply_rows_adjoint(
    matrix: &mut [Complex64],
    dimension: usize,
    first_row: usize,
    columns: std::ops::Range<usize>,
    block: &[Complex64; 4],
) {
    for column in columns {
        let first = matrix[first_row * dimension + column];
        let second = matrix[(first_row + 1) * dimension + column];
        matrix[first_row * dimension + column] = block[0].conj() * first + block[2].conj() * second;
        matrix[(first_row + 1) * dimension + column] =
            block[1].conj() * first + block[3].conj() * second;
    }
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
    fn real_schur_retains_and_complexifies_a_conjugate_pair() {
        let matrix = RealMatrix::new(2, 2, vec![0.0, -2.0, 0.5, 0.0]).unwrap();
        let decomposition = real_schur(&matrix).unwrap();
        assert_ne!(decomposition.form().as_slice()[2], 0.0);
        assert!(decomposition.eigenvalues()[0].im > 0.0);
        assert!(decomposition.eigenvalues()[1].im < 0.0);

        let reconstructed = multiply(
            &multiply(
                &decomposition.vectors().to_complex(),
                &decomposition.form().to_complex(),
            )
            .unwrap(),
            &decomposition.vectors().to_complex().adjoint(),
        )
        .unwrap();
        assert!(residual_norm(&reconstructed, &matrix.to_complex()) < 1.0e-12);

        let complex = complexify_real_schur(decomposition.form(), decomposition.vectors()).unwrap();
        assert!(complex.form().as_slice()[2].norm() < 1.0e-14);
        assert!(complex.eigenvalues()[0].im > 0.0);
        assert!(complex.eigenvalues()[1].im < 0.0);
        let complex_reconstructed = multiply(
            &multiply(complex.vectors(), complex.form()).unwrap(),
            &complex.vectors().adjoint(),
        )
        .unwrap();
        assert!(residual_norm(&complex_reconstructed, &matrix.to_complex()) < 1.0e-12);

        let error = reorder_real_schur(
            decomposition.form(),
            decomposition.vectors(),
            &[true, false],
        )
        .unwrap_err();
        assert_eq!(error, DecompositionError::SplitConjugatePair { index: 0 });
    }

    #[test]
    fn generalized_real_schur_preserves_pair_order_and_eigenvectors() {
        let left =
            RealMatrix::new(3, 3, vec![0.0, -2.0, 0.4, 0.5, 0.0, -0.2, 0.0, 0.0, 3.0]).unwrap();
        let right =
            RealMatrix::new(3, 3, vec![1.0, 0.1, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.0]).unwrap();
        let decomposition = generalized_real_schur(&left, &right).unwrap();
        let left_reconstructed = multiply(
            &multiply(
                &decomposition.left_vectors().to_complex(),
                &decomposition.left_form().to_complex(),
            )
            .unwrap(),
            &decomposition.right_vectors().to_complex().adjoint(),
        )
        .unwrap();
        let right_reconstructed = multiply(
            &multiply(
                &decomposition.left_vectors().to_complex(),
                &decomposition.right_form().to_complex(),
            )
            .unwrap(),
            &decomposition.right_vectors().to_complex().adjoint(),
        )
        .unwrap();
        assert!(residual_norm(&left_reconstructed, &left.to_complex()) < 1.0e-12);
        assert!(residual_norm(&right_reconstructed, &right.to_complex()) < 1.0e-12);

        let complex = complexify_real_generalized_schur(
            decomposition.left_form(),
            decomposition.right_form(),
            decomposition.left_vectors(),
            decomposition.right_vectors(),
        )
        .unwrap();
        for row in 1..3 {
            for column in 0..row {
                assert!(complex.left_form().as_slice()[row * 3 + column].norm() < 1.0e-12);
                assert!(complex.right_form().as_slice()[row * 3 + column].norm() < 1.0e-12);
            }
        }
        let complex_left = multiply(
            &multiply(complex.left_vectors(), complex.left_form()).unwrap(),
            &complex.right_vectors().adjoint(),
        )
        .unwrap();
        let complex_right = multiply(
            &multiply(complex.left_vectors(), complex.right_form()).unwrap(),
            &complex.right_vectors().adjoint(),
        )
        .unwrap();
        assert!(residual_norm(&complex_left, &left.to_complex()) < 1.0e-11);
        assert!(residual_norm(&complex_right, &right.to_complex()) < 1.0e-11);

        let vectors = eigenvectors_from_generalized_real_schur(
            decomposition.left_form(),
            decomposition.right_form(),
            decomposition.left_vectors(),
            decomposition.right_vectors(),
            &[true, true, true],
            false,
            true,
        )
        .unwrap();
        let right_vectors = vectors.right().unwrap();
        for column in 0..3 {
            let alpha = complex.alpha()[column];
            let beta = complex.beta()[column];
            for row in 0..3 {
                let left_applied = (0..3)
                    .map(|inner| {
                        left.as_slice()[row * 3 + inner]
                            * right_vectors.as_slice()[inner * 3 + column]
                    })
                    .sum::<Complex64>()
                    * beta;
                let right_applied = (0..3)
                    .map(|inner| {
                        right.as_slice()[row * 3 + inner]
                            * right_vectors.as_slice()[inner * 3 + column]
                    })
                    .sum::<Complex64>()
                    * alpha;
                assert!((left_applied - right_applied).norm() < 1.0e-10);
            }
        }
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
