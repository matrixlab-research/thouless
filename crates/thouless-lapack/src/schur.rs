//! Safe complex Schur and generalized Schur decomposition boundaries.

use num_complex::Complex64;

use super::{check_info, workspace_length, Error};

/// Real Schur form `A = Q T Qᵀ`.
///
/// `T` is quasi-upper-triangular: its diagonal contains real 1-by-1
/// eigenvalue blocks and real 2-by-2 blocks for complex-conjugate pairs.
#[derive(Clone, Debug, PartialEq)]
pub struct RealSchur {
    form: Vec<f64>,
    vectors: Vec<f64>,
    eigenvalues: Vec<Complex64>,
}

impl RealSchur {
    /// Quasi-upper-triangular form in row-major order.
    #[must_use]
    pub fn form_row_major(&self) -> &[f64] {
        &self.form
    }

    /// Orthogonal Schur vectors in row-major order.
    #[must_use]
    pub fn vectors_row_major(&self) -> &[f64] {
        &self.vectors
    }

    /// Eigenvalues in Schur block order.
    #[must_use]
    pub fn eigenvalues(&self) -> &[Complex64] {
        &self.eigenvalues
    }
}

/// Complex Schur form `A = Q T Qᴴ`.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexSchur {
    form: Vec<Complex64>,
    vectors: Vec<Complex64>,
    eigenvalues: Vec<Complex64>,
}

impl ComplexSchur {
    /// Upper-triangular form in row-major order.
    #[must_use]
    pub fn form_row_major(&self) -> &[Complex64] {
        &self.form
    }

    /// Unitary Schur vectors in row-major order.
    #[must_use]
    pub fn vectors_row_major(&self) -> &[Complex64] {
        &self.vectors
    }

    /// Eigenvalues in Schur diagonal order.
    #[must_use]
    pub fn eigenvalues(&self) -> &[Complex64] {
        &self.eigenvalues
    }
}

/// Complex generalized Schur form
/// `A = Q S Zᴴ`, `B = Q T Zᴴ`.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralizedComplexSchur {
    left_form: Vec<Complex64>,
    right_form: Vec<Complex64>,
    left_vectors: Vec<Complex64>,
    right_vectors: Vec<Complex64>,
    alpha: Vec<Complex64>,
    beta: Vec<Complex64>,
}

/// Real generalized Schur form `A = Q S Zᵀ`, `B = Q T Zᵀ`.
///
/// `S` is quasi-upper-triangular and `T` is upper-triangular.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralizedRealSchur {
    left_form: Vec<f64>,
    right_form: Vec<f64>,
    left_vectors: Vec<f64>,
    right_vectors: Vec<f64>,
    alpha: Vec<Complex64>,
    beta: Vec<f64>,
}

impl GeneralizedRealSchur {
    /// First quasi-upper-triangular form in row-major order.
    #[must_use]
    pub fn left_form_row_major(&self) -> &[f64] {
        &self.left_form
    }

    /// Second upper-triangular form in row-major order.
    #[must_use]
    pub fn right_form_row_major(&self) -> &[f64] {
        &self.right_form
    }

    /// Left orthogonal vectors in row-major order.
    #[must_use]
    pub fn left_vectors_row_major(&self) -> &[f64] {
        &self.left_vectors
    }

    /// Right orthogonal vectors in row-major order.
    #[must_use]
    pub fn right_vectors_row_major(&self) -> &[f64] {
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

impl GeneralizedComplexSchur {
    /// First triangular form `S` in row-major order.
    #[must_use]
    pub fn left_form_row_major(&self) -> &[Complex64] {
        &self.left_form
    }

    /// Second triangular form `T` in row-major order.
    #[must_use]
    pub fn right_form_row_major(&self) -> &[Complex64] {
        &self.right_form
    }

    /// Left unitary vectors `Q` in row-major order.
    #[must_use]
    pub fn left_vectors_row_major(&self) -> &[Complex64] {
        &self.left_vectors
    }

    /// Right unitary vectors `Z` in row-major order.
    #[must_use]
    pub fn right_vectors_row_major(&self) -> &[Complex64] {
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

/// Selected left and right eigenvector columns.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexEigenvectors {
    left: Option<Vec<Complex64>>,
    right: Option<Vec<Complex64>>,
    selected_count: usize,
}

/// Requested sides of an ordinary or generalized eigenproblem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EigenvectorSides {
    /// Do not compute eigenvectors.
    Neither,
    /// Compute left eigenvectors.
    Left,
    /// Compute right eigenvectors.
    Right,
    /// Compute both left and right eigenvectors.
    Both,
}

impl ComplexEigenvectors {
    /// Selected left eigenvectors in row-major `(dimension, selected_count)`
    /// layout.
    #[must_use]
    pub fn left_row_major(&self) -> Option<&[Complex64]> {
        self.left.as_deref()
    }

    /// Selected right eigenvectors in row-major `(dimension, selected_count)`
    /// layout.
    #[must_use]
    pub fn right_row_major(&self) -> Option<&[Complex64]> {
        self.right.as_deref()
    }

    /// Number of selected eigenvector columns.
    #[must_use]
    pub const fn selected_count(&self) -> usize {
        self.selected_count
    }
}

/// Compute a real Schur decomposition without destroying conjugate-pair
/// blocks.
pub fn real_schur(dimension: usize, row_major_entries: &[f64]) -> Result<RealSchur, Error> {
    validate_real_square(dimension, row_major_entries)?;
    if dimension == 0 {
        return Ok(RealSchur {
            form: Vec::new(),
            vectors: Vec::new(),
            eigenvalues: Vec::new(),
        });
    }

    let n = lapack_dimension(dimension)?;
    let mut form = to_real_column_major(dimension, row_major_entries);
    let mut vectors = vec![0.0; dimension * dimension];
    let mut real_parts = vec![0.0; dimension];
    let mut imaginary_parts = vec![0.0; dimension];
    let mut selected_dimension = 0;
    let mut boolean_work = vec![0; dimension.max(1)];
    let mut work_query = [0.0];
    let mut info = 0;

    // SAFETY: all DGEES buffers have their documented dimensions.  SORT='N'
    // guarantees that the null selection callback is not invoked.
    unsafe {
        lapack::dgees(
            b'V',
            b'N',
            None,
            n,
            &mut form,
            n,
            &mut selected_dimension,
            &mut real_parts,
            &mut imaginary_parts,
            &mut vectors,
            n,
            &mut work_query,
            -1,
            &mut boolean_work,
            &mut info,
        );
    }
    check_info(info)?;
    let work_length = workspace_length(work_query[0])?;
    let mut work = vec![0.0; work_length];

    // SAFETY: the successful query supplied the allocated workspace length;
    // all remaining buffers retain their validated sizes.
    unsafe {
        lapack::dgees(
            b'V',
            b'N',
            None,
            n,
            &mut form,
            n,
            &mut selected_dimension,
            &mut real_parts,
            &mut imaginary_parts,
            &mut vectors,
            n,
            &mut work,
            i32::try_from(work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut boolean_work,
            &mut info,
        );
    }
    check_info(info)?;

    Ok(RealSchur {
        form: to_real_row_major(dimension, &form),
        vectors: to_real_row_major(dimension, &vectors),
        eigenvalues: real_parts
            .into_iter()
            .zip(imaginary_parts)
            .map(|(real, imaginary)| Complex64::new(real, imaginary))
            .collect(),
    })
}

/// Reorder a real Schur form while retaining its real block structure.
///
/// Callers must select both entries of every 2-by-2 conjugate-pair block.
pub fn reorder_real_schur(
    dimension: usize,
    form_row_major: &[f64],
    vectors_row_major: &[f64],
    selected: &[bool],
) -> Result<RealSchur, Error> {
    validate_real_square(dimension, form_row_major)?;
    validate_real_square(dimension, vectors_row_major)?;
    validate_selection(dimension, selected)?;
    if dimension == 0 {
        return Ok(RealSchur {
            form: Vec::new(),
            vectors: Vec::new(),
            eigenvalues: Vec::new(),
        });
    }

    let n = lapack_dimension(dimension)?;
    let mut form = to_real_column_major(dimension, form_row_major);
    let mut vectors = to_real_column_major(dimension, vectors_row_major);
    let selection = logical_selection(selected);
    let mut real_parts = vec![0.0; dimension];
    let mut imaginary_parts = vec![0.0; dimension];
    let mut selected_dimension = 0;
    let mut reciprocal_condition = [0.0];
    let mut separation = [0.0];
    let work_length = dimension.max(1);
    let mut work = vec![0.0; work_length];
    let mut integer_work = [0];
    let mut info = 0;

    // SAFETY: DTRSEN receives a validated real Schur form, a length-n
    // selection mask and the documented JOB='N' workspaces.
    unsafe {
        lapack::dtrsen(
            b'N',
            b'V',
            &selection,
            n,
            &mut form,
            n,
            &mut vectors,
            n,
            &mut real_parts,
            &mut imaginary_parts,
            &mut selected_dimension,
            &mut reciprocal_condition,
            &mut separation,
            &mut work,
            i32::try_from(work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut integer_work,
            1,
            &mut info,
        );
    }
    check_info(info)?;

    Ok(RealSchur {
        form: to_real_row_major(dimension, &form),
        vectors: to_real_row_major(dimension, &vectors),
        eigenvalues: real_parts
            .into_iter()
            .zip(imaginary_parts)
            .map(|(real, imaginary)| Complex64::new(real, imaginary))
            .collect(),
    })
}

/// Compute a complex Schur decomposition.
pub fn complex_schur(
    dimension: usize,
    row_major_entries: &[Complex64],
) -> Result<ComplexSchur, Error> {
    validate_square(dimension, row_major_entries)?;
    if dimension == 0 {
        return Ok(ComplexSchur {
            form: Vec::new(),
            vectors: Vec::new(),
            eigenvalues: Vec::new(),
        });
    }

    let n = lapack_dimension(dimension)?;
    let mut form = to_column_major(dimension, row_major_entries);
    let mut vectors = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    let mut eigenvalues = vec![Complex64::new(0.0, 0.0); dimension];
    let mut selected_dimension = 0;
    let mut real_work = vec![0.0; dimension.max(1)];
    let mut boolean_work = vec![0; dimension.max(1)];
    let mut work_query = [Complex64::new(0.0, 0.0)];
    let mut info = 0;

    // SAFETY: all matrix and result buffers have the dimensions required by
    // ZGEES. SORT='N' means the null selection callback is never invoked.
    unsafe {
        lapack::zgees(
            b'V',
            b'N',
            None,
            n,
            &mut form,
            n,
            &mut selected_dimension,
            &mut eigenvalues,
            &mut vectors,
            n,
            &mut work_query,
            -1,
            &mut real_work,
            &mut boolean_work,
            &mut info,
        );
    }
    check_info(info)?;
    let work_length = workspace_length(work_query[0].re)?;
    let mut work = vec![Complex64::new(0.0, 0.0); work_length];

    // SAFETY: the successful query supplied the allocated workspace length;
    // all other buffers are unchanged and remain dimensionally valid.
    unsafe {
        lapack::zgees(
            b'V',
            b'N',
            None,
            n,
            &mut form,
            n,
            &mut selected_dimension,
            &mut eigenvalues,
            &mut vectors,
            n,
            &mut work,
            i32::try_from(work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut real_work,
            &mut boolean_work,
            &mut info,
        );
    }
    check_info(info)?;

    Ok(ComplexSchur {
        form: to_row_major(dimension, &form),
        vectors: to_row_major(dimension, &vectors),
        eigenvalues,
    })
}

/// Reorder a complex Schur form so selected eigenvalues lead the diagonal.
pub fn reorder_complex_schur(
    dimension: usize,
    form_row_major: &[Complex64],
    vectors_row_major: &[Complex64],
    selected: &[bool],
) -> Result<ComplexSchur, Error> {
    validate_square(dimension, form_row_major)?;
    validate_square(dimension, vectors_row_major)?;
    validate_selection(dimension, selected)?;
    if dimension == 0 {
        return Ok(ComplexSchur {
            form: Vec::new(),
            vectors: Vec::new(),
            eigenvalues: Vec::new(),
        });
    }

    let n = lapack_dimension(dimension)?;
    let mut form = to_column_major(dimension, form_row_major);
    let mut vectors = to_column_major(dimension, vectors_row_major);
    let selection = logical_selection(selected);
    let mut eigenvalues = vec![Complex64::new(0.0, 0.0); dimension];
    let mut selected_dimension = 0;
    let mut reciprocal_condition = [0.0];
    let mut separation = [0.0];
    let work_length = dimension.max(1);
    let mut work = vec![Complex64::new(0.0, 0.0); work_length];
    let mut info = 0;

    // SAFETY: ZTRSEN receives square column-major Schur factors, a length-n
    // selection vector, and the documented JOB='N' workspace of max(1, n).
    unsafe {
        lapack::ztrsen(
            b'N',
            b'V',
            &selection,
            n,
            &mut form,
            n,
            &mut vectors,
            n,
            &mut eigenvalues,
            &mut selected_dimension,
            &mut reciprocal_condition,
            &mut separation,
            &mut work,
            i32::try_from(work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut info,
        );
    }
    check_info(info)?;

    Ok(ComplexSchur {
        form: to_row_major(dimension, &form),
        vectors: to_row_major(dimension, &vectors),
        eigenvalues,
    })
}

/// Compute selected eigenvectors of the matrix represented by a complex Schur
/// decomposition.
pub fn complex_schur_eigenvectors(
    dimension: usize,
    form_row_major: &[Complex64],
    vectors_row_major: &[Complex64],
    selected: &[bool],
    compute_left: bool,
    compute_right: bool,
) -> Result<ComplexEigenvectors, Error> {
    validate_square(dimension, form_row_major)?;
    validate_square(dimension, vectors_row_major)?;
    validate_selection(dimension, selected)?;
    let selected_count = selected.iter().filter(|value| **value).count();
    if dimension == 0 || (!compute_left && !compute_right) {
        return Ok(ComplexEigenvectors {
            left: compute_left.then(Vec::new),
            right: compute_right.then(Vec::new),
            selected_count,
        });
    }

    let n = lapack_dimension(dimension)?;
    let mut form = to_column_major(dimension, form_row_major);
    let mut left = to_column_major(dimension, vectors_row_major);
    let mut right = left.clone();
    let ignored_selection = vec![1; dimension];
    let mut returned_count = 0;
    let mut work = vec![Complex64::new(0.0, 0.0); (2 * dimension).max(1)];
    let mut real_work = vec![0.0; dimension.max(1)];
    let side = match (compute_left, compute_right) {
        (true, true) => b'B',
        (true, false) => b'L',
        (false, true) => b'R',
        (false, false) => unreachable!(),
    };
    let mut info = 0;

    // SAFETY: ZTREVC receives a square complex Schur form, both n-by-n
    // back-transformation buffers, and documented 2n/n workspaces.
    unsafe {
        lapack::ztrevc(
            side,
            b'B',
            &ignored_selection,
            n,
            &mut form,
            n,
            &mut left,
            n,
            &mut right,
            n,
            n,
            &mut returned_count,
            &mut work,
            &mut real_work,
            &mut info,
        );
    }
    check_info(info)?;
    if returned_count != n {
        return Err(Error::NoConvergence {
            detail: returned_count,
        });
    }

    Ok(ComplexEigenvectors {
        left: compute_left.then(|| selected_columns(dimension, &left, selected)),
        right: compute_right.then(|| selected_columns(dimension, &right, selected)),
        selected_count,
    })
}

/// Compute a complex generalized Schur decomposition.
pub fn generalized_complex_schur(
    dimension: usize,
    left_row_major: &[Complex64],
    right_row_major: &[Complex64],
) -> Result<GeneralizedComplexSchur, Error> {
    validate_square(dimension, left_row_major)?;
    validate_square(dimension, right_row_major)?;
    if dimension == 0 {
        return Ok(GeneralizedComplexSchur {
            left_form: Vec::new(),
            right_form: Vec::new(),
            left_vectors: Vec::new(),
            right_vectors: Vec::new(),
            alpha: Vec::new(),
            beta: Vec::new(),
        });
    }

    let n = lapack_dimension(dimension)?;
    let mut left_form = to_column_major(dimension, left_row_major);
    let mut right_form = to_column_major(dimension, right_row_major);
    let mut left_vectors = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    let mut right_vectors = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    let mut alpha = vec![Complex64::new(0.0, 0.0); dimension];
    let mut beta = vec![Complex64::new(0.0, 0.0); dimension];
    let mut selected_dimension = 0;
    let mut real_work = vec![0.0; (8 * dimension).max(1)];
    let mut boolean_work = vec![0; dimension.max(1)];
    let mut work_query = [Complex64::new(0.0, 0.0)];
    let mut info = 0;

    // SAFETY: all ZGGES buffers are sized for n-by-n matrices. SORT='N'
    // guarantees the null selection callback is not invoked.
    unsafe {
        lapack::zgges(
            b'V',
            b'V',
            b'N',
            None,
            n,
            &mut left_form,
            n,
            &mut right_form,
            n,
            &mut selected_dimension,
            &mut alpha,
            &mut beta,
            &mut left_vectors,
            n,
            &mut right_vectors,
            n,
            &mut work_query,
            -1,
            &mut real_work,
            &mut boolean_work,
            &mut info,
        );
    }
    check_info(info)?;
    let work_length = workspace_length(work_query[0].re)?;
    let mut work = vec![Complex64::new(0.0, 0.0); work_length];

    // SAFETY: the successful ZGGES query supplied the workspace allocation;
    // all matrix and output buffers retain their validated sizes.
    unsafe {
        lapack::zgges(
            b'V',
            b'V',
            b'N',
            None,
            n,
            &mut left_form,
            n,
            &mut right_form,
            n,
            &mut selected_dimension,
            &mut alpha,
            &mut beta,
            &mut left_vectors,
            n,
            &mut right_vectors,
            n,
            &mut work,
            i32::try_from(work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut real_work,
            &mut boolean_work,
            &mut info,
        );
    }
    check_info(info)?;

    Ok(GeneralizedComplexSchur {
        left_form: to_row_major(dimension, &left_form),
        right_form: to_row_major(dimension, &right_form),
        left_vectors: to_row_major(dimension, &left_vectors),
        right_vectors: to_row_major(dimension, &right_vectors),
        alpha,
        beta,
    })
}

/// Compute a real generalized Schur decomposition.
pub fn generalized_real_schur(
    dimension: usize,
    left_row_major: &[f64],
    right_row_major: &[f64],
) -> Result<GeneralizedRealSchur, Error> {
    validate_real_square(dimension, left_row_major)?;
    validate_real_square(dimension, right_row_major)?;
    if dimension == 0 {
        return Ok(GeneralizedRealSchur {
            left_form: Vec::new(),
            right_form: Vec::new(),
            left_vectors: Vec::new(),
            right_vectors: Vec::new(),
            alpha: Vec::new(),
            beta: Vec::new(),
        });
    }

    let n = lapack_dimension(dimension)?;
    let mut left_form = to_real_column_major(dimension, left_row_major);
    let mut right_form = to_real_column_major(dimension, right_row_major);
    let mut left_vectors = vec![0.0; dimension * dimension];
    let mut right_vectors = vec![0.0; dimension * dimension];
    let mut alpha_real = vec![0.0; dimension];
    let mut alpha_imaginary = vec![0.0; dimension];
    let mut beta = vec![0.0; dimension];
    let mut selected_dimension = 0;
    let mut boolean_work = vec![0; dimension.max(1)];
    let mut work_query = [0.0];
    let mut info = 0;

    // SAFETY: all DGGES buffers are sized for n-by-n matrices.  SORT='N'
    // guarantees that the null selection callback is not invoked.
    unsafe {
        lapack::dgges(
            b'V',
            b'V',
            b'N',
            None,
            n,
            &mut left_form,
            n,
            &mut right_form,
            n,
            &mut selected_dimension,
            &mut alpha_real,
            &mut alpha_imaginary,
            &mut beta,
            &mut left_vectors,
            n,
            &mut right_vectors,
            n,
            &mut work_query,
            -1,
            &mut boolean_work,
            &mut info,
        );
    }
    check_info(info)?;
    let work_length = workspace_length(work_query[0])?;
    let mut work = vec![0.0; work_length];

    // SAFETY: the successful workspace query supplied the allocation used
    // here; all matrix and output buffers remain valid.
    unsafe {
        lapack::dgges(
            b'V',
            b'V',
            b'N',
            None,
            n,
            &mut left_form,
            n,
            &mut right_form,
            n,
            &mut selected_dimension,
            &mut alpha_real,
            &mut alpha_imaginary,
            &mut beta,
            &mut left_vectors,
            n,
            &mut right_vectors,
            n,
            &mut work,
            i32::try_from(work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut boolean_work,
            &mut info,
        );
    }
    check_info(info)?;

    Ok(GeneralizedRealSchur {
        left_form: to_real_row_major(dimension, &left_form),
        right_form: to_real_row_major(dimension, &right_form),
        left_vectors: to_real_row_major(dimension, &left_vectors),
        right_vectors: to_real_row_major(dimension, &right_vectors),
        alpha: alpha_real
            .into_iter()
            .zip(alpha_imaginary)
            .map(|(real, imaginary)| Complex64::new(real, imaginary))
            .collect(),
        beta,
    })
}

/// Reorder a real generalized Schur form while retaining real block
/// structure.  Both members of every conjugate-pair block must be selected
/// together.
pub fn reorder_generalized_real_schur(
    dimension: usize,
    left_form_row_major: &[f64],
    right_form_row_major: &[f64],
    left_vectors_row_major: &[f64],
    right_vectors_row_major: &[f64],
    selected: &[bool],
) -> Result<GeneralizedRealSchur, Error> {
    validate_real_square(dimension, left_form_row_major)?;
    validate_real_square(dimension, right_form_row_major)?;
    validate_real_square(dimension, left_vectors_row_major)?;
    validate_real_square(dimension, right_vectors_row_major)?;
    validate_selection(dimension, selected)?;
    if dimension == 0 {
        return Ok(GeneralizedRealSchur {
            left_form: Vec::new(),
            right_form: Vec::new(),
            left_vectors: Vec::new(),
            right_vectors: Vec::new(),
            alpha: Vec::new(),
            beta: Vec::new(),
        });
    }

    let n = lapack_dimension(dimension)?;
    let mut left_form = to_real_column_major(dimension, left_form_row_major);
    let mut right_form = to_real_column_major(dimension, right_form_row_major);
    let mut left_vectors = to_real_column_major(dimension, left_vectors_row_major);
    let mut right_vectors = to_real_column_major(dimension, right_vectors_row_major);
    let selection = logical_selection(selected);
    let mut alpha_real = vec![0.0; dimension];
    let mut alpha_imaginary = vec![0.0; dimension];
    let mut beta = vec![0.0; dimension];
    let mut selected_dimension = 0;
    let mut reciprocal_left = [0.0];
    let mut reciprocal_right = [0.0];
    let mut separation = [0.0, 0.0];
    let mut work_query = [0.0];
    let mut integer_work_query = [0];
    let mut info = 0;
    let job = [0];
    let update_left = [1];
    let update_right = [1];

    // SAFETY: DTGSEN query mode receives dimensionally valid generalized
    // Schur buffers and writes only workspace recommendations and scalars.
    unsafe {
        lapack::dtgsen(
            &job,
            &update_left,
            &update_right,
            &selection,
            n,
            &mut left_form,
            n,
            &mut right_form,
            n,
            &mut alpha_real,
            &mut alpha_imaginary,
            &mut beta,
            &mut left_vectors,
            n,
            &mut right_vectors,
            n,
            &mut selected_dimension,
            &mut reciprocal_left,
            &mut reciprocal_right,
            &mut separation,
            &mut work_query,
            -1,
            &mut integer_work_query,
            -1,
            &mut info,
        );
    }
    check_info(info)?;
    let work_length = workspace_length(work_query[0])?;
    let integer_work_length =
        usize::try_from(integer_work_query[0]).map_err(|_| Error::InvalidWorkspace)?;
    if integer_work_length == 0 {
        return Err(Error::InvalidWorkspace);
    }
    let mut work = vec![0.0; work_length];
    let mut integer_work = vec![0; integer_work_length];

    // SAFETY: DTGSEN receives the workspaces returned by its successful query
    // and the same validated generalized Schur buffers.
    unsafe {
        lapack::dtgsen(
            &job,
            &update_left,
            &update_right,
            &selection,
            n,
            &mut left_form,
            n,
            &mut right_form,
            n,
            &mut alpha_real,
            &mut alpha_imaginary,
            &mut beta,
            &mut left_vectors,
            n,
            &mut right_vectors,
            n,
            &mut selected_dimension,
            &mut reciprocal_left,
            &mut reciprocal_right,
            &mut separation,
            &mut work,
            i32::try_from(work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut integer_work,
            i32::try_from(integer_work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut info,
        );
    }
    check_info(info)?;

    Ok(GeneralizedRealSchur {
        left_form: to_real_row_major(dimension, &left_form),
        right_form: to_real_row_major(dimension, &right_form),
        left_vectors: to_real_row_major(dimension, &left_vectors),
        right_vectors: to_real_row_major(dimension, &right_vectors),
        alpha: alpha_real
            .into_iter()
            .zip(alpha_imaginary)
            .map(|(real, imaginary)| Complex64::new(real, imaginary))
            .collect(),
        beta,
    })
}

/// Reorder a generalized complex Schur form.
pub fn reorder_generalized_complex_schur(
    dimension: usize,
    left_form_row_major: &[Complex64],
    right_form_row_major: &[Complex64],
    left_vectors_row_major: &[Complex64],
    right_vectors_row_major: &[Complex64],
    selected: &[bool],
) -> Result<GeneralizedComplexSchur, Error> {
    validate_square(dimension, left_form_row_major)?;
    validate_square(dimension, right_form_row_major)?;
    validate_square(dimension, left_vectors_row_major)?;
    validate_square(dimension, right_vectors_row_major)?;
    validate_selection(dimension, selected)?;
    if dimension == 0 {
        return Ok(GeneralizedComplexSchur {
            left_form: Vec::new(),
            right_form: Vec::new(),
            left_vectors: Vec::new(),
            right_vectors: Vec::new(),
            alpha: Vec::new(),
            beta: Vec::new(),
        });
    }

    let n = lapack_dimension(dimension)?;
    let mut left_form = to_column_major(dimension, left_form_row_major);
    let mut right_form = to_column_major(dimension, right_form_row_major);
    let mut left_vectors = to_column_major(dimension, left_vectors_row_major);
    let mut right_vectors = to_column_major(dimension, right_vectors_row_major);
    let selection = logical_selection(selected);
    let mut alpha = (0..dimension)
        .map(|index| left_form[index + index * dimension])
        .collect::<Vec<_>>();
    let mut beta = (0..dimension)
        .map(|index| right_form[index + index * dimension])
        .collect::<Vec<_>>();
    let mut selected_dimension = 0;
    let mut reciprocal_left = [0.0];
    let mut reciprocal_right = [0.0];
    let mut separation = [0.0, 0.0];
    let mut work_query = [Complex64::new(0.0, 0.0)];
    let mut integer_work_query = [0];
    let mut info = 0;
    let job = [0];
    let update_left = [1];
    let update_right = [1];

    // SAFETY: buffers contain a dimensionally valid generalized Schur form;
    // ZTGSEN query mode writes only workspace recommendations and scalars.
    unsafe {
        lapack::ztgsen(
            &job,
            &update_left,
            &update_right,
            &selection,
            n,
            &mut left_form,
            n,
            &mut right_form,
            n,
            &mut alpha,
            &mut beta,
            &mut left_vectors,
            n,
            &mut right_vectors,
            n,
            &mut selected_dimension,
            &mut reciprocal_left,
            &mut reciprocal_right,
            &mut separation,
            &mut work_query,
            -1,
            &mut integer_work_query,
            -1,
            &mut info,
        );
    }
    check_info(info)?;
    let work_length = workspace_length(work_query[0].re)?;
    let integer_work_length =
        usize::try_from(integer_work_query[0]).map_err(|_| Error::InvalidWorkspace)?;
    if integer_work_length == 0 {
        return Err(Error::InvalidWorkspace);
    }
    let mut work = vec![Complex64::new(0.0, 0.0); work_length];
    let mut integer_work = vec![0; integer_work_length];

    // SAFETY: ZTGSEN receives its queried complex and integer workspaces and
    // the same validated generalized Schur buffers.
    unsafe {
        lapack::ztgsen(
            &job,
            &update_left,
            &update_right,
            &selection,
            n,
            &mut left_form,
            n,
            &mut right_form,
            n,
            &mut alpha,
            &mut beta,
            &mut left_vectors,
            n,
            &mut right_vectors,
            n,
            &mut selected_dimension,
            &mut reciprocal_left,
            &mut reciprocal_right,
            &mut separation,
            &mut work,
            i32::try_from(work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut integer_work,
            i32::try_from(integer_work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut info,
        );
    }
    check_info(info)?;

    Ok(GeneralizedComplexSchur {
        left_form: to_row_major(dimension, &left_form),
        right_form: to_row_major(dimension, &right_form),
        left_vectors: to_row_major(dimension, &left_vectors),
        right_vectors: to_row_major(dimension, &right_vectors),
        alpha,
        beta,
    })
}

/// Compute selected generalized eigenvectors from a complex generalized Schur
/// decomposition.
pub fn generalized_complex_schur_eigenvectors(
    dimension: usize,
    left_form_row_major: &[Complex64],
    right_form_row_major: &[Complex64],
    left_vectors_row_major: &[Complex64],
    right_vectors_row_major: &[Complex64],
    selected: &[bool],
    sides: EigenvectorSides,
) -> Result<ComplexEigenvectors, Error> {
    validate_square(dimension, left_form_row_major)?;
    validate_square(dimension, right_form_row_major)?;
    validate_square(dimension, left_vectors_row_major)?;
    validate_square(dimension, right_vectors_row_major)?;
    validate_selection(dimension, selected)?;
    let (compute_left, compute_right) = match sides {
        EigenvectorSides::Neither => (false, false),
        EigenvectorSides::Left => (true, false),
        EigenvectorSides::Right => (false, true),
        EigenvectorSides::Both => (true, true),
    };
    let selected_count = selected.iter().filter(|value| **value).count();
    if dimension == 0 || (!compute_left && !compute_right) {
        return Ok(ComplexEigenvectors {
            left: compute_left.then(Vec::new),
            right: compute_right.then(Vec::new),
            selected_count,
        });
    }

    let n = lapack_dimension(dimension)?;
    let left_form = to_column_major(dimension, left_form_row_major);
    let right_form = to_column_major(dimension, right_form_row_major);
    let mut left = to_column_major(dimension, left_vectors_row_major);
    let mut right = to_column_major(dimension, right_vectors_row_major);
    let ignored_selection = vec![1; dimension];
    let mut returned_count = 0;
    let mut work = vec![Complex64::new(0.0, 0.0); (2 * dimension).max(1)];
    let mut real_work = vec![0.0; (2 * dimension).max(1)];
    let side = match (compute_left, compute_right) {
        (true, true) => b'B',
        (true, false) => b'L',
        (false, true) => b'R',
        (false, false) => unreachable!(),
    };
    let mut info = 0;

    // SAFETY: ZTGEVC receives two square triangular forms, n-by-n
    // back-transformation buffers, and documented 2n workspaces.
    unsafe {
        lapack::ztgevc(
            side,
            b'B',
            &ignored_selection,
            n,
            &left_form,
            n,
            &right_form,
            n,
            &mut left,
            n,
            &mut right,
            n,
            n,
            &mut returned_count,
            &mut work,
            &mut real_work,
            &mut info,
        );
    }
    check_info(info)?;
    if returned_count != n {
        return Err(Error::NoConvergence {
            detail: returned_count,
        });
    }

    Ok(ComplexEigenvectors {
        left: compute_left.then(|| selected_columns(dimension, &left, selected)),
        right: compute_right.then(|| selected_columns(dimension, &right, selected)),
        selected_count,
    })
}

fn validate_square(dimension: usize, entries: &[Complex64]) -> Result<(), Error> {
    let expected = dimension
        .checked_mul(dimension)
        .ok_or(Error::DimensionTooLarge)?;
    if entries.len() != expected {
        return Err(Error::InvalidInputLength {
            expected,
            actual: entries.len(),
        });
    }
    Ok(())
}

fn validate_real_square(dimension: usize, entries: &[f64]) -> Result<(), Error> {
    let expected = dimension
        .checked_mul(dimension)
        .ok_or(Error::DimensionTooLarge)?;
    if entries.len() != expected {
        return Err(Error::InvalidInputLength {
            expected,
            actual: entries.len(),
        });
    }
    Ok(())
}

fn validate_selection(dimension: usize, selected: &[bool]) -> Result<(), Error> {
    if selected.len() != dimension {
        return Err(Error::InvalidSelectionLength {
            expected: dimension,
            actual: selected.len(),
        });
    }
    Ok(())
}

fn lapack_dimension(dimension: usize) -> Result<i32, Error> {
    i32::try_from(dimension).map_err(|_| Error::DimensionTooLarge)
}

fn logical_selection(selected: &[bool]) -> Vec<i32> {
    selected.iter().map(|value| i32::from(*value)).collect()
}

fn to_column_major(dimension: usize, row_major: &[Complex64]) -> Vec<Complex64> {
    (0..dimension)
        .flat_map(|column| (0..dimension).map(move |row| row_major[row * dimension + column]))
        .collect()
}

fn to_row_major(dimension: usize, column_major: &[Complex64]) -> Vec<Complex64> {
    (0..dimension)
        .flat_map(|row| (0..dimension).map(move |column| column_major[row + column * dimension]))
        .collect()
}

fn to_real_column_major(dimension: usize, row_major: &[f64]) -> Vec<f64> {
    (0..dimension)
        .flat_map(|column| (0..dimension).map(move |row| row_major[row * dimension + column]))
        .collect()
}

fn to_real_row_major(dimension: usize, column_major: &[f64]) -> Vec<f64> {
    (0..dimension)
        .flat_map(|row| (0..dimension).map(move |column| column_major[row + column * dimension]))
        .collect()
}

fn selected_columns(
    dimension: usize,
    column_major: &[Complex64],
    selected: &[bool],
) -> Vec<Complex64> {
    let indices = selected
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.then_some(index))
        .collect::<Vec<_>>();
    (0..dimension)
        .flat_map(|row| {
            indices
                .iter()
                .map(move |column| column_major[row + column * dimension])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn multiply(dimension: usize, left: &[Complex64], right: &[Complex64]) -> Vec<Complex64> {
        (0..dimension)
            .flat_map(|row| {
                (0..dimension).map(move |column| {
                    (0..dimension)
                        .map(|inner| {
                            left[row * dimension + inner] * right[inner * dimension + column]
                        })
                        .sum()
                })
            })
            .collect()
    }

    fn adjoint(dimension: usize, matrix: &[Complex64]) -> Vec<Complex64> {
        (0..dimension)
            .flat_map(|row| {
                (0..dimension).map(move |column| matrix[column * dimension + row].conj())
            })
            .collect()
    }

    fn assert_close(left: &[Complex64], right: &[Complex64]) {
        assert_eq!(left.len(), right.len());
        assert!(left
            .iter()
            .zip(right)
            .all(|(left, right)| (*left - *right).norm() < 1.0e-10));
    }

    fn multiply_real(dimension: usize, left: &[f64], right: &[f64]) -> Vec<f64> {
        (0..dimension)
            .flat_map(|row| {
                (0..dimension).map(move |column| {
                    (0..dimension)
                        .map(|inner| {
                            left[row * dimension + inner] * right[inner * dimension + column]
                        })
                        .sum()
                })
            })
            .collect()
    }

    fn transpose_real(dimension: usize, matrix: &[f64]) -> Vec<f64> {
        (0..dimension)
            .flat_map(|row| (0..dimension).map(move |column| matrix[column * dimension + row]))
            .collect()
    }

    fn assert_real_close(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        assert!(left
            .iter()
            .zip(right)
            .all(|(left, right)| (*left - *right).abs() < 1.0e-10));
    }

    #[test]
    fn real_schur_reconstructs_and_reorders_whole_blocks() {
        let matrix = vec![0.0, -2.0, 0.3, 0.5, 0.0, -0.1, 0.0, 0.0, 3.0];
        let decomposition = real_schur(3, &matrix).unwrap();
        assert_ne!(decomposition.form_row_major()[3], 0.0);
        assert!(decomposition.eigenvalues()[0].im > 0.0);
        let reconstructed = multiply_real(
            3,
            &multiply_real(
                3,
                decomposition.vectors_row_major(),
                decomposition.form_row_major(),
            ),
            &transpose_real(3, decomposition.vectors_row_major()),
        );
        assert_real_close(&reconstructed, &matrix);

        let reordered = reorder_real_schur(
            3,
            decomposition.form_row_major(),
            decomposition.vectors_row_major(),
            &[false, false, true],
        )
        .unwrap();
        assert!((reordered.eigenvalues()[0].re - 3.0).abs() < 1.0e-12);
        let reconstructed = multiply_real(
            3,
            &multiply_real(3, reordered.vectors_row_major(), reordered.form_row_major()),
            &transpose_real(3, reordered.vectors_row_major()),
        );
        assert_real_close(&reconstructed, &matrix);
    }

    #[test]
    fn generalized_real_schur_reconstructs_and_reorders_whole_blocks() {
        let left = vec![0.0, -2.0, 0.3, 0.5, 0.0, -0.1, 0.0, 0.0, 3.0];
        let right = vec![1.0, 0.15, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.0];
        let decomposition = generalized_real_schur(3, &left, &right).unwrap();
        assert_ne!(decomposition.left_form_row_major()[3], 0.0);
        let reconstructed_left = multiply_real(
            3,
            &multiply_real(
                3,
                decomposition.left_vectors_row_major(),
                decomposition.left_form_row_major(),
            ),
            &transpose_real(3, decomposition.right_vectors_row_major()),
        );
        let reconstructed_right = multiply_real(
            3,
            &multiply_real(
                3,
                decomposition.left_vectors_row_major(),
                decomposition.right_form_row_major(),
            ),
            &transpose_real(3, decomposition.right_vectors_row_major()),
        );
        assert_real_close(&reconstructed_left, &left);
        assert_real_close(&reconstructed_right, &right);

        let reordered = reorder_generalized_real_schur(
            3,
            decomposition.left_form_row_major(),
            decomposition.right_form_row_major(),
            decomposition.left_vectors_row_major(),
            decomposition.right_vectors_row_major(),
            &[false, false, true],
        )
        .unwrap();
        let reconstructed_left = multiply_real(
            3,
            &multiply_real(
                3,
                reordered.left_vectors_row_major(),
                reordered.left_form_row_major(),
            ),
            &transpose_real(3, reordered.right_vectors_row_major()),
        );
        let reconstructed_right = multiply_real(
            3,
            &multiply_real(
                3,
                reordered.left_vectors_row_major(),
                reordered.right_form_row_major(),
            ),
            &transpose_real(3, reordered.right_vectors_row_major()),
        );
        assert_real_close(&reconstructed_left, &left);
        assert_real_close(&reconstructed_right, &right);
    }

    #[test]
    fn complex_schur_reconstructs_a_nonnormal_matrix() {
        let matrix = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-0.5, 0.2),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(0.3, 0.0),
            Complex64::new(0.0, -0.7),
            Complex64::new(-2.0, 0.0),
        ];
        let decomposition = complex_schur(3, &matrix).unwrap();
        let reconstructed = multiply(
            3,
            &multiply(
                3,
                decomposition.vectors_row_major(),
                decomposition.form_row_major(),
            ),
            &adjoint(3, decomposition.vectors_row_major()),
        );
        assert_close(&reconstructed, &matrix);
    }

    #[test]
    fn generalized_schur_reconstructs_both_matrices() {
        let left = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.5),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.2, -0.3),
        ];
        let right = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.2),
            Complex64::new(0.1, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let decomposition = generalized_complex_schur(2, &left, &right).unwrap();
        let left_reconstructed = multiply(
            2,
            &multiply(
                2,
                decomposition.left_vectors_row_major(),
                decomposition.left_form_row_major(),
            ),
            &adjoint(2, decomposition.right_vectors_row_major()),
        );
        let right_reconstructed = multiply(
            2,
            &multiply(
                2,
                decomposition.left_vectors_row_major(),
                decomposition.right_form_row_major(),
            ),
            &adjoint(2, decomposition.right_vectors_row_major()),
        );
        assert_close(&left_reconstructed, &left);
        assert_close(&right_reconstructed, &right);
    }
}
