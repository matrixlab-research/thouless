//! Audited safe boundary around the LAPACK routines used by Thouless.

#![deny(unsafe_op_in_unsafe_fn)]

extern crate lapack_src;

use num_complex::Complex64;

mod schur;

pub use schur::{
    complex_schur, complex_schur_eigenvectors, generalized_complex_schur,
    generalized_complex_schur_eigenvectors, generalized_real_schur, real_schur,
    reorder_complex_schur, reorder_generalized_complex_schur, reorder_generalized_real_schur,
    reorder_real_schur, ComplexEigenvectors, ComplexSchur, EigenvectorSides,
    GeneralizedComplexSchur, GeneralizedRealSchur, RealSchur,
};

#[cfg(target_os = "macos")]
unsafe extern "C" {
    #[link_name = "zgesdd$NEWLAPACK"]
    fn accelerate_zgesdd(
        jobz: *const std::ffi::c_char,
        rows: *const i32,
        columns: *const i32,
        matrix: *mut Complex64,
        leading_dimension: *const i32,
        singular_values: *mut f64,
        left_vectors: *mut Complex64,
        left_leading_dimension: *const i32,
        right_vectors_adjoint: *mut Complex64,
        right_leading_dimension: *const i32,
        work: *mut Complex64,
        work_length: *const i32,
        real_work: *mut f64,
        integer_work: *mut i32,
        info: *mut i32,
    );
}

#[allow(clippy::too_many_arguments)]
unsafe fn complex_gesdd(
    jobz: u8,
    rows: i32,
    columns: i32,
    matrix: &mut [Complex64],
    leading_dimension: i32,
    singular_values: &mut [f64],
    left_vectors: &mut [Complex64],
    left_leading_dimension: i32,
    right_vectors_adjoint: &mut [Complex64],
    right_leading_dimension: i32,
    work: &mut [Complex64],
    work_length: i32,
    real_work: &mut [f64],
    integer_work: &mut [i32],
    info: &mut i32,
) {
    #[cfg(target_os = "macos")]
    unsafe {
        accelerate_zgesdd(
            &(jobz as std::ffi::c_char),
            &rows,
            &columns,
            matrix.as_mut_ptr(),
            &leading_dimension,
            singular_values.as_mut_ptr(),
            left_vectors.as_mut_ptr(),
            &left_leading_dimension,
            right_vectors_adjoint.as_mut_ptr(),
            &right_leading_dimension,
            work.as_mut_ptr(),
            &work_length,
            real_work.as_mut_ptr(),
            integer_work.as_mut_ptr(),
            info,
        );
    }
    #[cfg(not(target_os = "macos"))]
    unsafe {
        lapack::zgesdd(
            jobz,
            rows,
            columns,
            matrix,
            leading_dimension,
            singular_values,
            left_vectors,
            left_leading_dimension,
            right_vectors_adjoint,
            right_leading_dimension,
            work,
            work_length,
            real_work,
            integer_work,
            info,
        );
    }
}

/// A column-major Hermitian eigensystem.
#[derive(Clone, Debug, PartialEq)]
pub struct HermitianEigensystem {
    eigenvalues: Vec<f64>,
    eigenvectors: Vec<Complex64>,
}

/// A square complex QR decomposition.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexQr {
    unitary: Vec<Complex64>,
    diagonal: Vec<Complex64>,
    first_superdiagonal: Vec<Complex64>,
}

/// A square complex singular-value decomposition.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexSvd {
    left_vectors: Vec<Complex64>,
    singular_values: Vec<f64>,
    right_vectors_adjoint: Vec<Complex64>,
}

impl ComplexSvd {
    /// Returns the left singular vectors in row-major order.
    #[must_use]
    pub fn left_vectors_row_major(&self) -> &[Complex64] {
        &self.left_vectors
    }

    /// Returns singular values in descending order.
    #[must_use]
    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    /// Returns the adjoint of the right singular-vector matrix in row-major order.
    #[must_use]
    pub fn right_vectors_adjoint_row_major(&self) -> &[Complex64] {
        &self.right_vectors_adjoint
    }
}

impl ComplexQr {
    /// Returns the unitary factor in row-major order.
    #[must_use]
    pub fn unitary_row_major(&self) -> &[Complex64] {
        &self.unitary
    }

    /// Returns the diagonal of the upper-triangular factor.
    #[must_use]
    pub fn diagonal(&self) -> &[Complex64] {
        &self.diagonal
    }

    /// Returns the first superdiagonal of the upper-triangular factor.
    #[must_use]
    pub fn first_superdiagonal(&self) -> &[Complex64] {
        &self.first_superdiagonal
    }
}

impl HermitianEigensystem {
    /// Returns eigenvalues in ascending order.
    #[must_use]
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Returns column eigenvectors in column-major order.
    #[must_use]
    pub fn eigenvectors_column_major(&self) -> &[Complex64] {
        &self.eigenvectors
    }
}

/// Failures at the validated LAPACK boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The row-major input does not contain `dimension²` values.
    InvalidInputLength {
        /// Required number of values.
        expected: usize,
        /// Supplied number of values.
        actual: usize,
    },
    /// The selection vector does not contain one entry per matrix row.
    InvalidSelectionLength {
        /// Required number of entries.
        expected: usize,
        /// Supplied number of entries.
        actual: usize,
    },
    /// The matrix dimension does not fit LAPACK's integer ABI.
    DimensionTooLarge,
    /// LAPACK rejected an argument despite boundary validation.
    InvalidLapackArgument {
        /// One-based argument position reported by LAPACK.
        argument: i32,
    },
    /// LAPACK did not converge.
    NoConvergence {
        /// Backend-specific convergence detail.
        detail: i32,
    },
    /// LAPACK returned an invalid workspace query.
    InvalidWorkspace,
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInputLength { expected, actual } => write!(
                formatter,
                "square matrix requires {expected} values; received {actual}"
            ),
            Self::InvalidSelectionLength { expected, actual } => write!(
                formatter,
                "selection requires {expected} values; received {actual}"
            ),
            Self::DimensionTooLarge => {
                write!(formatter, "matrix dimension exceeds the LAPACK integer ABI")
            }
            Self::InvalidLapackArgument { argument } => {
                write!(formatter, "LAPACK rejected argument {argument}")
            }
            Self::NoConvergence { detail } => {
                write!(formatter, "LAPACK eigensolver did not converge ({detail})")
            }
            Self::InvalidWorkspace => {
                write!(formatter, "LAPACK returned an invalid workspace size")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Diagonalizes a Hermitian matrix with LAPACK's divide-and-conquer driver.
///
/// `row_major_entries` contains the full matrix. LAPACK reads its lower
/// triangle and returns ascending eigenvalues and column eigenvectors.
pub fn hermitian_eigensystem(
    dimension: usize,
    row_major_entries: &[Complex64],
) -> Result<HermitianEigensystem, Error> {
    let entry_count = dimension
        .checked_mul(dimension)
        .ok_or(Error::DimensionTooLarge)?;
    if row_major_entries.len() != entry_count {
        return Err(Error::InvalidInputLength {
            expected: entry_count,
            actual: row_major_entries.len(),
        });
    }
    if dimension == 0 {
        return Ok(HermitianEigensystem {
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
        });
    }
    let lapack_dimension = i32::try_from(dimension).map_err(|_| Error::DimensionTooLarge)?;
    let mut eigenvectors = (0..dimension)
        .flat_map(|column| {
            (0..dimension).map(move |row| row_major_entries[row * dimension + column])
        })
        .collect::<Vec<_>>();
    let mut eigenvalues = vec![0.0; dimension];
    let mut work_query = [Complex64::new(0.0, 0.0)];
    let mut real_work_query = [0.0];
    let mut integer_work_query = [0];
    let mut info = 0;

    // SAFETY: all dimensions and buffers have been validated above. In
    // workspace-query mode LAPACK writes only the first element of each
    // workspace and the `info` scalar.
    unsafe {
        lapack::zheevd(
            b'V',
            b'L',
            lapack_dimension,
            &mut eigenvectors,
            lapack_dimension,
            &mut eigenvalues,
            &mut work_query,
            -1,
            &mut real_work_query,
            -1,
            &mut integer_work_query,
            -1,
            &mut info,
        );
    }
    check_info(info)?;
    let work_length = workspace_length(work_query[0].re)?;
    let real_work_length = workspace_length(real_work_query[0])?;
    let integer_work_length =
        usize::try_from(integer_work_query[0]).map_err(|_| Error::InvalidWorkspace)?;
    if integer_work_length == 0 {
        return Err(Error::InvalidWorkspace);
    }
    let mut work = vec![Complex64::new(0.0, 0.0); work_length];
    let mut real_work = vec![0.0; real_work_length];
    let mut integer_work = vec![0; integer_work_length];

    // SAFETY: LAPACK's queried workspace sizes are allocated exactly, and
    // matrix/eigenvalue buffers satisfy the zheevd ABI for `n` and `lda`.
    unsafe {
        lapack::zheevd(
            b'V',
            b'L',
            lapack_dimension,
            &mut eigenvectors,
            lapack_dimension,
            &mut eigenvalues,
            &mut work,
            i32::try_from(work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut real_work,
            i32::try_from(real_work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut integer_work,
            i32::try_from(integer_work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut info,
        );
    }
    check_info(info)?;
    Ok(HermitianEigensystem {
        eigenvalues,
        eigenvectors,
    })
}

/// Computes a square complex QR decomposition with the configured LAPACK backend.
pub fn complex_qr(dimension: usize, row_major_entries: &[Complex64]) -> Result<ComplexQr, Error> {
    let entry_count = dimension
        .checked_mul(dimension)
        .ok_or(Error::DimensionTooLarge)?;
    if row_major_entries.len() != entry_count {
        return Err(Error::InvalidInputLength {
            expected: entry_count,
            actual: row_major_entries.len(),
        });
    }
    if dimension == 0 {
        return Ok(ComplexQr {
            unitary: Vec::new(),
            diagonal: Vec::new(),
            first_superdiagonal: Vec::new(),
        });
    }
    let lapack_dimension = i32::try_from(dimension).map_err(|_| Error::DimensionTooLarge)?;
    let mut factors = (0..dimension)
        .flat_map(|column| {
            (0..dimension).map(move |row| row_major_entries[row * dimension + column])
        })
        .collect::<Vec<_>>();
    let mut tau = vec![Complex64::new(0.0, 0.0); dimension];
    let mut work_query = [Complex64::new(0.0, 0.0)];
    let mut info = 0;

    // SAFETY: dimensions and buffers satisfy the zgeqrf workspace-query ABI.
    unsafe {
        lapack::zgeqrf(
            lapack_dimension,
            lapack_dimension,
            &mut factors,
            lapack_dimension,
            &mut tau,
            &mut work_query,
            -1,
            &mut info,
        );
    }
    check_info(info)?;
    let work_length = workspace_length(work_query[0].re)?;
    let mut work = vec![Complex64::new(0.0, 0.0); work_length];

    // SAFETY: the matrix, reflector, and queried workspace buffers have the
    // lengths required by zgeqrf for a square `dimension` matrix.
    unsafe {
        lapack::zgeqrf(
            lapack_dimension,
            lapack_dimension,
            &mut factors,
            lapack_dimension,
            &mut tau,
            &mut work,
            i32::try_from(work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut info,
        );
    }
    check_info(info)?;
    let diagonal = (0..dimension)
        .map(|index| factors[index + index * dimension])
        .collect();
    let first_superdiagonal = (0..dimension.saturating_sub(1))
        .map(|index| factors[index + (index + 1) * dimension])
        .collect();

    work_query[0] = Complex64::new(0.0, 0.0);
    // SAFETY: the packed reflectors and tau buffer are the successful output
    // of zgeqrf, and the one-element workspace is valid in query mode.
    unsafe {
        lapack::zungqr(
            lapack_dimension,
            lapack_dimension,
            lapack_dimension,
            &mut factors,
            lapack_dimension,
            &tau,
            &mut work_query,
            -1,
            &mut info,
        );
    }
    check_info(info)?;
    let work_length = workspace_length(work_query[0].re)?;
    work.resize(work_length, Complex64::new(0.0, 0.0));

    // SAFETY: zungqr receives the validated packed reflectors, tau buffer,
    // and its own queried workspace allocation.
    unsafe {
        lapack::zungqr(
            lapack_dimension,
            lapack_dimension,
            lapack_dimension,
            &mut factors,
            lapack_dimension,
            &tau,
            &mut work,
            i32::try_from(work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut info,
        );
    }
    check_info(info)?;
    let factors = &factors;
    let unitary = (0..dimension)
        .flat_map(|row| (0..dimension).map(move |column| factors[row + column * dimension]))
        .collect();
    Ok(ComplexQr {
        unitary,
        diagonal,
        first_superdiagonal,
    })
}

/// Computes the full SVD of a square complex matrix with LAPACK's
/// divide-and-conquer driver.
pub fn complex_svd(dimension: usize, row_major_entries: &[Complex64]) -> Result<ComplexSvd, Error> {
    let entry_count = dimension
        .checked_mul(dimension)
        .ok_or(Error::DimensionTooLarge)?;
    if row_major_entries.len() != entry_count {
        return Err(Error::InvalidInputLength {
            expected: entry_count,
            actual: row_major_entries.len(),
        });
    }
    if dimension == 0 {
        return Ok(ComplexSvd {
            left_vectors: Vec::new(),
            singular_values: Vec::new(),
            right_vectors_adjoint: Vec::new(),
        });
    }
    let lapack_dimension = i32::try_from(dimension).map_err(|_| Error::DimensionTooLarge)?;
    let mut matrix = vec![Complex64::new(0.0, 0.0); entry_count];
    let mut singular_values = vec![0.0; dimension];
    let mut left_vectors = vec![Complex64::new(0.0, 0.0); entry_count];
    let mut right_vectors_adjoint = vec![Complex64::new(0.0, 0.0); entry_count];
    let mut work_query = [Complex64::new(0.0, 0.0)];
    let real_work_length = 5usize
        .checked_mul(entry_count)
        .and_then(|value| value.checked_add(7 * dimension))
        .ok_or(Error::DimensionTooLarge)?;
    let mut real_work = vec![0.0; real_work_length.max(1)];
    let mut integer_work = vec![
        0;
        8usize
            .checked_mul(dimension)
            .ok_or(Error::DimensionTooLarge)?
    ];
    let mut info = 0;

    // SAFETY: dimensions and all fixed-size buffers satisfy zgesdd's
    // workspace-query ABI. The real and integer work arrays use the
    // documented divide-and-conquer bounds.
    unsafe {
        complex_gesdd(
            b'A',
            lapack_dimension,
            lapack_dimension,
            &mut matrix,
            lapack_dimension,
            &mut singular_values,
            &mut left_vectors,
            lapack_dimension,
            &mut right_vectors_adjoint,
            lapack_dimension,
            &mut work_query,
            -1,
            &mut real_work,
            &mut integer_work,
            &mut info,
        );
    }
    check_info(info)?;
    let work_length = workspace_length(work_query[0].re)?;
    let mut work = vec![Complex64::new(0.0, 0.0); work_length];

    // The query does not depend on matrix entries. Use a separate zero matrix
    // so no backend can carry data-dependent query state into the actual SVD.
    matrix = (0..dimension)
        .flat_map(|column| {
            (0..dimension).map(move |row| row_major_entries[row * dimension + column])
        })
        .collect();
    // SAFETY: zgesdd receives square column-major buffers and its queried
    // complex workspace plus documented real and integer work allocations.
    unsafe {
        complex_gesdd(
            b'A',
            lapack_dimension,
            lapack_dimension,
            &mut matrix,
            lapack_dimension,
            &mut singular_values,
            &mut left_vectors,
            lapack_dimension,
            &mut right_vectors_adjoint,
            lapack_dimension,
            &mut work,
            i32::try_from(work_length).map_err(|_| Error::InvalidWorkspace)?,
            &mut real_work,
            &mut integer_work,
            &mut info,
        );
    }
    check_info(info)?;
    let left_vectors_column_major = left_vectors;
    let right_vectors_adjoint_column_major = right_vectors_adjoint;
    let left_vectors = (0..dimension)
        .flat_map(|row| {
            let left_vectors_column_major = &left_vectors_column_major;
            (0..dimension).map(move |column| left_vectors_column_major[row + column * dimension])
        })
        .collect();
    let right_vectors_adjoint = (0..dimension)
        .flat_map(|row| {
            let right_vectors_adjoint_column_major = &right_vectors_adjoint_column_major;
            (0..dimension)
                .map(move |column| right_vectors_adjoint_column_major[row + column * dimension])
        })
        .collect();
    Ok(ComplexSvd {
        left_vectors,
        singular_values,
        right_vectors_adjoint,
    })
}

fn workspace_length(value: f64) -> Result<usize, Error> {
    if !value.is_finite() || value < 1.0 || value > i32::MAX.into() {
        return Err(Error::InvalidWorkspace);
    }
    Ok(value as usize)
}

fn check_info(info: i32) -> Result<(), Error> {
    if info < 0 {
        Err(Error::InvalidLapackArgument { argument: -info })
    } else if info > 0 {
        Err(Error::NoConvergence { detail: info })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divide_and_conquer_driver_diagonalizes_a_real_dimer() {
        let solution = hermitian_eigensystem(
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.5, 0.0),
                Complex64::new(-1.5, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();
        assert_eq!(solution.eigenvalues(), &[-1.5, 1.5]);
        let vectors = solution.eigenvectors_column_major();
        for column in 0..2 {
            for row in 0..2 {
                let residual = (0..2)
                    .map(|inner| {
                        let matrix = if row == inner {
                            Complex64::new(0.0, 0.0)
                        } else {
                            Complex64::new(-1.5, 0.0)
                        };
                        matrix * vectors[inner + column * 2]
                    })
                    .sum::<Complex64>()
                    - vectors[row + column * 2] * solution.eigenvalues()[column];
                assert!(residual.norm() < 1.0e-12);
            }
        }
    }

    #[test]
    fn qr_driver_returns_a_unitary_factor() {
        let decomposition = complex_qr(
            2,
            &[
                Complex64::new(1.0, 2.0),
                Complex64::new(3.0, -1.0),
                Complex64::new(-2.0, 0.5),
                Complex64::new(0.0, 4.0),
            ],
        )
        .unwrap();
        let unitary = decomposition.unitary_row_major();
        for row in 0..2 {
            for column in 0..2 {
                let overlap = (0..2)
                    .map(|inner| unitary[row * 2 + inner] * unitary[column * 2 + inner].conj())
                    .sum::<Complex64>();
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!((overlap - Complex64::new(expected, 0.0)).norm() < 1.0e-12);
            }
        }
    }

    #[test]
    fn divide_and_conquer_svd_reconstructs_a_complex_matrix() {
        let matrix = [
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(-2.0, 0.5),
            Complex64::new(0.0, 4.0),
        ];
        let decomposition = complex_svd(2, &matrix).unwrap();
        let left = decomposition.left_vectors_row_major();
        let right_adjoint = decomposition.right_vectors_adjoint_row_major();
        for row in 0..2 {
            for column in 0..2 {
                let reconstructed = (0..2)
                    .map(|inner| {
                        left[row * 2 + inner]
                            * decomposition.singular_values()[inner]
                            * right_adjoint[inner * 2 + column]
                    })
                    .sum::<Complex64>();
                assert!((reconstructed - matrix[row * 2 + column]).norm() < 1.0e-12);
            }
        }
    }
}
