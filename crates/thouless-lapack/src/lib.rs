//! Audited safe boundary around the LAPACK routines used by Thouless.

#![deny(unsafe_op_in_unsafe_fn)]

extern crate lapack_src;

use num_complex::Complex64;

/// A column-major Hermitian eigensystem.
#[derive(Clone, Debug, PartialEq)]
pub struct HermitianEigensystem {
    eigenvalues: Vec<f64>,
    eigenvectors: Vec<Complex64>,
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
                "Hermitian matrix requires {expected} values; received {actual}"
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
}
