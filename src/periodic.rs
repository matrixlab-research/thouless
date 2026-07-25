//! Periodic-boundary folding into explicit Bloch-momentum parameters.

use std::error::Error;
use std::fmt;

use crate::{Complex64, ComplexMatrix};

/// One matrix contribution carrying a lattice translation.
#[derive(Clone, Debug, PartialEq)]
pub struct PeriodicTerm {
    value: ComplexMatrix,
    translation: Vec<i64>,
    include_adjoint: bool,
}

impl PeriodicTerm {
    /// Creates a periodic matrix contribution.
    #[must_use]
    pub fn new(
        value: ComplexMatrix,
        translation: impl Into<Vec<i64>>,
        include_adjoint: bool,
    ) -> Self {
        Self {
            value,
            translation: translation.into(),
            include_adjoint,
        }
    }

    /// Matrix value before its Bloch phase is applied.
    #[must_use]
    pub const fn value(&self) -> &ComplexMatrix {
        &self.value
    }

    /// Translation in integer symmetry coordinates.
    #[must_use]
    pub fn translation(&self) -> &[i64] {
        &self.translation
    }

    /// Whether the Hermitian-conjugate contribution is included.
    #[must_use]
    pub const fn includes_adjoint(&self) -> bool {
        self.include_adjoint
    }
}

/// Errors raised while folding periodic terms.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PeriodicFoldError {
    /// A translation has a different dimension than the momentum.
    MomentumDimension {
        /// Number of translation components.
        translation: usize,
        /// Number of momentum components.
        momentum: usize,
    },
    /// Contributions have different matrix shapes.
    MatrixShape {
        /// Expected shape.
        expected: (usize, usize),
        /// Actual shape.
        actual: (usize, usize),
    },
    /// An adjoint pair requires a square matrix.
    AdjointRequiresSquare {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// A momentum is NaN or infinite.
    NonFiniteMomentum,
    /// No terms were supplied.
    NoTerms,
}

impl fmt::Display for PeriodicFoldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MomentumDimension {
                translation,
                momentum,
            } => write!(
                formatter,
                "translation dimension {translation} differs from momentum dimension {momentum}"
            ),
            Self::MatrixShape { expected, actual } => write!(
                formatter,
                "periodic contribution has shape {actual:?}, expected {expected:?}"
            ),
            Self::AdjointRequiresSquare { rows, columns } => write!(
                formatter,
                "an adjoint-paired contribution must be square, got ({rows}, {columns})"
            ),
            Self::NonFiniteMomentum => write!(formatter, "Bloch momentum must be finite"),
            Self::NoTerms => write!(formatter, "at least one periodic contribution is required"),
        }
    }
}

impl Error for PeriodicFoldError {}

/// Returns `exp(i R · k)` for an integer translation and reduced momentum.
pub fn bloch_phase(translation: &[i64], momentum: &[f64]) -> Result<Complex64, PeriodicFoldError> {
    if translation.len() != momentum.len() {
        return Err(PeriodicFoldError::MomentumDimension {
            translation: translation.len(),
            momentum: momentum.len(),
        });
    }
    if momentum.iter().any(|value| !value.is_finite()) {
        return Err(PeriodicFoldError::NonFiniteMomentum);
    }
    let angle = translation
        .iter()
        .zip(momentum)
        .map(|(translation, momentum)| *translation as f64 * momentum)
        .sum();
    Ok(Complex64::from_polar(1.0, angle))
}

/// Folds and sums matrix contributions at one Bloch momentum.
///
/// When `include_adjoint` is set on a term, the result contains both
/// `exp(i R·k) V` and its Hermitian conjugate.  This is the operation used
/// when a periodic hopping folds onto an onsite block.
pub fn fold_terms(
    terms: &[PeriodicTerm],
    momentum: &[f64],
) -> Result<ComplexMatrix, PeriodicFoldError> {
    let first = terms.first().ok_or(PeriodicFoldError::NoTerms)?;
    let shape = first.value().shape();
    let mut data = vec![Complex64::new(0.0, 0.0); shape.0 * shape.1];
    for term in terms {
        if term.value().shape() != shape {
            return Err(PeriodicFoldError::MatrixShape {
                expected: shape,
                actual: term.value().shape(),
            });
        }
        if term.includes_adjoint() && shape.0 != shape.1 {
            return Err(PeriodicFoldError::AdjointRequiresSquare {
                rows: shape.0,
                columns: shape.1,
            });
        }
        let phase = bloch_phase(term.translation(), momentum)?;
        for (result, value) in data.iter_mut().zip(term.value().as_slice()) {
            *result += phase * value;
        }
        if term.includes_adjoint() {
            for row in 0..shape.0 {
                for column in 0..shape.1 {
                    data[row * shape.1 + column] +=
                        (phase * term.value().as_slice()[column * shape.1 + row]).conj();
                }
            }
        }
    }
    ComplexMatrix::new(shape.0, shape.1, data).map_err(|_| PeriodicFoldError::MatrixShape {
        expected: shape,
        actual: shape,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloch_phase_composes_integer_translations() {
        let momentum = [0.37, -0.91];
        let left = bloch_phase(&[2, -1], &momentum).unwrap();
        let right = bloch_phase(&[-3, 4], &momentum).unwrap();
        let combined = bloch_phase(&[-1, 3], &momentum).unwrap();
        assert!((left * right - combined).norm() < 1.0e-14);
    }

    #[test]
    fn hopping_folded_onto_onsite_is_hermitian() {
        let hopping = ComplexMatrix::new(
            2,
            2,
            vec![
                Complex64::new(0.2, 0.3),
                Complex64::new(-0.4, 0.7),
                Complex64::new(0.8, -0.1),
                Complex64::new(-0.2, 0.5),
            ],
        )
        .unwrap();
        let folded = fold_terms(&[PeriodicTerm::new(hopping, [2, -1], true)], &[0.4, 0.2]).unwrap();
        assert!(folded.is_hermitian(1.0e-14).unwrap());
    }

    #[test]
    fn opposite_scalar_hoppings_cancel_at_zero_momentum() {
        let positive =
            PeriodicTerm::new(ComplexMatrix::scalar(Complex64::new(0.0, 1.0)), [1], false);
        let negative = PeriodicTerm::new(
            ComplexMatrix::scalar(Complex64::new(0.0, -1.0)),
            [-1],
            false,
        );
        let folded = fold_terms(&[positive, negative], &[0.0]).unwrap();
        assert!(folded.as_slice()[0].norm() < 1.0e-14);
    }
}
