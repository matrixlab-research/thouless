//! Projection of physical observables into sampled state subspaces.

use crate::{Complex64, ComplexMatrix, ObservableError};

/// Projects a real diagonal basis observable into a state subspace.
///
/// State vectors are rows of `states`; `diagonal[b]` is the observable value
/// of basis state `b`. The result is the Hermitian matrix
/// `O_mn = Σ_b ψ*_mb O_b ψ_nb`.
pub fn project_diagonal_observable(
    states: &ComplexMatrix,
    diagonal: &[f64],
) -> Result<ComplexMatrix, ObservableError> {
    if states.rows() == 0 || states.columns() == 0 {
        return Err(ObservableError::EmptyStateFrame);
    }
    if diagonal.len() != states.columns() {
        return Err(ObservableError::InvalidDiagonalLength {
            expected: states.columns(),
            actual: diagonal.len(),
        });
    }
    if diagonal.iter().any(|value| !value.is_finite()) {
        return Err(ObservableError::NonFiniteValue);
    }

    let state_count = states.rows();
    let basis_count = states.columns();
    let mut projected = ComplexMatrix::zeros(state_count, state_count);
    for bra in 0..state_count {
        for ket in 0..state_count {
            let value: Complex64 = (0..basis_count)
                .map(|basis| {
                    states.as_slice()[bra * basis_count + basis].conj()
                        * diagonal[basis]
                        * states.as_slice()[ket * basis_count + basis]
                })
                .sum();
            projected
                .set(bra, ket, value)
                .expect("projected indices are in bounds");
        }
    }
    Ok(projected)
}
