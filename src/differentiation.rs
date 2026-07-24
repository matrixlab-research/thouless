//! Reusable finite-difference operators for parameterized matrix families.

use crate::{Complex64, ComplexMatrix, DifferentiationError};

/// Finite-difference stencil selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DifferenceScheme {
    /// Centered interior differences with one-sided nonperiodic boundaries.
    Central,
    /// Forward differences with a backward nonperiodic final boundary.
    Forward,
}

/// Differentiates a uniformly sampled matrix-valued function.
pub fn finite_difference_uniform(
    samples: &[ComplexMatrix],
    step: f64,
    periodic: bool,
    scheme: DifferenceScheme,
) -> Result<Vec<ComplexMatrix>, DifferentiationError> {
    if samples.len() < 2 {
        return Err(DifferentiationError::InsufficientSamples);
    }
    if !step.is_finite() || step == 0.0 {
        return Err(DifferentiationError::InvalidStep);
    }
    let shape = samples[0].shape();
    if samples.iter().any(|sample| sample.shape() != shape) {
        return Err(DifferentiationError::ShapeMismatch);
    }

    let last = samples.len() - 1;
    (0..samples.len())
        .map(|index| match scheme {
            DifferenceScheme::Central if periodic => {
                let before = if index == 0 { last } else { index - 1 };
                let after = if index == last { 0 } else { index + 1 };
                difference(&samples[after], &samples[before], 2.0 * step)
            }
            DifferenceScheme::Central if index == 0 => difference(&samples[1], &samples[0], step),
            DifferenceScheme::Central if index == last => {
                difference(&samples[last], &samples[last - 1], step)
            }
            DifferenceScheme::Central => {
                difference(&samples[index + 1], &samples[index - 1], 2.0 * step)
            }
            DifferenceScheme::Forward if index < last => {
                difference(&samples[index + 1], &samples[index], step)
            }
            DifferenceScheme::Forward if periodic => difference(&samples[0], &samples[last], step),
            DifferenceScheme::Forward => difference(&samples[last], &samples[last - 1], step),
        })
        .collect()
}

fn difference(
    positive: &ComplexMatrix,
    negative: &ComplexMatrix,
    denominator: f64,
) -> Result<ComplexMatrix, DifferentiationError> {
    let factor = Complex64::new(1.0 / denominator, 0.0);
    let data = positive
        .as_slice()
        .iter()
        .zip(negative.as_slice())
        .map(|(left, right)| (*left - *right) * factor)
        .collect();
    Ok(ComplexMatrix::new(
        positive.rows(),
        positive.columns(),
        data,
    )?)
}
