//! Propagating Bloch modes of nearest-cell periodic leads.

use std::fmt;

use crate::decomposition::{
    eigenvectors_from_generalized_schur, generalized_schur, DecompositionError,
};
use crate::{Complex64, ComplexMatrix, MatrixError};

const UNIT_CIRCLE_TOLERANCE: f64 = 1.0e-7;
const VELOCITY_TOLERANCE: f64 = 1.0e-10;

/// Current-normalized propagating lead modes.
#[derive(Clone, Debug, PartialEq)]
pub struct PropagatingLeadModes {
    wave_functions: ComplexMatrix,
    velocities: Vec<f64>,
    momenta: Vec<f64>,
    incoming_count: usize,
}

impl PropagatingLeadModes {
    /// Current-normalized wave functions as columns.
    #[must_use]
    pub const fn wave_functions(&self) -> &ComplexMatrix {
        &self.wave_functions
    }

    /// Group velocities in mode order.
    #[must_use]
    pub fn velocities(&self) -> &[f64] {
        &self.velocities
    }

    /// Bloch momenta in mode order.
    #[must_use]
    pub fn momenta(&self) -> &[f64] {
        &self.momenta
    }

    /// Number of negative-velocity incoming modes.
    #[must_use]
    pub const fn incoming_count(&self) -> usize {
        self.incoming_count
    }
}

/// Failures raised by propagating-mode analysis.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LeadModeError {
    /// Cell and hopping matrices do not share one square shape.
    InvalidShape,
    /// The cell Hamiltonian is not Hermitian.
    NonHermitianCell,
    /// Generalized eigendecomposition failed.
    DecompositionFailure,
    /// Matrix construction failed.
    Matrix(MatrixError),
}

impl fmt::Display for LeadModeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => write!(
                formatter,
                "cell Hamiltonian and hopping must share one square shape"
            ),
            Self::NonHermitianCell => write!(formatter, "cell Hamiltonian is not Hermitian"),
            Self::DecompositionFailure => {
                write!(formatter, "lead generalized eigendecomposition failed")
            }
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for LeadModeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatrixError> for LeadModeError {
    fn from(error: MatrixError) -> Self {
        Self::Matrix(error)
    }
}

impl From<DecompositionError> for LeadModeError {
    fn from(_: DecompositionError) -> Self {
        Self::DecompositionFailure
    }
}

/// Solve current-carrying modes of
/// `H(k) = H₀ + T exp(-ik) + Tᴴ exp(ik)`.
pub fn propagating_modes(
    cell_hamiltonian: &ComplexMatrix,
    inter_cell_hopping: &ComplexMatrix,
) -> Result<PropagatingLeadModes, LeadModeError> {
    let dimension = cell_hamiltonian.rows();
    if cell_hamiltonian.columns() != dimension
        || inter_cell_hopping.shape() != (dimension, dimension)
    {
        return Err(LeadModeError::InvalidShape);
    }
    if !cell_hamiltonian.is_hermitian(1.0e-10)? {
        return Err(LeadModeError::NonHermitianCell);
    }
    if dimension == 0 {
        return Ok(PropagatingLeadModes {
            wave_functions: ComplexMatrix::zeros(0, 0),
            velocities: Vec::new(),
            momenta: Vec::new(),
            incoming_count: 0,
        });
    }

    let pencil_dimension = 2 * dimension;
    let mut left = ComplexMatrix::zeros(pencil_dimension, pencil_dimension);
    let mut right = ComplexMatrix::zeros(pencil_dimension, pencil_dimension);
    for row in 0..dimension {
        for column in 0..dimension {
            left.set(
                row,
                column,
                -cell_hamiltonian.as_slice()[row * dimension + column],
            )?;
            left.set(
                row,
                dimension + column,
                -inter_cell_hopping.as_slice()[row * dimension + column],
            )?;
            right.set(
                row,
                column,
                inter_cell_hopping.as_slice()[column * dimension + row].conj(),
            )?;
        }
        left.set(dimension + row, row, Complex64::new(1.0, 0.0))?;
        right.set(dimension + row, dimension + row, Complex64::new(1.0, 0.0))?;
    }

    let decomposition = generalized_schur(&left, &right)?;
    let selected = vec![true; pencil_dimension];
    let vectors = eigenvectors_from_generalized_schur(
        decomposition.left_form(),
        decomposition.right_form(),
        decomposition.left_vectors(),
        decomposition.right_vectors(),
        &selected,
        false,
        true,
    )?;
    let right_vectors = vectors
        .right()
        .expect("right generalized eigenvectors were requested");
    let mut candidates = Vec::new();
    for mode in 0..pencil_dimension {
        let alpha = decomposition.alpha()[mode];
        let beta = decomposition.beta()[mode];
        if beta.norm() == 0.0 {
            continue;
        }
        let bloch_factor = alpha / beta;
        if !bloch_factor.re.is_finite()
            || !bloch_factor.im.is_finite()
            || (bloch_factor.norm() - 1.0).abs() > UNIT_CIRCLE_TOLERANCE
        {
            continue;
        }
        let mut wave = (0..dimension)
            .map(|row| right_vectors.as_slice()[row * pencil_dimension + mode])
            .collect::<Vec<_>>();
        let norm = wave.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        for value in &mut wave {
            *value /= norm;
        }

        let velocity_operator_wave = (0..dimension)
            .map(|row| {
                (0..dimension)
                    .map(|column| {
                        let backward = inter_cell_hopping.as_slice()[column * dimension + row]
                            .conj()
                            * bloch_factor;
                        let forward =
                            inter_cell_hopping.as_slice()[row * dimension + column] / bloch_factor;
                        (backward - forward) * wave[column]
                    })
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        let velocity = (Complex64::new(0.0, 1.0)
            * wave
                .iter()
                .zip(&velocity_operator_wave)
                .map(|(left, right)| left.conj() * right)
                .sum::<Complex64>())
        .re;
        if velocity.abs() <= VELOCITY_TOLERANCE {
            continue;
        }
        let scale = velocity.abs().sqrt();
        for value in &mut wave {
            *value /= scale;
        }
        candidates.push((velocity, bloch_factor.arg(), wave));
    }

    candidates.sort_by(|left, right| {
        (left.0 > 0.0)
            .cmp(&(right.0 > 0.0))
            .then_with(|| left.1.total_cmp(&right.1))
    });
    let incoming_count = candidates
        .iter()
        .filter(|(velocity, _, _)| *velocity < 0.0)
        .count();
    let mode_count = candidates.len();
    let wave_functions = ComplexMatrix::new(
        dimension,
        mode_count,
        (0..dimension)
            .flat_map(|row| candidates.iter().map(move |(_, _, wave)| wave[row]))
            .collect(),
    )?;

    Ok(PropagatingLeadModes {
        wave_functions,
        velocities: candidates
            .iter()
            .map(|(velocity, _, _)| *velocity)
            .collect(),
        momenta: candidates
            .iter()
            .map(|(_, momentum, _)| *momentum)
            .collect(),
        incoming_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_chain_has_two_oppositely_directed_modes() {
        let modes = propagating_modes(
            &ComplexMatrix::scalar(Complex64::new(0.3, 0.0)),
            &ComplexMatrix::scalar(Complex64::new(0.7, 0.0)),
        )
        .unwrap();
        let momentum = (-0.3_f64 / 1.4).acos();
        let velocity = 1.4 * momentum.sin();
        assert_eq!(modes.incoming_count(), 1);
        assert_eq!(modes.velocities().len(), 2);
        assert!((modes.velocities()[0] + velocity).abs() < 1.0e-10);
        assert!((modes.velocities()[1] - velocity).abs() < 1.0e-10);
        assert!((modes.momenta()[0] - momentum).abs() < 1.0e-10);
        assert!((modes.momenta()[1] + momentum).abs() < 1.0e-10);
    }
}
