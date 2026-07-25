//! Propagating Bloch modes of nearest-cell periodic leads.

use std::fmt;

use nalgebra::DMatrix;

use crate::decomposition::{
    eigenvectors_from_generalized_schur, generalized_schur, DecompositionError,
};
use crate::spectrum::hermitian_eigensystem;
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
    stabilized_vectors: ComplexMatrix,
    stabilized_vectors_lambda_inverse: ComplexMatrix,
    square_root_hopping: ComplexMatrix,
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

    /// First half of the stabilized translation eigenvectors.
    #[must_use]
    pub const fn stabilized_vectors(&self) -> &ComplexMatrix {
        &self.stabilized_vectors
    }

    /// Second half of the stabilized translation eigenvectors.
    #[must_use]
    pub const fn stabilized_vectors_lambda_inverse(&self) -> &ComplexMatrix {
        &self.stabilized_vectors_lambda_inverse
    }

    /// Square-root hopping factor defining the stabilized basis.
    #[must_use]
    pub const fn square_root_hopping(&self) -> &ComplexMatrix {
        &self.square_root_hopping
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
            stabilized_vectors: ComplexMatrix::zeros(0, 0),
            stabilized_vectors_lambda_inverse: ComplexMatrix::zeros(0, 0),
            square_root_hopping: ComplexMatrix::zeros(0, 0),
        });
    }
    let hopping_norm = inter_cell_hopping
        .as_slice()
        .iter()
        .map(Complex64::norm_sqr)
        .sum::<f64>()
        .sqrt();
    if hopping_norm == 0.0 {
        return Ok(PropagatingLeadModes {
            wave_functions: ComplexMatrix::zeros(dimension, 0),
            velocities: Vec::new(),
            momenta: Vec::new(),
            incoming_count: 0,
            stabilized_vectors: ComplexMatrix::zeros(dimension, 0),
            stabilized_vectors_lambda_inverse: ComplexMatrix::zeros(dimension, 0),
            square_root_hopping: ComplexMatrix::zeros(dimension, 0),
        });
    }

    let cell_norm = cell_hamiltonian
        .as_slice()
        .iter()
        .map(Complex64::norm_sqr)
        .sum::<f64>()
        .sqrt();
    let hopping_backend = backend(inter_cell_hopping);
    let singular_values = hopping_backend.clone().svd(false, false).singular_values;
    let singular_tolerance = singular_values[0] * 1.0e-10;
    let hopping_rank = singular_values
        .iter()
        .filter(|&&value| value > singular_tolerance)
        .count();
    let raw_modes = if hopping_rank < dimension {
        reduced_raw_modes(cell_hamiltonian, inter_cell_hopping, hopping_rank)?
    } else {
        regular_raw_modes(
            cell_hamiltonian,
            inter_cell_hopping,
            cell_norm.max(hopping_norm),
        )?
    };

    let mut candidates = Vec::new();
    let mut assigned = vec![false; raw_modes.len()];
    for seed in 0..raw_modes.len() {
        if assigned[seed] {
            continue;
        }
        let group = (0..raw_modes.len())
            .filter(|&mode| {
                !assigned[mode]
                    && (raw_modes[mode].0 - raw_modes[seed].0).norm() <= UNIT_CIRCLE_TOLERANCE
            })
            .collect::<Vec<_>>();
        for &mode in &group {
            assigned[mode] = true;
        }
        let basis = orthonormalize_columns(
            &group
                .iter()
                .map(|&mode| raw_modes[mode].1.clone())
                .collect::<Vec<_>>(),
        );
        if basis.is_empty() {
            continue;
        }
        let bloch_factor = raw_modes[seed].0 / raw_modes[seed].0.norm();
        let velocity_images = basis
            .iter()
            .map(|wave| {
                (0..dimension)
                    .map(|row| {
                        Complex64::new(0.0, 1.0)
                            * (0..dimension)
                                .map(|column| {
                                    let backward = inter_cell_hopping.as_slice()
                                        [column * dimension + row]
                                        .conj()
                                        * bloch_factor;
                                    let forward = inter_cell_hopping.as_slice()
                                        [row * dimension + column]
                                        / bloch_factor;
                                    (backward - forward) * wave[column]
                                })
                                .sum::<Complex64>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut velocity_matrix = ComplexMatrix::new(
            basis.len(),
            basis.len(),
            (0..basis.len())
                .flat_map(|row| {
                    let basis = &basis;
                    let velocity_images = &velocity_images;
                    (0..basis.len()).map(move |column| {
                        basis[row]
                            .iter()
                            .zip(&velocity_images[column])
                            .map(|(left, right)| left.conj() * right)
                            .sum::<Complex64>()
                    })
                })
                .collect(),
        )?;
        for row in 0..basis.len() {
            for column in 0..row {
                let value = 0.5
                    * (velocity_matrix.get(row, column)?
                        + velocity_matrix.get(column, row)?.conj());
                velocity_matrix.set(row, column, value)?;
                velocity_matrix.set(column, row, value.conj())?;
            }
            velocity_matrix.set(
                row,
                row,
                Complex64::new(velocity_matrix.get(row, row)?.re, 0.0),
            )?;
        }
        let velocity_scale = velocity_matrix
            .as_slice()
            .iter()
            .map(|value| value.norm())
            .fold(0.0_f64, f64::max);
        if velocity_scale == 0.0 {
            continue;
        }
        let scaled_velocity_matrix = ComplexMatrix::new(
            basis.len(),
            basis.len(),
            velocity_matrix
                .as_slice()
                .iter()
                .map(|value| value / velocity_scale)
                .collect(),
        )?;
        let velocity_decomposition = hermitian_eigensystem(&scaled_velocity_matrix, 1.0e-10)
            .map_err(|_| LeadModeError::DecompositionFailure)?;
        for mode in 0..basis.len() {
            let velocity = velocity_decomposition.eigenvalues()[mode] * velocity_scale;
            if velocity.abs() <= VELOCITY_TOLERANCE * cell_norm.max(hopping_norm).max(1.0) {
                continue;
            }
            let rotation = velocity_decomposition.eigenvectors();
            let mut wave = (0..dimension)
                .map(|row| {
                    (0..basis.len())
                        .map(|column| {
                            basis[column][row] * rotation.as_slice()[column * basis.len() + mode]
                        })
                        .sum::<Complex64>()
                })
                .collect::<Vec<_>>();
            let scale = velocity.abs().sqrt();
            for value in &mut wave {
                *value /= scale;
            }
            candidates.push((velocity, bloch_factor.arg(), bloch_factor, wave));
        }
    }

    candidates.sort_by(|left, right| {
        let left_momentum = if left.0 > 0.0 { -left.1 } else { left.1 };
        let right_momentum = if right.0 > 0.0 { -right.1 } else { right.1 };
        (left.0 > 0.0)
            .cmp(&(right.0 > 0.0))
            .then_with(|| left_momentum.total_cmp(&right_momentum))
            .then_with(|| left.0.abs().total_cmp(&right.0.abs()))
    });
    let incoming_count = candidates
        .iter()
        .filter(|(velocity, _, _, _)| *velocity < 0.0)
        .count();
    let mode_count = candidates.len();
    let wave_functions = ComplexMatrix::new(
        dimension,
        mode_count,
        (0..dimension)
            .flat_map(|row| candidates.iter().map(move |(_, _, _, wave)| wave[row]))
            .collect(),
    )?;
    let hopping_scale = hopping_norm.sqrt();
    let stabilized_vectors = ComplexMatrix::new(
        dimension,
        mode_count,
        (0..dimension)
            .flat_map(|row| {
                candidates.iter().map(move |(_, _, bloch_factor, wave)| {
                    (0..dimension)
                        .map(|column| {
                            inter_cell_hopping.as_slice()[column * dimension + row].conj()
                                * wave[column]
                        })
                        .sum::<Complex64>()
                        * bloch_factor
                        / hopping_scale
                })
            })
            .collect(),
    )?;
    let stabilized_vectors_lambda_inverse = ComplexMatrix::new(
        dimension,
        mode_count,
        (0..dimension)
            .flat_map(|row| {
                candidates
                    .iter()
                    .map(move |(_, _, _, wave)| wave[row] * hopping_scale)
            })
            .collect(),
    )?;
    let mut square_root_hopping = ComplexMatrix::zeros(dimension, dimension);
    for index in 0..dimension {
        square_root_hopping.set(index, index, Complex64::new(hopping_scale, 0.0))?;
    }

    Ok(PropagatingLeadModes {
        wave_functions,
        velocities: candidates
            .iter()
            .map(|(velocity, _, _, _)| *velocity)
            .collect(),
        momenta: candidates
            .iter()
            .map(|(_, momentum, _, _)| *momentum)
            .collect(),
        incoming_count,
        stabilized_vectors,
        stabilized_vectors_lambda_inverse,
        square_root_hopping,
    })
}

fn regular_raw_modes(
    cell_hamiltonian: &ComplexMatrix,
    inter_cell_hopping: &ComplexMatrix,
    pencil_scale: f64,
) -> Result<Vec<(Complex64, Vec<Complex64>)>, LeadModeError> {
    let dimension = cell_hamiltonian.rows();
    let pencil_dimension = 2 * dimension;
    let mut left = ComplexMatrix::zeros(pencil_dimension, pencil_dimension);
    let mut right = ComplexMatrix::zeros(pencil_dimension, pencil_dimension);
    for row in 0..dimension {
        for column in 0..dimension {
            left.set(
                row,
                column,
                -cell_hamiltonian.as_slice()[row * dimension + column] / pencil_scale,
            )?;
            left.set(
                row,
                dimension + column,
                -inter_cell_hopping.as_slice()[row * dimension + column] / pencil_scale,
            )?;
            right.set(
                row,
                column,
                inter_cell_hopping.as_slice()[column * dimension + row].conj() / pencil_scale,
            )?;
        }
        left.set(dimension + row, row, Complex64::new(1.0, 0.0))?;
        right.set(dimension + row, dimension + row, Complex64::new(1.0, 0.0))?;
    }
    raw_modes_from_pencil(&left, &right, |mode, _, vectors| {
        (0..dimension)
            .map(|row| vectors.as_slice()[row * pencil_dimension + mode])
            .collect()
    })
}

fn reduced_raw_modes(
    cell_hamiltonian: &ComplexMatrix,
    inter_cell_hopping: &ComplexMatrix,
    rank: usize,
) -> Result<Vec<(Complex64, Vec<Complex64>)>, LeadModeError> {
    let dimension = cell_hamiltonian.rows();
    let decomposition = backend(inter_cell_hopping).svd(true, true);
    let left_vectors = decomposition.u.ok_or(LeadModeError::DecompositionFailure)?;
    let right_vectors = decomposition
        .v_t
        .ok_or(LeadModeError::DecompositionFailure)?
        .adjoint();
    let mut left_factor = DMatrix::<Complex64>::zeros(dimension, rank);
    let mut right_factor = DMatrix::<Complex64>::zeros(dimension, rank);
    for column in 0..rank {
        let scale = decomposition.singular_values[column].sqrt();
        for row in 0..dimension {
            left_factor[(row, column)] = left_vectors[(row, column)] * scale;
            right_factor[(row, column)] = right_vectors[(row, column)] * scale;
        }
    }

    let stabilizer = &left_factor * left_factor.adjoint() + &right_factor * right_factor.adjoint();
    let stabilized_cell = backend(cell_hamiltonian) + stabilizer * Complex64::new(0.0, 1.0);
    let cell_inverse = stabilized_cell
        .try_inverse()
        .ok_or(LeadModeError::DecompositionFailure)?;
    let inverse_right = &cell_inverse * &right_factor;
    let left_inverse_right = left_factor.adjoint() * &inverse_right;
    let right_inverse_right = right_factor.adjoint() * inverse_right;
    let inverse_left = &cell_inverse * &left_factor;
    let left_inverse_left = left_factor.adjoint() * &inverse_left;
    let right_inverse_left = right_factor.adjoint() * inverse_left;

    let pencil_dimension = 2 * rank;
    let mut left = DMatrix::<Complex64>::zeros(pencil_dimension, pencil_dimension);
    let mut right = DMatrix::<Complex64>::zeros(pencil_dimension, pencil_dimension);
    for index in 0..rank {
        left[(rank + index, index)] = Complex64::new(1.0, 0.0);
        right[(index, rank + index)] = Complex64::new(-1.0, 0.0);
    }
    add_block(
        &mut left,
        0,
        0,
        &left_inverse_right,
        Complex64::new(0.0, -1.0),
    );
    add_block(
        &mut left,
        0,
        rank,
        &left_inverse_right,
        Complex64::new(1.0, 0.0),
    );
    add_block(
        &mut left,
        rank,
        0,
        &right_inverse_right,
        Complex64::new(0.0, -1.0),
    );
    add_block(
        &mut left,
        rank,
        rank,
        &right_inverse_right,
        Complex64::new(1.0, 0.0),
    );
    add_block(
        &mut right,
        0,
        0,
        &left_inverse_left,
        Complex64::new(-1.0, 0.0),
    );
    add_block(
        &mut right,
        0,
        rank,
        &left_inverse_left,
        Complex64::new(0.0, 1.0),
    );
    add_block(
        &mut right,
        rank,
        0,
        &right_inverse_left,
        Complex64::new(-1.0, 0.0),
    );
    add_block(
        &mut right,
        rank,
        rank,
        &right_inverse_left,
        Complex64::new(0.0, 1.0),
    );

    let left = owned(&left)?;
    let right = owned(&right)?;
    raw_modes_from_pencil(&left, &right, |mode, inverse_bloch, vectors| {
        let projected = backend(vectors).column(mode).into_owned();
        let first = projected.rows(0, rank).into_owned();
        let second = projected.rows(rank, rank).into_owned();
        let rhs = -&left_factor * (&first * inverse_bloch) - &right_factor * &second
            + (&right_factor * &first + &left_factor * (&second * inverse_bloch))
                * Complex64::new(0.0, 1.0);
        let wave = &cell_inverse * rhs;
        wave.iter().copied().collect()
    })
    .map(|modes| {
        modes
            .into_iter()
            .map(|(inverse_bloch, wave)| (Complex64::new(1.0, 0.0) / inverse_bloch, wave))
            .collect()
    })
}

fn raw_modes_from_pencil<F>(
    left: &ComplexMatrix,
    right: &ComplexMatrix,
    extract_wave: F,
) -> Result<Vec<(Complex64, Vec<Complex64>)>, LeadModeError>
where
    F: Fn(usize, Complex64, &ComplexMatrix) -> Vec<Complex64>,
{
    let decomposition = generalized_schur(left, right)?;
    let pencil_dimension = left.rows();
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
    let mut raw_modes = Vec::new();
    for mode in 0..pencil_dimension {
        let beta = decomposition.beta()[mode];
        if beta.norm() == 0.0 {
            continue;
        }
        let eigenvalue = decomposition.alpha()[mode] / beta;
        if !eigenvalue.re.is_finite()
            || !eigenvalue.im.is_finite()
            || (eigenvalue.norm() - 1.0).abs() > UNIT_CIRCLE_TOLERANCE
        {
            continue;
        }
        let mut wave = extract_wave(mode, eigenvalue, right_vectors);
        let norm = wave.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        if norm <= 1.0e-14 {
            continue;
        }
        for value in &mut wave {
            *value /= norm;
        }
        raw_modes.push((eigenvalue, wave));
    }
    Ok(raw_modes)
}

fn add_block(
    target: &mut DMatrix<Complex64>,
    row_offset: usize,
    column_offset: usize,
    source: &DMatrix<Complex64>,
    factor: Complex64,
) {
    for row in 0..source.nrows() {
        for column in 0..source.ncols() {
            target[(row_offset + row, column_offset + column)] += factor * source[(row, column)];
        }
    }
}

fn backend(matrix: &ComplexMatrix) -> DMatrix<Complex64> {
    DMatrix::from_row_slice(matrix.rows(), matrix.columns(), matrix.as_slice())
}

fn owned(matrix: &DMatrix<Complex64>) -> Result<ComplexMatrix, LeadModeError> {
    ComplexMatrix::new(
        matrix.nrows(),
        matrix.ncols(),
        (0..matrix.nrows())
            .flat_map(|row| (0..matrix.ncols()).map(move |column| matrix[(row, column)]))
            .collect(),
    )
    .map_err(Into::into)
}

fn orthonormalize_columns(columns: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let mut basis: Vec<Vec<Complex64>> = Vec::new();
    for column in columns {
        let mut vector = column.clone();
        for existing in &basis {
            let overlap = existing
                .iter()
                .zip(&vector)
                .map(|(left, right)| left.conj() * right)
                .sum::<Complex64>();
            for (value, direction) in vector.iter_mut().zip(existing) {
                *value -= overlap * direction;
            }
        }
        let norm = vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        if norm <= 1.0e-10 {
            continue;
        }
        for value in &mut vector {
            *value /= norm;
        }
        basis.push(vector);
    }
    basis
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
        let vectors = modes.stabilized_vectors();
        let inverse = modes.stabilized_vectors_lambda_inverse();
        for mode in 0..2 {
            let current = Complex64::new(0.0, 1.0)
                * vectors.get(0, mode).unwrap().conj()
                * inverse.get(0, mode).unwrap();
            let current = current + current.conj();
            let expected = if mode == 0 { 1.0 } else { -1.0 };
            assert!((current.re - expected).abs() < 1.0e-10);
        }
        assert!(
            (modes.square_root_hopping().get(0, 0).unwrap().re - 0.7_f64.sqrt()).abs() < 1.0e-12
        );
    }

    #[test]
    fn degenerate_unit_bloch_factor_is_resolved_by_current() {
        let cell = ComplexMatrix::new(
            2,
            2,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();
        let hopping = ComplexMatrix::new(
            2,
            2,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();
        let modes = propagating_modes(&cell, &hopping).unwrap();
        assert_eq!(modes.incoming_count(), 1);
        assert!((modes.velocities()[0] + 1.0).abs() < 1.0e-12);
        assert!((modes.velocities()[1] - 1.0).abs() < 1.0e-12);
        assert!(modes
            .momenta()
            .iter()
            .all(|momentum| momentum.abs() < 1.0e-12));
    }

    #[test]
    fn momenta_are_invariant_under_global_energy_rescaling() {
        let baseline = propagating_modes(
            &ComplexMatrix::scalar(Complex64::new(0.3, 0.0)),
            &ComplexMatrix::scalar(Complex64::new(0.7, 0.0)),
        )
        .unwrap();
        let scale = 1.0e20;
        let scaled = propagating_modes(
            &ComplexMatrix::scalar(Complex64::new(0.3 * scale, 0.0)),
            &ComplexMatrix::scalar(Complex64::new(0.7 * scale, 0.0)),
        )
        .unwrap();
        for (baseline, scaled) in baseline.momenta().iter().zip(scaled.momenta()) {
            assert!((baseline - scaled).abs() < 1.0e-12);
        }
        for (baseline, scaled) in baseline.velocities().iter().zip(scaled.velocities()) {
            assert!((scaled / scale - baseline).abs() < 1.0e-10);
        }
    }
}
