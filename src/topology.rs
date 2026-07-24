//! Gauge-invariant discrete topology for sampled state subspaces.

use nalgebra::linalg::Schur;
use nalgebra::DMatrix;

use crate::{Complex64, ComplexMatrix, TopologyError};

const SINGULAR_OVERLAP_TOLERANCE: f64 = 1.0e-14;

/// Computes the Abelian phase of a discrete Wilson line.
///
/// Each frame stores orthonormal state vectors as rows and basis amplitudes as
/// columns. For a closed Wilson loop, callers include the closure frame as the
/// final element. Only the subspace is used, so the result is invariant under
/// independent unitary rotations of the supplied frames.
pub fn wilson_line_phase(frames: &[ComplexMatrix]) -> Result<f64, TopologyError> {
    validate_frames(frames)?;
    let mut phase_product = Complex64::new(1.0, 0.0);
    for pair in frames.windows(2) {
        let determinant = overlap_determinant(&pair[0], &pair[1]);
        let norm = determinant.norm();
        if norm <= SINGULAR_OVERLAP_TOLERANCE {
            return Err(TopologyError::SingularOverlap);
        }
        phase_product *= determinant / norm;
    }
    Ok(-phase_product.arg())
}

/// Computes the oriented Abelian Berry flux through one plaquette.
///
/// Corners are ordered around the boundary and the first corner is appended
/// internally to close the loop.
pub fn plaquette_flux(corners: &[ComplexMatrix; 4]) -> Result<f64, TopologyError> {
    let frames = [
        corners[0].clone(),
        corners[1].clone(),
        corners[2].clone(),
        corners[3].clone(),
        corners[0].clone(),
    ];
    wilson_line_phase(&frames)
}

/// Returns the unitary parallel-transport factor between two state frames.
///
/// The overlap matrix is factorized as `M = U Σ V†`; the returned link is
/// `U V†`. This removes changes of norm and retains only transport within the
/// selected subspace.
pub fn parallel_transport_link(
    left: &ComplexMatrix,
    right: &ComplexMatrix,
) -> Result<ComplexMatrix, TopologyError> {
    validate_frames(&[left.clone(), right.clone()])?;
    let overlap = overlap_matrix(left, right);
    let decomposition = overlap.svd(true, true);
    if decomposition
        .singular_values
        .iter()
        .any(|value| *value <= SINGULAR_OVERLAP_TOLERANCE)
    {
        return Err(TopologyError::SingularOverlap);
    }
    let left_unitary = decomposition
        .u
        .expect("left singular vectors were requested");
    let right_adjoint = decomposition
        .v_t
        .expect("right singular vectors were requested");
    let link = left_unitary * right_adjoint;
    let mut entries = Vec::with_capacity(link.nrows() * link.ncols());
    for row in 0..link.nrows() {
        for column in 0..link.ncols() {
            entries.push(link[(row, column)]);
        }
    }
    ComplexMatrix::new(link.nrows(), link.ncols(), entries)
        .map_err(|_| TopologyError::IncompatibleFrames)
}

/// Returns the ordered Wilson-loop unitary for a sampled state-frame path.
pub fn wilson_loop_unitary(frames: &[ComplexMatrix]) -> Result<ComplexMatrix, TopologyError> {
    validate_frames(frames)?;
    let state_count = frames[0].rows();
    let mut loop_unitary = DMatrix::<Complex64>::identity(state_count, state_count);
    for pair in frames.windows(2) {
        let link = parallel_transport_link(&pair[0], &pair[1])?;
        let link = DMatrix::from_row_slice(state_count, state_count, link.as_slice());
        loop_unitary *= link;
    }
    let mut entries = Vec::with_capacity(state_count * state_count);
    for row in 0..state_count {
        for column in 0..state_count {
            entries.push(loop_unitary[(row, column)]);
        }
    }
    ComplexMatrix::new(state_count, state_count, entries)
        .map_err(|_| TopologyError::IncompatibleFrames)
}

/// Returns sorted negative phases of the Wilson-loop eigenvalues.
pub fn wilson_loop_eigenphases(frames: &[ComplexMatrix]) -> Result<Vec<f64>, TopologyError> {
    let loop_unitary = wilson_loop_unitary(frames)?;
    let matrix = DMatrix::from_row_slice(
        loop_unitary.rows(),
        loop_unitary.columns(),
        loop_unitary.as_slice(),
    );
    let eigenvalues = matrix
        .eigenvalues()
        .ok_or(TopologyError::EigendecompositionFailed)?;
    let mut phases: Vec<f64> = eigenvalues.iter().map(|value| -value.arg()).collect();
    phases.sort_by(f64::total_cmp);
    Ok(phases)
}

/// Converts a unitary parallel-transport link into a Hermitian connection.
///
/// The principal matrix logarithm is evaluated through the complex Schur
/// decomposition, giving `A = -log(U) / (i Δκ)`.
pub fn connection_from_link(
    link: &ComplexMatrix,
    coordinate_step: f64,
) -> Result<ComplexMatrix, TopologyError> {
    if link.rows() != link.columns() {
        return Err(TopologyError::NonSquareLink);
    }
    if !coordinate_step.is_finite() || coordinate_step == 0.0 {
        return Err(TopologyError::InvalidConnectionStep);
    }
    let dimension = link.rows();
    let matrix = DMatrix::from_row_slice(dimension, dimension, link.as_slice());
    let (vectors, triangular) = Schur::new(matrix).unpack();
    let mut phases = DMatrix::<Complex64>::zeros(dimension, dimension);
    for index in 0..dimension {
        phases[(index, index)] =
            Complex64::new(-triangular[(index, index)].arg() / coordinate_step, 0.0);
    }
    let connection = &vectors * phases * vectors.adjoint();
    let mut entries = Vec::with_capacity(dimension * dimension);
    for row in 0..dimension {
        for column in 0..dimension {
            entries.push(connection[(row, column)]);
        }
    }
    ComplexMatrix::new(dimension, dimension, entries)
        .map_err(|_| TopologyError::EigendecompositionFailed)
}

fn validate_frames(frames: &[ComplexMatrix]) -> Result<(), TopologyError> {
    if frames.len() < 2 {
        return Err(TopologyError::InsufficientFrames);
    }
    let shape = frames[0].shape();
    if shape.0 == 0
        || shape.1 == 0
        || shape.0 > shape.1
        || frames.iter().any(|frame| frame.shape() != shape)
    {
        return Err(TopologyError::IncompatibleFrames);
    }
    Ok(())
}

fn overlap_determinant(left: &ComplexMatrix, right: &ComplexMatrix) -> Complex64 {
    overlap_matrix(left, right).determinant()
}

fn overlap_matrix(left: &ComplexMatrix, right: &ComplexMatrix) -> DMatrix<Complex64> {
    let state_count = left.rows();
    let basis_count = left.columns();
    let mut overlap = DMatrix::zeros(state_count, state_count);
    for left_state in 0..state_count {
        for right_state in 0..state_count {
            overlap[(left_state, right_state)] = (0..basis_count)
                .map(|basis| {
                    left.as_slice()[left_state * basis_count + basis].conj()
                        * right.as_slice()[right_state * basis_count + basis]
                })
                .sum();
        }
    }
    overlap
}
