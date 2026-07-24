//! Gauge-invariant discrete topology for sampled state subspaces.

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
