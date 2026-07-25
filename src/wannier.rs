//! Rust-native sampled-frame operations for Wannier construction.
//!
//! The API is expressed in terms of orthonormal state frames, trial orbitals,
//! periodic meshes, and matrix-valued Fourier fields.  It is independent of
//! any source-package object model and can therefore support native callers as
//! well as compatibility layers.

use std::f64::consts::TAU;

use nalgebra::linalg::{Schur, SVD};
use nalgebra::DMatrix;
use rustfft::FftPlanner;

use crate::spectrum::hermitian_eigensystem;
use crate::{Complex64, ComplexMatrix};

const ORTHONORMAL_TOLERANCE: f64 = 1.0e-8;
const UNITARY_TOLERANCE: f64 = 1.0e-8;
const SINGULAR_DIAGONAL_TOLERANCE: f64 = 1.0e-12;

/// Failures in sampled-frame projection, overlap, or Fourier operations.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WannierError {
    /// The mesh has no axes, contains a zero extent, or overflows its size.
    InvalidMesh,
    /// No frames were supplied or frame shapes differ.
    InvalidFrames,
    /// A state frame is not orthonormal within the native tolerance.
    NonOrthonormalFrame,
    /// Trial orbitals do not have a compatible shape.
    InvalidTrials,
    /// Trial projection loses rank at one or more samples.
    RankDeficientProjection,
    /// Boundary twists do not define one unitary diagonal per mesh axis.
    InvalidBoundaryTwists,
    /// A mesh displacement has the wrong dimensionality.
    InvalidDisplacements,
    /// Neighbor matrices, vectors, or weights are inconsistent.
    InvalidNeighborGeometry,
    /// Interpolation points have the wrong dimensionality.
    InvalidInterpolationPoints,
    /// Localization controls are non-finite or outside their valid range.
    InvalidOptimization,
    /// A diagonal neighbor overlap vanished during localization.
    SingularNeighborOverlap,
    /// Candidate, frozen, trial, or initial subspaces are inconsistent.
    InvalidSubspace,
    /// The requested fixed-rank subspace could not be constructed.
    SubspaceConstructionFailed,
    /// A Hermitian subspace update could not be diagonalized.
    EigensystemFailed,
}

impl std::fmt::Display for WannierError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidMesh => "Wannier operations require a finite nonempty regular mesh",
            Self::InvalidFrames => "sampled frames must be nonempty and have one common shape",
            Self::NonOrthonormalFrame => "sampled state rows must form an orthonormal frame",
            Self::InvalidTrials => {
                "trial orbitals must be finite, nonempty, independent, and basis-compatible"
            }
            Self::RankDeficientProjection => {
                "trial projection is rank deficient at one or more mesh samples"
            }
            Self::InvalidBoundaryTwists => {
                "boundary twists must provide unit phases for every axis and basis state"
            }
            Self::InvalidDisplacements => {
                "neighbor displacements must have one component per mesh axis"
            }
            Self::InvalidNeighborGeometry => {
                "neighbor overlaps, vectors, and weights must have compatible dimensions"
            }
            Self::InvalidInterpolationPoints => {
                "interpolation points must have one coordinate per mesh axis"
            }
            Self::InvalidOptimization => {
                "localization controls must be finite and nonnegative, with a positive step"
            }
            Self::SingularNeighborOverlap => {
                "maximal localization encountered a vanishing diagonal neighbor overlap"
            }
            Self::InvalidSubspace => {
                "candidate, frozen, trial, and initial subspaces must be basis-compatible"
            }
            Self::SubspaceConstructionFailed => {
                "the requested fixed-rank subspace could not be constructed"
            }
            Self::EigensystemFailed => {
                "a Hermitian disentanglement update could not be diagonalized"
            }
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for WannierError {}

/// Gauge-invariant and gauge-dependent quadratic-spread diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct SpreadDecomposition {
    centers: Vec<Vec<f64>>,
    spreads: Vec<f64>,
    invariant: f64,
    diagonal: f64,
    off_diagonal: f64,
}

/// Result and convergence diagnostics for maximal-localization optimization.
#[derive(Clone, Debug, PartialEq)]
pub struct GaugeOptimization {
    frames: Vec<ComplexMatrix>,
    initial_spread: f64,
    final_spread: f64,
    gradient_norm: f64,
    iterations: usize,
    converged: bool,
}

/// Result and convergence diagnostics for fixed-rank subspace optimization.
#[derive(Clone, Debug, PartialEq)]
pub struct SubspaceOptimization {
    frames: Vec<ComplexMatrix>,
    initial_invariant_spread: f64,
    final_invariant_spread: f64,
    iterations: usize,
    converged: bool,
}

impl SubspaceOptimization {
    /// Optimized fixed-rank frames, in the same mesh order as the candidates.
    #[must_use]
    pub fn frames(&self) -> &[ComplexMatrix] {
        &self.frames
    }

    /// Initial discrete gauge-invariant spread.
    #[must_use]
    pub const fn initial_invariant_spread(&self) -> f64 {
        self.initial_invariant_spread
    }

    /// Final discrete gauge-invariant spread.
    #[must_use]
    pub const fn final_invariant_spread(&self) -> f64 {
        self.final_invariant_spread
    }

    /// Number of completed self-consistent updates.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Whether the invariant-spread change reached the requested tolerance.
    #[must_use]
    pub const fn converged(&self) -> bool {
        self.converged
    }
}

impl GaugeOptimization {
    /// Optimized orthonormal frames, in the same mesh order as the input.
    #[must_use]
    pub fn frames(&self) -> &[ComplexMatrix] {
        &self.frames
    }

    /// Consumes the report and returns the optimized frames.
    #[must_use]
    pub fn into_frames(self) -> Vec<ComplexMatrix> {
        self.frames
    }

    /// Initial gauge-dependent spread, `Omega_D + Omega_OD`.
    #[must_use]
    pub const fn initial_spread(&self) -> f64 {
        self.initial_spread
    }

    /// Final gauge-dependent spread, `Omega_D + Omega_OD`.
    #[must_use]
    pub const fn final_spread(&self) -> f64 {
        self.final_spread
    }

    /// Root-mean-square Frobenius norm of the final anti-Hermitian gradient.
    #[must_use]
    pub const fn gradient_norm(&self) -> f64 {
        self.gradient_norm
    }

    /// Number of accepted optimization steps.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Whether both the spread-change and gradient stopping criteria were met.
    #[must_use]
    pub const fn converged(&self) -> bool {
        self.converged
    }
}

impl SpreadDecomposition {
    /// Cartesian center of every Wannier state.
    #[must_use]
    pub fn centers(&self) -> &[Vec<f64>] {
        &self.centers
    }

    /// Individual quadratic spreads.
    #[must_use]
    pub fn spreads(&self) -> &[f64] {
        &self.spreads
    }

    /// Gauge-invariant contribution `Omega_I`.
    #[must_use]
    pub const fn invariant(&self) -> f64 {
        self.invariant
    }

    /// Gauge-dependent diagonal contribution `Omega_D`.
    #[must_use]
    pub const fn diagonal(&self) -> f64 {
        self.diagonal
    }

    /// Gauge-dependent off-diagonal contribution `Omega_OD`.
    #[must_use]
    pub const fn off_diagonal(&self) -> f64 {
        self.off_diagonal
    }
}

fn mesh_size(shape: &[usize]) -> Result<usize, WannierError> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(WannierError::InvalidMesh);
    }
    shape.iter().try_fold(1usize, |product, &extent| {
        product.checked_mul(extent).ok_or(WannierError::InvalidMesh)
    })
}

fn validate_sampled_matrices(
    shape: &[usize],
    matrices: &[ComplexMatrix],
) -> Result<(usize, usize), WannierError> {
    if mesh_size(shape)? != matrices.len() || matrices.is_empty() {
        return Err(WannierError::InvalidFrames);
    }
    let matrix_shape = matrices[0].shape();
    if matrix_shape.0 == 0
        || matrix_shape.1 == 0
        || matrices.iter().any(|matrix| matrix.shape() != matrix_shape)
    {
        return Err(WannierError::InvalidFrames);
    }
    Ok(matrix_shape)
}

fn rows_are_orthonormal(frame: &ComplexMatrix) -> bool {
    for left in 0..frame.rows() {
        for right in 0..frame.rows() {
            let overlap = (0..frame.columns())
                .map(|basis| {
                    frame.as_slice()[left * frame.columns() + basis].conj()
                        * frame.as_slice()[right * frame.columns() + basis]
                })
                .sum::<Complex64>();
            let expected = if left == right {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            if (overlap - expected).norm() > ORTHONORMAL_TOLERANCE {
                return false;
            }
        }
    }
    true
}

fn dmatrix(matrix: &ComplexMatrix) -> DMatrix<Complex64> {
    DMatrix::from_row_slice(matrix.rows(), matrix.columns(), matrix.as_slice())
}

fn complex_matrix(matrix: &DMatrix<Complex64>) -> ComplexMatrix {
    let entries = (0..matrix.nrows())
        .flat_map(|row| (0..matrix.ncols()).map(move |column| matrix[(row, column)]))
        .collect();
    ComplexMatrix::new(matrix.nrows(), matrix.ncols(), entries)
        .expect("nalgebra produces finite matrices with a validated shape")
}

/// Optimally aligns sampled state subspaces with localized trial orbitals.
///
/// Every frame stores orthonormal states as rows.  The returned frames contain
/// one row per trial orbital and span the supplied source subspace.  A polar
/// factor is computed independently at every sample; loss of trial rank is an
/// explicit error rather than a silently unstable gauge choice.
pub fn project_trials(
    frames: &[ComplexMatrix],
    trials: &ComplexMatrix,
    singular_tolerance: f64,
) -> Result<Vec<ComplexMatrix>, WannierError> {
    if frames.is_empty() || singular_tolerance < 0.0 || !singular_tolerance.is_finite() {
        return Err(WannierError::InvalidFrames);
    }
    let source_states = frames[0].rows();
    let basis_size = frames[0].columns();
    if source_states == 0
        || basis_size == 0
        || frames
            .iter()
            .any(|frame| frame.shape() != (source_states, basis_size))
    {
        return Err(WannierError::InvalidFrames);
    }
    if frames.iter().any(|frame| !rows_are_orthonormal(frame)) {
        return Err(WannierError::NonOrthonormalFrame);
    }
    let trial_count = trials.rows();
    if trial_count == 0 || trial_count > source_states || trials.columns() != basis_size {
        return Err(WannierError::InvalidTrials);
    }

    let trial_matrix = dmatrix(trials);
    let mut result = Vec::with_capacity(frames.len());
    for frame in frames {
        let frame_matrix = dmatrix(frame);
        let overlap = frame_matrix.map(|value| value.conj()) * trial_matrix.transpose();
        let decomposition = SVD::new(overlap.clone(), true, true);
        if decomposition
            .singular_values
            .iter()
            .any(|value| *value <= singular_tolerance)
        {
            return Err(WannierError::RankDeficientProjection);
        }
        let left = decomposition
            .u
            .ok_or(WannierError::RankDeficientProjection)?;
        let right_adjoint = decomposition
            .v_t
            .ok_or(WannierError::RankDeficientProjection)?;
        let polar = left * right_adjoint;
        let projected = polar.transpose() * frame_matrix;
        result.push(complex_matrix(&projected));
    }
    Ok(result)
}

/// Expresses one operator per sample in the corresponding state frame.
///
/// Frames store ket coefficients as rows, so the returned matrix element is
/// `<frame_n|operator|frame_m>`.
pub fn operators_in_frames(
    frames: &[ComplexMatrix],
    operators: &[ComplexMatrix],
) -> Result<Vec<ComplexMatrix>, WannierError> {
    if frames.is_empty() || frames.len() != operators.len() {
        return Err(WannierError::InvalidFrames);
    }
    let state_count = frames[0].rows();
    let basis_size = frames[0].columns();
    if state_count == 0
        || basis_size == 0
        || frames
            .iter()
            .any(|frame| frame.shape() != (state_count, basis_size))
        || operators
            .iter()
            .any(|operator| operator.shape() != (basis_size, basis_size))
    {
        return Err(WannierError::InvalidFrames);
    }
    if frames.iter().any(|frame| !rows_are_orthonormal(frame)) {
        return Err(WannierError::NonOrthonormalFrame);
    }
    Ok(frames
        .iter()
        .zip(operators)
        .map(|(frame, operator)| {
            let frame = dmatrix(frame);
            let rotated = frame.map(|value| value.conj()) * dmatrix(operator) * frame.transpose();
            complex_matrix(&rotated)
        })
        .collect())
}

fn flat_coordinates(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coordinates[axis] = index % shape[axis];
        index /= shape[axis];
    }
    coordinates
}

fn flat_index(coordinates: &[usize], shape: &[usize]) -> usize {
    coordinates
        .iter()
        .zip(shape)
        .fold(0usize, |index, (&coordinate, &extent)| {
            index * extent + coordinate
        })
}

/// Computes state-overlap matrices for arbitrary periodic mesh displacements.
///
/// `boundary_twists[axis][basis]` is the diagonal basis transformation applied
/// when a positive displacement wraps once across that mesh axis.  Negative
/// wraps apply its conjugate.  This supports embedded orbitals and general
/// multi-axis toroidal meshes without source-package conventions.
pub fn periodic_overlaps(
    mesh_shape: &[usize],
    frames: &[ComplexMatrix],
    displacements: &[Vec<isize>],
    boundary_twists: &[Vec<Complex64>],
) -> Result<Vec<Vec<ComplexMatrix>>, WannierError> {
    let (state_count, basis_size) = validate_sampled_matrices(mesh_shape, frames)?;
    if frames.iter().any(|frame| !rows_are_orthonormal(frame)) {
        return Err(WannierError::NonOrthonormalFrame);
    }
    if boundary_twists.len() != mesh_shape.len()
        || boundary_twists.iter().any(|twist| {
            twist.len() != basis_size
                || twist
                    .iter()
                    .any(|value| (value.norm() - 1.0).abs() > UNITARY_TOLERANCE)
        })
    {
        return Err(WannierError::InvalidBoundaryTwists);
    }
    if displacements
        .iter()
        .any(|displacement| displacement.len() != mesh_shape.len())
    {
        return Err(WannierError::InvalidDisplacements);
    }

    let mut all_samples = Vec::with_capacity(frames.len());
    for (sample_index, left) in frames.iter().enumerate() {
        let coordinates = flat_coordinates(sample_index, mesh_shape);
        let mut sample_overlaps = Vec::with_capacity(displacements.len());
        for displacement in displacements {
            let mut shifted_coordinates = vec![0; mesh_shape.len()];
            let mut wrap_counts = vec![0isize; mesh_shape.len()];
            for axis in 0..mesh_shape.len() {
                let extent =
                    isize::try_from(mesh_shape[axis]).map_err(|_| WannierError::InvalidMesh)?;
                let shifted = isize::try_from(coordinates[axis])
                    .map_err(|_| WannierError::InvalidMesh)?
                    + displacement[axis];
                shifted_coordinates[axis] = usize::try_from(shifted.rem_euclid(extent))
                    .map_err(|_| WannierError::InvalidMesh)?;
                wrap_counts[axis] = shifted.div_euclid(extent);
            }
            let right = &frames[flat_index(&shifted_coordinates, mesh_shape)];
            let mut basis_twists = vec![Complex64::new(1.0, 0.0); basis_size];
            for (axis_twists, &count) in boundary_twists.iter().zip(&wrap_counts) {
                if count > 0 {
                    let exponent =
                        i32::try_from(count).map_err(|_| WannierError::InvalidDisplacements)?;
                    for (total, &twist) in basis_twists.iter_mut().zip(axis_twists) {
                        *total *= twist.powi(exponent);
                    }
                } else if count < 0 {
                    let exponent =
                        i32::try_from(-count).map_err(|_| WannierError::InvalidDisplacements)?;
                    for (total, &twist) in basis_twists.iter_mut().zip(axis_twists) {
                        *total *= twist.conj().powi(exponent);
                    }
                }
            }
            let mut overlap = ComplexMatrix::zeros(state_count, state_count);
            for left_state in 0..state_count {
                for right_state in 0..state_count {
                    let left_row =
                        &left.as_slice()[left_state * basis_size..(left_state + 1) * basis_size];
                    let right_row =
                        &right.as_slice()[right_state * basis_size..(right_state + 1) * basis_size];
                    let value = left_row
                        .iter()
                        .zip(right_row)
                        .zip(&basis_twists)
                        .map(|((&left_value, &right_value), &twist)| {
                            left_value.conj() * right_value * twist
                        })
                        .sum();
                    overlap
                        .set(left_state, right_state, value)
                        .expect("validated state indices fit overlap matrix");
                }
            }
            sample_overlaps.push(overlap);
        }
        all_samples.push(sample_overlaps);
    }
    Ok(all_samples)
}

fn append_orthonormal_row(
    rows: &mut Vec<Vec<Complex64>>,
    mut candidate: Vec<Complex64>,
    tolerance: f64,
) {
    for row in rows.iter() {
        let overlap = row
            .iter()
            .zip(&candidate)
            .map(|(&left, &right)| left.conj() * right)
            .sum::<Complex64>();
        for (value, &basis) in candidate.iter_mut().zip(row) {
            *value -= overlap * basis;
        }
    }
    let norm = candidate
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    if norm > tolerance {
        for value in &mut candidate {
            *value /= norm;
        }
        rows.push(candidate);
    }
}

fn append_projected_rows(
    rows: &mut Vec<Vec<Complex64>>,
    source: &ComplexMatrix,
    projector: &DMatrix<Complex64>,
    target_states: usize,
) {
    let basis_size = source.columns();
    for row in 0..source.rows() {
        let source_row = DMatrix::from_row_slice(
            1,
            basis_size,
            &source.as_slice()[row * basis_size..(row + 1) * basis_size],
        );
        let projected = source_row * projector;
        append_orthonormal_row(
            rows,
            projected.row(0).iter().copied().collect(),
            SINGULAR_DIAGONAL_TOLERANCE,
        );
        if rows.len() == target_states {
            break;
        }
    }
}

fn initial_subspace_frames(
    candidates: &[ComplexMatrix],
    frozen_counts: &[usize],
    target_states: usize,
    initial_frames: Option<&[ComplexMatrix]>,
    trials: Option<&ComplexMatrix>,
) -> Result<Vec<ComplexMatrix>, WannierError> {
    let basis_size = candidates[0].columns();
    let mut result = Vec::with_capacity(candidates.len());
    for (sample_index, candidate) in candidates.iter().enumerate() {
        let candidate_matrix = dmatrix(candidate);
        let projector = candidate_matrix.adjoint() * &candidate_matrix;
        let mut rows = Vec::<Vec<Complex64>>::with_capacity(target_states);
        for frozen in 0..frozen_counts[sample_index] {
            rows.push(
                candidate.as_slice()[frozen * basis_size..(frozen + 1) * basis_size].to_vec(),
            );
        }
        if let Some(initial) = initial_frames {
            append_projected_rows(&mut rows, &initial[sample_index], &projector, target_states);
        }
        if rows.len() < target_states {
            if let Some(trials) = trials {
                append_projected_rows(&mut rows, trials, &projector, target_states);
            }
        }
        if rows.len() < target_states {
            append_projected_rows(&mut rows, candidate, &projector, target_states);
        }
        if rows.len() != target_states {
            return Err(WannierError::SubspaceConstructionFailed);
        }
        result.push(
            ComplexMatrix::new(
                target_states,
                basis_size,
                rows.into_iter().flatten().collect(),
            )
            .map_err(|_| WannierError::SubspaceConstructionFailed)?,
        );
    }
    Ok(result)
}

fn projectors(frames: &[ComplexMatrix]) -> Vec<DMatrix<Complex64>> {
    frames
        .iter()
        .map(|frame| {
            let frame = dmatrix(frame);
            frame.adjoint() * frame
        })
        .collect()
}

fn shifted_projector(
    projectors: &[DMatrix<Complex64>],
    sample_index: usize,
    mesh_shape: &[usize],
    displacement: &[isize],
    boundary_twists: &[Vec<Complex64>],
) -> Result<DMatrix<Complex64>, WannierError> {
    let coordinates = flat_coordinates(sample_index, mesh_shape);
    let basis_size = projectors[0].nrows();
    let mut shifted_coordinates = vec![0; mesh_shape.len()];
    let mut total_twist = vec![Complex64::new(1.0, 0.0); basis_size];
    for axis in 0..mesh_shape.len() {
        let extent = isize::try_from(mesh_shape[axis]).map_err(|_| WannierError::InvalidMesh)?;
        let shifted = isize::try_from(coordinates[axis]).map_err(|_| WannierError::InvalidMesh)?
            + displacement[axis];
        shifted_coordinates[axis] =
            usize::try_from(shifted.rem_euclid(extent)).map_err(|_| WannierError::InvalidMesh)?;
        let wrap_count = shifted.div_euclid(extent);
        let exponent =
            i32::try_from(wrap_count.unsigned_abs()).map_err(|_| WannierError::InvalidMesh)?;
        if wrap_count > 0 {
            for (total, &twist) in total_twist.iter_mut().zip(&boundary_twists[axis]) {
                *total *= twist.powi(exponent);
            }
        } else if wrap_count < 0 {
            for (total, &twist) in total_twist.iter_mut().zip(&boundary_twists[axis]) {
                *total *= twist.conj().powi(exponent);
            }
        }
    }
    let source = &projectors[flat_index(&shifted_coordinates, mesh_shape)];
    let mut transformed = source.clone();
    for row in 0..basis_size {
        for column in 0..basis_size {
            transformed[(row, column)] *= total_twist[row].conj() * total_twist[column];
        }
    }
    Ok(transformed)
}

fn invariant_spread_from_projectors(
    projectors: &[DMatrix<Complex64>],
    target_states: usize,
    mesh_shape: &[usize],
    displacements: &[Vec<isize>],
    boundary_twists: &[Vec<Complex64>],
    neighbor_weights: &[f64],
) -> Result<f64, WannierError> {
    let mut spread = 0.0;
    for (sample_index, projector) in projectors.iter().enumerate() {
        for (neighbor, displacement) in displacements.iter().enumerate() {
            let shifted = shifted_projector(
                projectors,
                sample_index,
                mesh_shape,
                displacement,
                boundary_twists,
            )?;
            spread += neighbor_weights[neighbor]
                * (target_states as f64 - (projector * shifted).trace().re);
        }
    }
    Ok(spread / projectors.len() as f64)
}

/// Selects a smooth fixed-rank subspace from larger sampled candidate spaces.
///
/// Candidate rows are orthonormal states.  The first `frozen_counts[k]` rows
/// are preserved exactly at each mesh sample; remaining rows span the outer
/// variational window.  The self-consistent update selects the leading
/// eigenvectors of the neighbor-projector average, which minimizes the
/// discrete gauge-invariant spread without referring to energies or a source
/// package's window representation.
#[allow(clippy::too_many_arguments)]
pub fn disentangle_subspace(
    mesh_shape: &[usize],
    candidates: &[ComplexMatrix],
    frozen_counts: &[usize],
    target_states: usize,
    initial_frames: Option<&[ComplexMatrix]>,
    trials: Option<&ComplexMatrix>,
    displacements: &[Vec<isize>],
    boundary_twists: &[Vec<Complex64>],
    neighbor_weights: &[f64],
    max_iterations: usize,
    tolerance: f64,
    mixing: f64,
) -> Result<SubspaceOptimization, WannierError> {
    let sample_count = mesh_size(mesh_shape)?;
    if candidates.len() != sample_count
        || candidates.is_empty()
        || frozen_counts.len() != sample_count
        || target_states == 0
        || !tolerance.is_finite()
        || tolerance < 0.0
        || !mixing.is_finite()
        || !(0.0..=1.0).contains(&mixing)
        || displacements.is_empty()
        || displacements.len() != neighbor_weights.len()
        || displacements
            .iter()
            .any(|displacement| displacement.len() != mesh_shape.len())
        || neighbor_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(WannierError::InvalidSubspace);
    }
    let basis_size = candidates[0].columns();
    if basis_size == 0
        || candidates.iter().enumerate().any(|(sample, candidate)| {
            candidate.columns() != basis_size
                || candidate.rows() < target_states
                || frozen_counts[sample] > target_states
                || !rows_are_orthonormal(candidate)
        })
        || boundary_twists.len() != mesh_shape.len()
        || boundary_twists.iter().any(|twist| {
            twist.len() != basis_size
                || twist
                    .iter()
                    .any(|value| (value.norm() - 1.0).abs() > UNITARY_TOLERANCE)
        })
        || initial_frames.is_some_and(|frames| {
            frames.len() != sample_count
                || frames.iter().any(|frame| {
                    frame.shape() != (target_states, basis_size) || !rows_are_orthonormal(frame)
                })
        })
        || trials.is_some_and(|trial| trial.rows() == 0 || trial.columns() != basis_size)
    {
        return Err(WannierError::InvalidSubspace);
    }

    let mut frames = initial_subspace_frames(
        candidates,
        frozen_counts,
        target_states,
        initial_frames,
        trials,
    )?;
    let mut current_projectors = projectors(&frames);
    let initial_spread = invariant_spread_from_projectors(
        &current_projectors,
        target_states,
        mesh_shape,
        displacements,
        boundary_twists,
        neighbor_weights,
    )?;
    let mut current_spread = initial_spread;
    let mut iterations = 0;
    let mut converged = false;

    for _ in 0..max_iterations {
        let mut next_frames = Vec::with_capacity(sample_count);
        for (sample_index, candidate) in candidates.iter().enumerate() {
            let mut averaged = DMatrix::<Complex64>::zeros(basis_size, basis_size);
            for (neighbor, displacement) in displacements.iter().enumerate() {
                averaged += shifted_projector(
                    &current_projectors,
                    sample_index,
                    mesh_shape,
                    displacement,
                    boundary_twists,
                )? * Complex64::new(neighbor_weights[neighbor], 0.0);
            }

            let frozen_count = frozen_counts[sample_index];
            let remaining = target_states - frozen_count;
            let candidate_matrix = dmatrix(candidate);
            let variational = candidate_matrix
                .rows(frozen_count, candidate.rows() - frozen_count)
                .into_owned();
            let mut selected_rows = (0..frozen_count)
                .map(|row| {
                    candidate_matrix
                        .row(row)
                        .iter()
                        .copied()
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            if remaining > 0 {
                if variational.nrows() < remaining {
                    return Err(WannierError::SubspaceConstructionFailed);
                }
                let projected =
                    variational.map(|value| value.conj()) * averaged * variational.transpose();
                let hermitian = (&projected + projected.adjoint()) * Complex64::new(0.5, 0.0);
                let eigensystem = hermitian_eigensystem(&complex_matrix(&hermitian), 1.0e-8)
                    .map_err(|_| WannierError::EigensystemFailed)?;
                let eigenvectors = dmatrix(eigensystem.eigenvectors());
                for column in (eigenvectors.ncols() - remaining)..eigenvectors.ncols() {
                    let state = eigenvectors.column(column).transpose() * &variational;
                    selected_rows.push(state.row(0).iter().copied().collect());
                }
            }
            next_frames.push(
                ComplexMatrix::new(
                    target_states,
                    basis_size,
                    selected_rows.into_iter().flatten().collect(),
                )
                .map_err(|_| WannierError::SubspaceConstructionFailed)?,
            );
        }

        let next_projectors = projectors(&next_frames);
        for (current, next) in current_projectors.iter_mut().zip(&next_projectors) {
            *current = next * Complex64::new(mixing, 0.0)
                + current.clone() * Complex64::new(1.0 - mixing, 0.0);
        }
        let next_spread = invariant_spread_from_projectors(
            &current_projectors,
            target_states,
            mesh_shape,
            displacements,
            boundary_twists,
            neighbor_weights,
        )?;
        let change = (next_spread - current_spread).abs();
        frames = next_frames;
        current_spread = next_spread;
        iterations += 1;
        if change <= tolerance {
            converged = true;
            break;
        }
    }

    Ok(SubspaceOptimization {
        frames,
        initial_invariant_spread: initial_spread,
        final_invariant_spread: current_spread,
        iterations,
        converged,
    })
}

/// Evaluates the Marzari-Vanderbilt discrete quadratic-spread decomposition.
pub fn spread_decomposition(
    overlaps: &[Vec<ComplexMatrix>],
    neighbor_vectors: &[Vec<f64>],
    neighbor_weights: &[f64],
) -> Result<SpreadDecomposition, WannierError> {
    if overlaps.is_empty()
        || neighbor_vectors.is_empty()
        || neighbor_vectors.len() != neighbor_weights.len()
        || overlaps
            .iter()
            .any(|sample| sample.len() != neighbor_vectors.len())
        || neighbor_vectors
            .iter()
            .any(|vector| vector.len() != neighbor_vectors[0].len())
        || neighbor_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(WannierError::InvalidNeighborGeometry);
    }
    let state_count = overlaps[0][0].rows();
    if state_count == 0
        || overlaps
            .iter()
            .flatten()
            .any(|matrix| matrix.shape() != (state_count, state_count))
    {
        return Err(WannierError::InvalidNeighborGeometry);
    }
    let dimension = neighbor_vectors[0].len();
    let normalization = overlaps.len() as f64;
    let mut centers = vec![vec![0.0; dimension]; state_count];
    let mut radius_squared = vec![0.0; state_count];
    let mut invariant = 0.0;
    let mut off_diagonal = 0.0;

    for sample in overlaps {
        for (neighbor, matrix) in sample.iter().enumerate() {
            let weight = neighbor_weights[neighbor];
            let mut diagonal_norm_squared = 0.0;
            for state in 0..state_count {
                let diagonal = matrix.as_slice()[state * state_count + state];
                let phase = diagonal.arg();
                let norm_squared = diagonal.norm_sqr();
                diagonal_norm_squared += norm_squared;
                radius_squared[state] +=
                    weight * (1.0 - norm_squared + phase * phase) / normalization;
                for (component, value) in neighbor_vectors[neighbor].iter().enumerate() {
                    centers[state][component] -= weight * phase * value / normalization;
                }
            }
            let matrix_norm_squared: f64 =
                matrix.as_slice().iter().map(|value| value.norm_sqr()).sum();
            invariant += weight * (state_count as f64 - matrix_norm_squared) / normalization;
            off_diagonal += weight * (matrix_norm_squared - diagonal_norm_squared) / normalization;
        }
    }

    let spreads = radius_squared
        .iter()
        .zip(&centers)
        .map(|(radius, center)| {
            radius
                - center
                    .iter()
                    .map(|component| component * component)
                    .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let mut diagonal = 0.0;
    for sample in overlaps {
        for (neighbor, matrix) in sample.iter().enumerate() {
            let weight = neighbor_weights[neighbor];
            for (state, center) in centers.iter().enumerate() {
                let phase = matrix.as_slice()[state * state_count + state].arg();
                let projection = neighbor_vectors[neighbor]
                    .iter()
                    .zip(center)
                    .map(|(vector, coordinate)| vector * coordinate)
                    .sum::<f64>();
                diagonal += weight * (-phase - projection).powi(2) / normalization;
            }
        }
    }
    Ok(SpreadDecomposition {
        centers,
        spreads,
        invariant,
        diagonal,
        off_diagonal,
    })
}

fn shifted_sample_index(
    sample_index: usize,
    mesh_shape: &[usize],
    displacement: &[isize],
) -> Result<usize, WannierError> {
    let coordinates = flat_coordinates(sample_index, mesh_shape);
    let shifted = coordinates
        .iter()
        .zip(mesh_shape)
        .zip(displacement)
        .map(|((&coordinate, &extent), &offset)| {
            let extent = isize::try_from(extent).map_err(|_| WannierError::InvalidMesh)?;
            let coordinate = isize::try_from(coordinate).map_err(|_| WannierError::InvalidMesh)?;
            usize::try_from((coordinate + offset).rem_euclid(extent))
                .map_err(|_| WannierError::InvalidMesh)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(flat_index(&shifted, mesh_shape))
}

fn gauge_spread(
    overlaps: &[Vec<ComplexMatrix>],
    neighbor_vectors: &[Vec<f64>],
    neighbor_weights: &[f64],
) -> Result<f64, WannierError> {
    let decomposition = spread_decomposition(overlaps, neighbor_vectors, neighbor_weights)?;
    Ok(decomposition.diagonal() + decomposition.off_diagonal())
}

fn localization_gradients(
    overlaps: &[Vec<ComplexMatrix>],
    neighbor_vectors: &[Vec<f64>],
    neighbor_weights: &[f64],
) -> Result<Vec<DMatrix<Complex64>>, WannierError> {
    let sample_count = overlaps.len();
    let state_count = overlaps[0][0].rows();
    let dimension = neighbor_vectors[0].len();
    let mut centers = vec![vec![0.0; dimension]; state_count];
    for sample in overlaps {
        for (neighbor, matrix) in sample.iter().enumerate() {
            for (state, center) in centers.iter_mut().enumerate() {
                let phase = matrix.as_slice()[state * state_count + state].arg();
                for (component, value) in neighbor_vectors[neighbor].iter().enumerate() {
                    center[component] -=
                        neighbor_weights[neighbor] * phase * value / sample_count as f64;
                }
            }
        }
    }

    let mut gradients = vec![DMatrix::<Complex64>::zeros(state_count, state_count); sample_count];
    for (sample_index, sample) in overlaps.iter().enumerate() {
        for (neighbor, matrix) in sample.iter().enumerate() {
            let matrix = dmatrix(matrix);
            let diagonal = (0..state_count)
                .map(|state| matrix[(state, state)])
                .collect::<Vec<_>>();
            if diagonal
                .iter()
                .any(|value| value.norm() <= SINGULAR_DIAGONAL_TOLERANCE)
            {
                return Err(WannierError::SingularNeighborOverlap);
            }
            let phases = diagonal.iter().map(|value| value.arg()).collect::<Vec<_>>();
            let q = centers
                .iter()
                .zip(&phases)
                .map(|(center, &phase)| {
                    phase
                        + neighbor_vectors[neighbor]
                            .iter()
                            .zip(center)
                            .map(|(vector, coordinate)| vector * coordinate)
                            .sum::<f64>()
                })
                .collect::<Vec<_>>();
            let weight = neighbor_weights[neighbor];
            for row in 0..state_count {
                for column in 0..state_count {
                    let r = matrix[(row, column)] * diagonal[column].conj();
                    let r_adjoint = matrix[(column, row)].conj() * diagonal[row];
                    let antihermitian_r = (r - r_adjoint) * 0.5;

                    let t = matrix[(row, column)] / diagonal[column] * q[column];
                    let t_adjoint = (matrix[(column, row)] / diagonal[row] * q[row]).conj();
                    let symmetric_t = (t + t_adjoint) / Complex64::new(0.0, 2.0);
                    gradients[sample_index][(row, column)] +=
                        (antihermitian_r - symmetric_t) * (4.0 * weight);
                }
            }
        }
    }
    Ok(gradients)
}

fn gradient_norm(gradients: &[DMatrix<Complex64>]) -> f64 {
    (gradients
        .iter()
        .flat_map(|gradient| gradient.iter())
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        / gradients.len() as f64)
        .sqrt()
}

fn antihermitian_exponential(generator: &DMatrix<Complex64>, step: f64) -> DMatrix<Complex64> {
    let antihermitian = (generator - generator.adjoint()) * Complex64::new(0.5 * step, 0.0);
    let (vectors, triangular) = Schur::new(antihermitian).unpack();
    let mut exponential = DMatrix::<Complex64>::zeros(generator.nrows(), generator.ncols());
    for index in 0..generator.nrows() {
        exponential[(index, index)] = triangular[(index, index)].exp();
    }
    &vectors * exponential * vectors.adjoint()
}

fn rotated_overlaps(
    original: &[Vec<ComplexMatrix>],
    rotations: &[DMatrix<Complex64>],
    mesh_shape: &[usize],
    displacements: &[Vec<isize>],
) -> Result<Vec<Vec<ComplexMatrix>>, WannierError> {
    let mut result = Vec::with_capacity(original.len());
    for (sample_index, sample) in original.iter().enumerate() {
        let mut neighbors = Vec::with_capacity(sample.len());
        for (neighbor, matrix) in sample.iter().enumerate() {
            let shifted = shifted_sample_index(sample_index, mesh_shape, &displacements[neighbor])?;
            let transformed =
                rotations[sample_index].adjoint() * dmatrix(matrix) * &rotations[shifted];
            neighbors.push(complex_matrix(&transformed));
        }
        result.push(neighbors);
    }
    Ok(result)
}

/// Minimizes the gauge-dependent Marzari-Vanderbilt spread on a periodic mesh.
///
/// The optimizer computes the anti-Hermitian spread gradient from periodic
/// neighbor overlaps, applies unitary exponential updates, and backtracks any
/// step that would increase `Omega_D + Omega_OD`.  It acts only within the
/// supplied subspace, so the gauge-invariant spread and projectors are
/// preserved.  Mesh geometry and boundary twists are explicit inputs rather
/// than source-package state.
#[allow(clippy::too_many_arguments)]
pub fn maximize_localization(
    mesh_shape: &[usize],
    frames: &[ComplexMatrix],
    displacements: &[Vec<isize>],
    boundary_twists: &[Vec<Complex64>],
    neighbor_vectors: &[Vec<f64>],
    neighbor_weights: &[f64],
    step_scale: f64,
    max_iterations: usize,
    spread_tolerance: f64,
    gradient_tolerance: f64,
) -> Result<GaugeOptimization, WannierError> {
    validate_sampled_matrices(mesh_shape, frames)?;
    if !step_scale.is_finite()
        || step_scale <= 0.0
        || !spread_tolerance.is_finite()
        || spread_tolerance < 0.0
        || !gradient_tolerance.is_finite()
        || gradient_tolerance < 0.0
    {
        return Err(WannierError::InvalidOptimization);
    }
    if displacements.is_empty()
        || displacements.len() != neighbor_vectors.len()
        || displacements.len() != neighbor_weights.len()
    {
        return Err(WannierError::InvalidNeighborGeometry);
    }
    let original = periodic_overlaps(mesh_shape, frames, displacements, boundary_twists)?;
    let state_count = frames[0].rows();
    let sample_count = frames.len();
    let mut rotations =
        vec![DMatrix::<Complex64>::identity(state_count, state_count); sample_count];
    let mut overlaps = original.clone();
    let initial_spread = gauge_spread(&overlaps, neighbor_vectors, neighbor_weights)?;
    let mut current_spread = initial_spread;
    let mut gradients = localization_gradients(&overlaps, neighbor_vectors, neighbor_weights)?;
    let mut final_gradient_norm = gradient_norm(&gradients);
    let weight_sum = neighbor_weights.iter().sum::<f64>();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(WannierError::InvalidNeighborGeometry);
    }

    let mut iterations = 0;
    let mut converged = final_gradient_norm <= gradient_tolerance;
    for _ in 0..max_iterations {
        if converged {
            break;
        }
        let mut trial_step = step_scale / (4.0 * weight_sum);
        let mut accepted = None;
        for _ in 0..16 {
            let candidate_rotations = rotations
                .iter()
                .zip(&gradients)
                .map(|(rotation, gradient)| {
                    rotation * antihermitian_exponential(gradient, trial_step)
                })
                .collect::<Vec<_>>();
            let candidate_overlaps =
                rotated_overlaps(&original, &candidate_rotations, mesh_shape, displacements)?;
            let candidate_spread =
                gauge_spread(&candidate_overlaps, neighbor_vectors, neighbor_weights)?;
            if candidate_spread <= current_spread + 1.0e-13 {
                accepted = Some((candidate_rotations, candidate_overlaps, candidate_spread));
                break;
            }
            trial_step *= 0.5;
        }
        let Some((candidate_rotations, candidate_overlaps, candidate_spread)) = accepted else {
            break;
        };
        let spread_change = (candidate_spread - current_spread).abs();
        rotations = candidate_rotations;
        overlaps = candidate_overlaps;
        current_spread = candidate_spread;
        iterations += 1;
        gradients = localization_gradients(&overlaps, neighbor_vectors, neighbor_weights)?;
        final_gradient_norm = gradient_norm(&gradients);
        converged = spread_change <= spread_tolerance && final_gradient_norm <= gradient_tolerance;
    }

    let optimized_frames = frames
        .iter()
        .zip(&rotations)
        .map(|(frame, rotation)| complex_matrix(&(rotation.transpose() * dmatrix(frame))))
        .collect();
    Ok(GaugeOptimization {
        frames: optimized_frames,
        initial_spread,
        final_spread: current_spread,
        gradient_norm: final_gradient_norm,
        iterations,
        converged,
    })
}

fn transform_axis(shape: &[usize], matrices: &mut [ComplexMatrix], axis: usize, inverse: bool) {
    let extent = shape[axis];
    let stride = shape[axis + 1..].iter().product::<usize>();
    let block = extent * stride;
    let outer_count = matrices.len() / block;
    let rows = matrices[0].rows();
    let columns = matrices[0].columns();
    let mut planner = FftPlanner::<f64>::new();
    let transform = if inverse {
        planner.plan_fft_inverse(extent)
    } else {
        planner.plan_fft_forward(extent)
    };
    let scale = if inverse { 1.0 / extent as f64 } else { 1.0 };
    let mut line = vec![Complex64::new(0.0, 0.0); extent];
    for outer in 0..outer_count {
        for offset in 0..stride {
            for row in 0..rows {
                for column in 0..columns {
                    for point in 0..extent {
                        line[point] = matrices[outer * block + point * stride + offset].as_slice()
                            [row * columns + column];
                    }
                    transform.process(&mut line);
                    for point in 0..extent {
                        matrices[outer * block + point * stride + offset]
                            .set(row, column, line[point] * scale)
                            .expect("validated component index fits transformed matrix");
                    }
                }
            }
        }
    }
}

/// Applies a normalized multidimensional inverse Fourier transform to frames.
pub fn inverse_bloch_transform(
    mesh_shape: &[usize],
    frames: &[ComplexMatrix],
) -> Result<Vec<ComplexMatrix>, WannierError> {
    validate_sampled_matrices(mesh_shape, frames)?;
    let mut result = frames.to_vec();
    for axis in 0..mesh_shape.len() {
        transform_axis(mesh_shape, &mut result, axis, true);
    }
    Ok(result)
}

/// Interpolates a periodic matrix field from a complete uniform mesh.
///
/// The routine forms normalized real-space Fourier coefficients and evaluates
/// their finite Fourier series at arbitrary reduced-coordinate points.
pub fn interpolate_periodic_matrices(
    mesh_shape: &[usize],
    samples: &[ComplexMatrix],
    points: &[Vec<f64>],
) -> Result<Vec<ComplexMatrix>, WannierError> {
    let (rows, columns) = validate_sampled_matrices(mesh_shape, samples)?;
    if points.iter().any(|point| {
        point.len() != mesh_shape.len() || point.iter().any(|coordinate| !coordinate.is_finite())
    }) {
        return Err(WannierError::InvalidInterpolationPoints);
    }
    let mut coefficients = samples.to_vec();
    for axis in 0..mesh_shape.len() {
        transform_axis(mesh_shape, &mut coefficients, axis, false);
    }
    let normalization = coefficients.len() as f64;
    for matrix in &mut coefficients {
        for row in 0..rows {
            for column in 0..columns {
                let value = matrix.as_slice()[row * columns + column] / normalization;
                matrix
                    .set(row, column, value)
                    .expect("validated component index fits coefficient matrix");
            }
        }
    }

    let frequencies = (0..coefficients.len())
        .map(|index| {
            flat_coordinates(index, mesh_shape)
                .into_iter()
                .zip(mesh_shape)
                .map(|(coordinate, &extent)| {
                    let cutoff = (extent - 1) / 2;
                    if coordinate <= cutoff {
                        coordinate as isize
                    } else {
                        coordinate as isize - extent as isize
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut result = Vec::with_capacity(points.len());
    for point in points {
        let mut interpolated = ComplexMatrix::zeros(rows, columns);
        for (frequency, coefficient) in frequencies.iter().zip(&coefficients) {
            let angle = TAU
                * frequency
                    .iter()
                    .zip(point)
                    .map(|(&mode, &coordinate)| mode as f64 * coordinate)
                    .sum::<f64>();
            let phase = Complex64::from_polar(1.0, angle);
            for row in 0..rows {
                for column in 0..columns {
                    interpolated
                        .add_entry(
                            row,
                            column,
                            coefficient.as_slice()[row * columns + column] * phase,
                        )
                        .expect("validated component index fits interpolation matrix");
                }
            }
        }
        result.push(interpolated);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(rows: usize, columns: usize, values: Vec<Complex64>) -> ComplexMatrix {
        ComplexMatrix::new(rows, columns, values).unwrap()
    }

    #[test]
    fn full_subspace_projection_recovers_the_trial_orbital() {
        let angle = 0.37;
        let phase = Complex64::from_polar(1.0, angle);
        let frames = vec![matrix(
            2,
            2,
            vec![
                phase / 2.0f64.sqrt(),
                Complex64::new(1.0 / 2.0f64.sqrt(), 0.0),
                -phase / 2.0f64.sqrt(),
                Complex64::new(1.0 / 2.0f64.sqrt(), 0.0),
            ],
        )];
        let trials = matrix(
            1,
            2,
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        );
        let projected = project_trials(&frames, &trials, 1.0e-12).unwrap();
        assert!((projected[0].as_slice()[0] - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
        assert!(projected[0].as_slice()[1].norm() < 1.0e-12);
    }

    #[test]
    fn atomic_frame_has_its_embedding_center_and_zero_spread() {
        let sample_count = 9;
        let embedding = 0.23;
        let frames = (0..sample_count)
            .map(|sample| {
                matrix(
                    1,
                    1,
                    vec![Complex64::from_polar(
                        1.0,
                        -TAU * embedding * sample as f64 / sample_count as f64,
                    )],
                )
            })
            .collect::<Vec<_>>();
        let overlaps = periodic_overlaps(
            &[sample_count],
            &frames,
            &[vec![1], vec![-1]],
            &[vec![Complex64::from_polar(1.0, -TAU * embedding)]],
        )
        .unwrap();
        let reciprocal_step = TAU / sample_count as f64;
        let weight = 1.0 / (2.0 * reciprocal_step * reciprocal_step);
        let spread = spread_decomposition(
            &overlaps,
            &[vec![reciprocal_step], vec![-reciprocal_step]],
            &[weight, weight],
        )
        .unwrap();
        assert!((spread.centers()[0][0] - embedding).abs() < 1.0e-12);
        assert!(spread.spreads()[0].abs() < 1.0e-12);
        assert!(spread.invariant().abs() < 1.0e-12);
        assert!(spread.diagonal().abs() < 1.0e-12);
        assert!(spread.off_diagonal().abs() < 1.0e-12);
    }

    #[test]
    fn constant_bloch_frame_transforms_only_to_the_home_cell() {
        let frames = vec![matrix(1, 1, vec![Complex64::new(1.0, 0.0)]); 8];
        let transformed = inverse_bloch_transform(&[8], &frames).unwrap();
        assert!((transformed[0].as_slice()[0] - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
        assert!(transformed[1..]
            .iter()
            .all(|frame| frame.as_slice()[0].norm() < 1.0e-12));
    }

    #[test]
    fn operator_rotation_preserves_a_hermitian_spectrum() {
        let inverse_sqrt_two = 1.0 / 2.0f64.sqrt();
        let frame = matrix(
            2,
            2,
            vec![
                Complex64::new(inverse_sqrt_two, 0.0),
                Complex64::new(inverse_sqrt_two, 0.0),
                Complex64::new(inverse_sqrt_two, 0.0),
                Complex64::new(-inverse_sqrt_two, 0.0),
            ],
        );
        let operator = matrix(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        );
        let rotated = operators_in_frames(&[frame], &[operator]).unwrap();
        assert!(rotated[0].is_hermitian(1.0e-12).unwrap());
        assert!(rotated[0].as_slice()[0].norm() < 1.0e-12);
        assert!((rotated[0].as_slice()[1] - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
        assert!((rotated[0].as_slice()[2] - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
        assert!(rotated[0].as_slice()[3].norm() < 1.0e-12);
    }

    #[test]
    fn finite_fourier_field_interpolates_an_unseen_cosine_point() {
        let sample_count = 8;
        let samples = (0..sample_count)
            .map(|sample| {
                let momentum = sample as f64 / sample_count as f64;
                matrix(
                    1,
                    1,
                    vec![Complex64::new(-2.0 * (TAU * momentum).cos(), 0.0)],
                )
            })
            .collect::<Vec<_>>();
        let point = 0.137;
        let interpolated =
            interpolate_periodic_matrices(&[sample_count], &samples, &[vec![point]]).unwrap();
        assert!((interpolated[0].as_slice()[0].re + 2.0 * (TAU * point).cos()).abs() < 1.0e-12);
        assert!(interpolated[0].as_slice()[0].im.abs() < 1.0e-12);
    }

    #[test]
    fn maximal_localization_removes_a_periodic_single_band_gauge() {
        let sample_count = 12;
        let frames = (0..sample_count)
            .map(|sample| {
                let momentum = TAU * sample as f64 / sample_count as f64;
                matrix(
                    1,
                    1,
                    vec![Complex64::from_polar(1.0, 0.63 * momentum.sin())],
                )
            })
            .collect::<Vec<_>>();
        let reciprocal_step = TAU / sample_count as f64;
        let neighbor_vectors = vec![vec![reciprocal_step], vec![-reciprocal_step]];
        let neighbor_weights = vec![
            1.0 / (2.0 * reciprocal_step.powi(2)),
            1.0 / (2.0 * reciprocal_step.powi(2)),
        ];
        let report = maximize_localization(
            &[sample_count],
            &frames,
            &[vec![1], vec![-1]],
            &[vec![Complex64::new(1.0, 0.0)]],
            &neighbor_vectors,
            &neighbor_weights,
            0.5,
            200,
            1.0e-12,
            1.0e-10,
        )
        .unwrap();

        assert!(report.iterations() > 0);
        assert!(report.final_spread() < report.initial_spread() * 1.0e-6);
        assert!(report
            .frames()
            .iter()
            .all(|frame| (frame.as_slice()[0].norm() - 1.0).abs() < 1.0e-12));
    }

    #[test]
    fn disentanglement_smooths_a_fixed_rank_subspace() {
        let sample_count = 14;
        let candidates = vec![ComplexMatrix::identity(2); sample_count];
        let initial = (0..sample_count)
            .map(|sample| {
                let momentum = TAU * sample as f64 / sample_count as f64;
                let angle = 0.71 * momentum.sin();
                matrix(
                    1,
                    2,
                    vec![
                        Complex64::new(angle.cos(), 0.0),
                        Complex64::new(angle.sin(), 0.0),
                    ],
                )
            })
            .collect::<Vec<_>>();
        let report = disentangle_subspace(
            &[sample_count],
            &candidates,
            &vec![0; sample_count],
            1,
            Some(&initial),
            None,
            &[vec![1], vec![-1]],
            &[vec![Complex64::new(1.0, 0.0); 2]],
            &[1.0, 1.0],
            200,
            1.0e-12,
            0.7,
        )
        .unwrap();

        assert!(report.iterations() > 0);
        assert!(report.final_invariant_spread() < report.initial_invariant_spread() * 1.0e-6);
        assert!(report.frames().iter().all(rows_are_orthonormal));
    }
}
