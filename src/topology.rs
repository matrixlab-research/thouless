//! Gauge-invariant discrete topology for sampled state subspaces.

use nalgebra::linalg::Schur;
use nalgebra::DMatrix;

use crate::model::{ModelSolveError, TightBindingModel};
use crate::spectrum::hermitian_eigensystem;
use crate::{Complex64, ComplexMatrix, TopologyError};

const SINGULAR_OVERLAP_TOLERANCE: f64 = 1.0e-14;

/// Second-Chern density integrated over three coordinates of a four-torus.
///
/// `slice_densities[lambda]` contains the Brillouin-zone integral at one
/// sample of the fourth coordinate, including the conventional
/// `1 / (4 π²)` normalization but excluding the fourth-coordinate step.
#[derive(Clone, Debug, PartialEq)]
pub struct SecondChernResult {
    slice_densities: Vec<f64>,
    value: f64,
}

impl SecondChernResult {
    /// Returns the normalized three-dimensional density at every fourth-axis
    /// sample.
    #[must_use]
    pub fn slice_densities(&self) -> &[f64] {
        &self.slice_densities
    }

    /// Returns the integrated second Chern number.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }
}

/// Errors raised while evaluating a second Chern number.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SecondChernError {
    /// Exactly four grid sizes are required.
    InvalidGridDimension {
        /// Number of supplied axes.
        actual: usize,
    },
    /// A central difference requires at least three samples.
    InsufficientSamples {
        /// Grid axis with too few samples.
        axis: usize,
        /// Number of supplied samples.
        actual: usize,
    },
    /// The number of Hamiltonians or derivative groups does not match the grid.
    InvalidHamiltonianDataCount {
        /// Number required by the grid shape.
        expected: usize,
        /// Number of Hamiltonians supplied.
        hamiltonians: usize,
        /// Number of derivative groups supplied.
        derivatives: usize,
    },
    /// Hamiltonians must be nonempty Hermitian square matrices of one common size.
    IncompatibleHamiltonian {
        /// Index of the incompatible Hamiltonian.
        index: usize,
    },
    /// Each grid point must provide four compatible Hermitian derivatives.
    IncompatibleHamiltonianDerivative {
        /// Grid point containing the incompatible derivative.
        index: usize,
        /// Coordinate direction of the incompatible derivative.
        axis: usize,
    },
    /// Occupied states must be unique valid indices and leave at least one empty state.
    InvalidOccupiedStates,
    /// The occupied and empty subspaces must be separated by a finite gap.
    ClosedOccupiedGap {
        /// Grid point where the occupied gap closes.
        index: usize,
    },
    /// Diagonalization failed for a validated Hamiltonian.
    EigensystemFailure {
        /// Grid point where diagonalization failed.
        index: usize,
    },
    /// Exactly four finite, nonzero coordinate steps are required.
    InvalidCoordinateSteps,
    /// The grid size cannot be represented.
    GridTooLarge,
}

impl std::fmt::Display for SecondChernError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidGridDimension { actual } => {
                write!(
                    formatter,
                    "second Chern grid has {actual} axes; expected four"
                )
            }
            Self::InsufficientSamples { axis, actual } => write!(
                formatter,
                "second Chern grid axis {axis} has {actual} samples; at least three are required"
            ),
            Self::InvalidHamiltonianDataCount {
                expected,
                hamiltonians,
                derivatives,
            } => write!(
                formatter,
                "second Chern grid requires {expected} Hamiltonians and derivative groups; \
                 received {hamiltonians} Hamiltonians and {derivatives} derivative groups"
            ),
            Self::IncompatibleHamiltonian { index } => write!(
                formatter,
                "Hamiltonian {index} is not a finite Hermitian square matrix of the common size"
            ),
            Self::IncompatibleHamiltonianDerivative { index, axis } => write!(
                formatter,
                "Hamiltonian derivative {axis} at grid point {index} is not a compatible \
                 Hermitian matrix"
            ),
            Self::InvalidOccupiedStates => write!(
                formatter,
                "occupied states must be unique valid indices and leave at least one empty state"
            ),
            Self::ClosedOccupiedGap { index } => write!(
                formatter,
                "occupied and empty states are degenerate at second Chern grid point {index}"
            ),
            Self::EigensystemFailure { index } => write!(
                formatter,
                "failed to diagonalize Hamiltonian at second Chern grid point {index}"
            ),
            Self::InvalidCoordinateSteps => write!(
                formatter,
                "second Chern calculation requires four finite nonzero coordinate steps"
            ),
            Self::GridTooLarge => {
                write!(formatter, "second Chern grid is too large to represent")
            }
        }
    }
}

impl std::error::Error for SecondChernError {}

/// Computes the second Chern number from Hamiltonians and their four
/// coordinate derivatives using the non-Abelian Kubo curvature.
///
/// Hamiltonians and derivative groups are ordered row-major over a
/// four-dimensional uniform grid. Each derivative group contains
/// `∂₀H, ∂₁H, ∂₂H, ∂₃H` in the physical coordinates whose spacings are
/// supplied in `coordinate_steps`. The first three spacings set the
/// Brillouin-zone volume element; the fourth completes the integral.
///
/// This formulation uses the full occupied-to-empty spectral response rather
/// than differentiating grid-sampled eigenvectors or projectors. It is
/// therefore invariant under arbitrary basis choices inside degenerate
/// occupied or empty subspaces.
pub fn second_chern_from_hamiltonian_derivatives(
    hamiltonians: &[ComplexMatrix],
    derivatives: &[[ComplexMatrix; 4]],
    grid_shape: &[usize],
    coordinate_steps: &[f64],
    fourth_axis_periodic: bool,
    occupied_states: &[usize],
) -> Result<SecondChernResult, SecondChernError> {
    if grid_shape.len() != 4 {
        return Err(SecondChernError::InvalidGridDimension {
            actual: grid_shape.len(),
        });
    }
    for (axis, &actual) in grid_shape.iter().enumerate() {
        if actual < 3 {
            return Err(SecondChernError::InsufficientSamples { axis, actual });
        }
    }
    if coordinate_steps.len() != 4
        || coordinate_steps
            .iter()
            .any(|step| !step.is_finite() || *step == 0.0)
    {
        return Err(SecondChernError::InvalidCoordinateSteps);
    }
    let grid_size = grid_shape
        .iter()
        .try_fold(1_usize, |product, size| product.checked_mul(*size))
        .ok_or(SecondChernError::GridTooLarge)?;
    if hamiltonians.len() != grid_size || derivatives.len() != grid_size {
        return Err(SecondChernError::InvalidHamiltonianDataCount {
            expected: grid_size,
            hamiltonians: hamiltonians.len(),
            derivatives: derivatives.len(),
        });
    }

    let dimension = hamiltonians.first().map_or(0, ComplexMatrix::rows);
    for (index, hamiltonian) in hamiltonians.iter().enumerate() {
        if dimension == 0
            || hamiltonian.shape() != (dimension, dimension)
            || !hamiltonian.is_hermitian(1.0e-8).unwrap_or(false)
        {
            return Err(SecondChernError::IncompatibleHamiltonian { index });
        }
        for (axis, derivative) in derivatives[index].iter().enumerate() {
            if derivative.shape() != (dimension, dimension)
                || !derivative.is_hermitian(1.0e-8).unwrap_or(false)
            {
                return Err(SecondChernError::IncompatibleHamiltonianDerivative { index, axis });
            }
        }
    }

    let mut occupied_mask = vec![false; dimension];
    for &state in occupied_states {
        if state >= dimension || occupied_mask[state] {
            return Err(SecondChernError::InvalidOccupiedStates);
        }
        occupied_mask[state] = true;
    }
    let empty_states = (0..dimension)
        .filter(|state| !occupied_mask[*state])
        .collect::<Vec<_>>();
    if occupied_states.is_empty() || empty_states.is_empty() {
        return Err(SecondChernError::InvalidOccupiedStates);
    }

    let normalization =
        coordinate_steps[..3].iter().product::<f64>() / (4.0 * std::f64::consts::PI.powi(2));
    let mut slice_densities = vec![0.0; grid_shape[3]];
    for (flat_index, hamiltonian) in hamiltonians.iter().enumerate() {
        let eigensystem = hermitian_eigensystem(hamiltonian, 1.0e-8)
            .map_err(|_| SecondChernError::EigensystemFailure { index: flat_index })?;
        let eigenvalues = eigensystem.eigenvalues();
        for &occupied in occupied_states {
            if empty_states
                .iter()
                .any(|&empty| (eigenvalues[occupied] - eigenvalues[empty]).abs() < 1.0e-12)
            {
                return Err(SecondChernError::ClosedOccupiedGap { index: flat_index });
            }
        }
        let eigenvectors = dmatrix(eigensystem.eigenvectors());
        let eigenvectors_adjoint = eigenvectors.adjoint();
        let rotated_derivatives = derivatives[flat_index]
            .iter()
            .map(|derivative| &eigenvectors_adjoint * dmatrix(derivative) * &eigenvectors)
            .collect::<Vec<_>>();
        let curvature = |first: usize, second: usize| {
            let occupied_count = occupied_states.len();
            let mut quantum_geometric =
                DMatrix::from_element(occupied_count, occupied_count, Complex64::new(0.0, 0.0));
            for (row, &left_occupied) in occupied_states.iter().enumerate() {
                for (column, &right_occupied) in occupied_states.iter().enumerate() {
                    quantum_geometric[(row, column)] = empty_states
                        .iter()
                        .map(|&empty| {
                            let left_gap = eigenvalues[left_occupied] - eigenvalues[empty];
                            let right_gap = eigenvalues[right_occupied] - eigenvalues[empty];
                            rotated_derivatives[first][(left_occupied, empty)]
                                * rotated_derivatives[second][(empty, right_occupied)]
                                / (left_gap * right_gap)
                        })
                        .sum();
                }
            }
            let mut result =
                DMatrix::from_element(occupied_count, occupied_count, Complex64::new(0.0, 0.0));
            for row in 0..occupied_count {
                for column in 0..occupied_count {
                    result[(row, column)] = Complex64::new(0.0, 1.0)
                        * (quantum_geometric[(row, column)]
                            - quantum_geometric[(column, row)].conj());
                }
            }
            result
        };
        let curvature_01 = curvature(0, 1);
        let curvature_02 = curvature(0, 2);
        let curvature_03 = curvature(0, 3);
        let curvature_12 = curvature(1, 2);
        let curvature_13 = curvature(1, 3);
        let curvature_23 = curvature(2, 3);
        let density = ((curvature_01 * curvature_23).trace()
            - (curvature_02 * curvature_13).trace()
            + (curvature_03 * curvature_12).trace())
        .re;
        let fourth_coordinate = unravel_index(flat_index, grid_shape)[3];
        slice_densities[fourth_coordinate] += normalization * density;
    }

    let fourth_integral = if fourth_axis_periodic {
        slice_densities.iter().sum::<f64>()
    } else {
        slice_densities
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if index == 0 || index + 1 == slice_densities.len() {
                    0.5 * value
                } else {
                    *value
                }
            })
            .sum::<f64>()
    };
    Ok(SecondChernResult {
        value: coordinate_steps[3] * fourth_integral,
        slice_densities,
    })
}

fn dmatrix(matrix: &ComplexMatrix) -> DMatrix<Complex64> {
    DMatrix::from_row_slice(matrix.rows(), matrix.columns(), matrix.as_slice())
}

/// Chern numbers evaluated on the spectator coordinates of a uniform grid.
///
/// The values use row-major ordering over `spectator_shape`. A two-dimensional
/// model therefore returns one value and an empty spectator shape.
#[derive(Clone, Debug, PartialEq)]
pub struct UniformGridChernNumbers {
    values: Vec<f64>,
    spectator_shape: Vec<usize>,
}

impl UniformGridChernNumbers {
    /// Returns the Chern number at every spectator-grid coordinate.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Returns the grid shape after removing the two integrated directions.
    #[must_use]
    pub fn spectator_shape(&self) -> &[usize] {
        &self.spectator_shape
    }
}

/// Errors raised while evaluating a Chern number on a uniform grid.
#[derive(Debug)]
#[non_exhaustive]
pub enum ChernNumberError {
    /// The mesh does not provide one size per periodic direction.
    InvalidGridDimension {
        /// Number of periodic directions.
        expected: usize,
        /// Number of supplied grid sizes.
        actual: usize,
    },
    /// The integration directions are equal or outside the periodic space.
    InvalidPlane {
        /// First requested direction.
        first: usize,
        /// Second requested direction.
        second: usize,
        /// Number of periodic directions.
        dimension: usize,
    },
    /// An integrated direction has fewer than two samples.
    InsufficientSamples {
        /// Direction with too few samples.
        axis: usize,
        /// Number of supplied samples.
        actual: usize,
        /// Minimum number required for this direction.
        minimum: usize,
    },
    /// At least one occupied state is required.
    EmptyOccupiedSubspace,
    /// An occupied-state index is outside the Hamiltonian basis.
    InvalidOccupiedState {
        /// Invalid state index.
        state: usize,
        /// Hamiltonian dimension.
        state_count: usize,
    },
    /// An occupied-state index occurs more than once.
    DuplicateOccupiedState {
        /// Repeated state index.
        state: usize,
    },
    /// The requested mesh is too large to represent.
    GridTooLarge,
    /// Model assembly or diagonalization failed.
    Model(ModelSolveError),
    /// A sampled state-frame overlap is singular or incompatible.
    Topology(TopologyError),
}

impl std::fmt::Display for ChernNumberError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidGridDimension { expected, actual } => write!(
                formatter,
                "uniform grid has {actual} dimensions; expected {expected}"
            ),
            Self::InvalidPlane {
                first,
                second,
                dimension,
            } => write!(
                formatter,
                "Chern plane ({first}, {second}) is invalid for periodic dimension {dimension}"
            ),
            Self::InsufficientSamples {
                axis,
                actual,
                minimum,
            } => write!(
                formatter,
                "uniform-grid axis {axis} has {actual} samples; at least {minimum} are required"
            ),
            Self::EmptyOccupiedSubspace => {
                write!(
                    formatter,
                    "Chern number requires at least one occupied state"
                )
            }
            Self::InvalidOccupiedState { state, state_count } => write!(
                formatter,
                "occupied state {state} is outside a Hamiltonian with {state_count} states"
            ),
            Self::DuplicateOccupiedState { state } => {
                write!(formatter, "occupied state {state} occurs more than once")
            }
            Self::GridTooLarge => write!(formatter, "uniform grid is too large to represent"),
            Self::Model(error) => error.fmt(formatter),
            Self::Topology(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ChernNumberError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Topology(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModelSolveError> for ChernNumberError {
    fn from(error: ModelSolveError) -> Self {
        Self::Model(error)
    }
}

impl From<TopologyError> for ChernNumberError {
    fn from(error: TopologyError) -> Self {
        Self::Topology(error)
    }
}

/// Computes first Chern numbers from occupied eigenspaces on a uniform torus.
///
/// The model is diagonalized once at every grid point. Plaquettes crossing a
/// Brillouin-zone boundary apply the orbital embedding gauge before evaluating
/// the Fukui-Hatsugai-Suzuki flux. Directions outside `plane` are retained as
/// spectator coordinates in the returned array.
pub fn chern_numbers_on_uniform_grid(
    model: &TightBindingModel,
    samples: &[usize],
    plane: [usize; 2],
    occupied_states: &[usize],
) -> Result<UniformGridChernNumbers, ChernNumberError> {
    let dimension = model.lattice().periodic_dimension();
    if samples.len() != dimension {
        return Err(ChernNumberError::InvalidGridDimension {
            expected: dimension,
            actual: samples.len(),
        });
    }
    let [first, second] = plane;
    if first == second || first >= dimension || second >= dimension {
        return Err(ChernNumberError::InvalidPlane {
            first,
            second,
            dimension,
        });
    }
    for (axis, &actual) in samples.iter().enumerate() {
        let minimum = if plane.contains(&axis) { 2 } else { 1 };
        if actual < minimum {
            return Err(ChernNumberError::InsufficientSamples {
                axis,
                actual,
                minimum,
            });
        }
    }
    if occupied_states.is_empty() {
        return Err(ChernNumberError::EmptyOccupiedSubspace);
    }
    let mut seen = std::collections::HashSet::new();
    for &state in occupied_states {
        if state >= model.state_count() {
            return Err(ChernNumberError::InvalidOccupiedState {
                state,
                state_count: model.state_count(),
            });
        }
        if !seen.insert(state) {
            return Err(ChernNumberError::DuplicateOccupiedState { state });
        }
    }

    let grid_size = checked_product(samples)?;
    let mut frames = Vec::with_capacity(grid_size);
    for flat_index in 0..grid_size {
        let index = unravel_index(flat_index, samples);
        let momentum = index
            .iter()
            .zip(samples)
            .map(|(coordinate, size)| *coordinate as f64 / *size as f64)
            .collect::<Vec<_>>();
        let eigensystem = model.eigensystem(&momentum)?;
        frames.push(occupied_frame(eigensystem.eigenvectors(), occupied_states)?);
    }

    let spectator_axes = (0..dimension)
        .filter(|axis| *axis != first && *axis != second)
        .collect::<Vec<_>>();
    let spectator_shape = spectator_axes
        .iter()
        .map(|axis| samples[*axis])
        .collect::<Vec<_>>();
    let spectator_size = checked_product(&spectator_shape)?;
    let mut values = Vec::with_capacity(spectator_size);
    for spectator_flat in 0..spectator_size {
        let spectator_index = unravel_index(spectator_flat, &spectator_shape);
        let mut base = vec![0; dimension];
        for (axis, coordinate) in spectator_axes.iter().zip(spectator_index) {
            base[*axis] = coordinate;
        }
        let mut flux = 0.0;
        for first_coordinate in 0..samples[first] {
            for second_coordinate in 0..samples[second] {
                base[first] = first_coordinate;
                base[second] = second_coordinate;
                let corners = [
                    sampled_frame(model, &frames, samples, &base, &[])?,
                    offset_frame(model, &frames, samples, &base, first)?,
                    offset_frame_pair(model, &frames, samples, &base, first, second)?,
                    offset_frame(model, &frames, samples, &base, second)?,
                ];
                flux += plaquette_flux(&corners)?;
            }
        }
        values.push(flux / std::f64::consts::TAU);
    }

    Ok(UniformGridChernNumbers {
        values,
        spectator_shape,
    })
}

fn checked_product(shape: &[usize]) -> Result<usize, ChernNumberError> {
    shape
        .iter()
        .try_fold(1_usize, |product, size| product.checked_mul(*size))
        .ok_or(ChernNumberError::GridTooLarge)
}

fn unravel_index(mut flat_index: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        index[axis] = flat_index % shape[axis];
        flat_index /= shape[axis];
    }
    index
}

fn ravel_index(index: &[usize], shape: &[usize]) -> usize {
    index
        .iter()
        .zip(shape)
        .fold(0, |flat, (coordinate, size)| flat * size + coordinate)
}

fn occupied_frame(
    eigenvectors: &ComplexMatrix,
    occupied_states: &[usize],
) -> Result<ComplexMatrix, ChernNumberError> {
    let basis_count = eigenvectors.rows();
    let mut values = Vec::with_capacity(occupied_states.len() * basis_count);
    for &state in occupied_states {
        for basis in 0..basis_count {
            values.push(
                eigenvectors
                    .get(basis, state)
                    .map_err(|_| TopologyError::IncompatibleFrames)?,
            );
        }
    }
    ComplexMatrix::new(occupied_states.len(), basis_count, values)
        .map_err(|_| TopologyError::IncompatibleFrames.into())
}

fn sampled_frame(
    model: &TightBindingModel,
    frames: &[ComplexMatrix],
    samples: &[usize],
    index: &[usize],
    crossed_axes: &[usize],
) -> Result<ComplexMatrix, ChernNumberError> {
    let frame = frames
        .get(ravel_index(index, samples))
        .ok_or(TopologyError::IncompatibleFrames)?;
    if crossed_axes.is_empty() {
        return Ok(frame.clone());
    }
    let mut basis_phases = Vec::with_capacity(model.state_count());
    for orbital in model.orbitals() {
        let phase_argument = crossed_axes
            .iter()
            .map(|axis| {
                let real_axis = model.lattice().periodic_axes()[*axis];
                orbital.reduced_position()[real_axis]
            })
            .sum::<f64>();
        let phase = Complex64::from_polar(1.0, -std::f64::consts::TAU * phase_argument);
        basis_phases.extend(std::iter::repeat(phase).take(orbital.degrees_of_freedom()));
    }
    let mut values = frame.as_slice().to_vec();
    for row in values.chunks_mut(frame.columns()) {
        for (value, phase) in row.iter_mut().zip(&basis_phases) {
            *value *= phase;
        }
    }
    ComplexMatrix::new(frame.rows(), frame.columns(), values)
        .map_err(|_| TopologyError::IncompatibleFrames.into())
}

fn offset_frame(
    model: &TightBindingModel,
    frames: &[ComplexMatrix],
    samples: &[usize],
    base: &[usize],
    axis: usize,
) -> Result<ComplexMatrix, ChernNumberError> {
    let mut index = base.to_vec();
    index[axis] += 1;
    let crossed = if index[axis] == samples[axis] {
        index[axis] = 0;
        vec![axis]
    } else {
        Vec::new()
    };
    sampled_frame(model, frames, samples, &index, &crossed)
}

fn offset_frame_pair(
    model: &TightBindingModel,
    frames: &[ComplexMatrix],
    samples: &[usize],
    base: &[usize],
    first: usize,
    second: usize,
) -> Result<ComplexMatrix, ChernNumberError> {
    let mut index = base.to_vec();
    let mut crossed = Vec::new();
    for axis in [first, second] {
        index[axis] += 1;
        if index[axis] == samples[axis] {
            index[axis] = 0;
            crossed.push(axis);
        }
    }
    sampled_frame(model, frames, samples, &index, &crossed)
}

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
