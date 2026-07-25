//! Propagating Bloch modes of nearest-cell periodic leads.

use std::fmt;

use nalgebra::DMatrix;

use crate::decomposition::{
    eigenvectors_from_generalized_schur, generalized_schur, DecompositionError,
};
use crate::spectrum::hermitian_eigensystem;
use crate::symmetry::{particle_hole_symmetric_basis, DiscreteSymmetry, SymmetryViolation};
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

/// Propagating modes resolved into orthogonal conservation-law subspaces.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedLeadModes {
    modes: PropagatingLeadModes,
    block_incoming_counts: Vec<usize>,
}

impl ProjectedLeadModes {
    /// Combined modes, grouped into all incoming and then all outgoing blocks.
    #[must_use]
    pub const fn modes(&self) -> &PropagatingLeadModes {
        &self.modes
    }

    /// Number of incoming modes in each projector subspace.
    #[must_use]
    pub fn block_incoming_counts(&self) -> &[usize] {
        &self.block_incoming_counts
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
    /// Conservation-law projectors are invalid or do not reduce both lead matrices.
    InvalidProjectors,
    /// Declared discrete symmetries are invalid or do not preserve the lead.
    InvalidSymmetries,
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
            Self::InvalidProjectors => write!(
                formatter,
                "projectors must be complete orthonormal subspaces that reduce the lead matrices"
            ),
            Self::InvalidSymmetries => write!(
                formatter,
                "discrete symmetries must be unitary and preserve the lead matrices"
            ),
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
            candidates.push((
                velocity,
                canonical_momentum(bloch_factor.arg()),
                bloch_factor,
                wave,
            ));
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

/// Solve propagating modes and choose a basis that obeys declared discrete
/// symmetries in both physical and stabilized representations.
pub fn propagating_modes_with_symmetries(
    cell_hamiltonian: &ComplexMatrix,
    inter_cell_hopping: &ComplexMatrix,
    time_reversal: Option<&ComplexMatrix>,
    particle_hole: Option<&ComplexMatrix>,
    chiral: Option<&ComplexMatrix>,
) -> Result<PropagatingLeadModes, LeadModeError> {
    let time_reversal = validate_lead_symmetry(
        cell_hamiltonian,
        inter_cell_hopping,
        time_reversal,
        SymmetryViolation::TimeReversal,
    )?;
    let particle_hole = validate_lead_symmetry(
        cell_hamiltonian,
        inter_cell_hopping,
        particle_hole,
        SymmetryViolation::ParticleHole,
    )?;
    let chiral = validate_lead_symmetry(
        cell_hamiltonian,
        inter_cell_hopping,
        chiral,
        SymmetryViolation::Chiral,
    )?;

    let mut modes = propagating_modes(cell_hamiltonian, inter_cell_hopping)?;
    if let Some(particle_hole) = particle_hole.as_ref() {
        adapt_particle_hole(&mut modes, particle_hole)?;
    }
    if time_reversal.is_none() {
        if let Some(chiral) = chiral.as_ref() {
            adapt_opposite_direction(&mut modes, chiral, false, true)?;
        }
    }
    if let Some(time_reversal) = time_reversal.as_ref() {
        adapt_opposite_direction(&mut modes, time_reversal, true, false)?;
    }
    Ok(modes)
}

fn validate_lead_symmetry(
    cell_hamiltonian: &ComplexMatrix,
    inter_cell_hopping: &ComplexMatrix,
    operator: Option<&ComplexMatrix>,
    kind: SymmetryViolation,
) -> Result<Option<ComplexMatrix>, LeadModeError> {
    let operator = validate_symmetry_operator(operator, kind)?;
    let Some(operator) = operator else {
        return Ok(None);
    };
    let operator_reference = Some(&operator);
    let symmetry = match kind {
        SymmetryViolation::TimeReversal => {
            DiscreteSymmetry::new(None, operator_reference.cloned(), None, None)
        }
        SymmetryViolation::ParticleHole => {
            DiscreteSymmetry::new(None, None, operator_reference.cloned(), None)
        }
        SymmetryViolation::Chiral => {
            DiscreteSymmetry::new(None, None, None, operator_reference.cloned())
        }
        SymmetryViolation::ConservationLaw => {
            return Err(LeadModeError::InvalidSymmetries);
        }
    }
    .map_err(|_| LeadModeError::InvalidSymmetries)?;
    for matrix in [cell_hamiltonian, inter_cell_hopping] {
        if symmetry
            .validate(matrix)
            .map_err(|_| LeadModeError::InvalidSymmetries)?
            .contains(&kind)
        {
            return Err(LeadModeError::InvalidSymmetries);
        }
    }
    Ok(Some(operator.clone()))
}

fn validate_symmetry_operator(
    operator: Option<&ComplexMatrix>,
    kind: SymmetryViolation,
) -> Result<Option<ComplexMatrix>, LeadModeError> {
    let Some(operator) = operator else {
        return Ok(None);
    };
    let symmetry = match kind {
        SymmetryViolation::TimeReversal => {
            DiscreteSymmetry::new(None, Some(operator.clone()), None, None)
        }
        SymmetryViolation::ParticleHole => {
            DiscreteSymmetry::new(None, None, Some(operator.clone()), None)
        }
        SymmetryViolation::Chiral => {
            DiscreteSymmetry::new(None, None, None, Some(operator.clone()))
        }
        SymmetryViolation::ConservationLaw => {
            return Err(LeadModeError::InvalidSymmetries);
        }
    }
    .map_err(|_| LeadModeError::InvalidSymmetries)?;
    let validated = match kind {
        SymmetryViolation::TimeReversal => symmetry.time_reversal(),
        SymmetryViolation::ParticleHole => symmetry.particle_hole(),
        SymmetryViolation::Chiral => symmetry.chiral(),
        SymmetryViolation::ConservationLaw => None,
    }
    .ok_or(LeadModeError::InvalidSymmetries)?;
    Ok(Some(validated.clone()))
}

/// Solve propagating modes independently in complete orthogonal subspaces.
///
/// The returned wave functions live in the original physical basis. The
/// stabilized vectors live in the concatenated projector basis, and the
/// square-root hopping maps that basis back to the physical basis.
pub fn propagating_modes_in_subspaces(
    cell_hamiltonian: &ComplexMatrix,
    inter_cell_hopping: &ComplexMatrix,
    projectors: &[ComplexMatrix],
) -> Result<ProjectedLeadModes, LeadModeError> {
    let dimension = cell_hamiltonian.rows();
    if cell_hamiltonian.shape() != (dimension, dimension)
        || inter_cell_hopping.shape() != (dimension, dimension)
        || projectors.is_empty()
    {
        return Err(LeadModeError::InvalidShape);
    }
    let symmetry = DiscreteSymmetry::new(Some(projectors.to_vec()), None, None, None)
        .map_err(|_| LeadModeError::InvalidProjectors)?;
    if symmetry
        .validate(cell_hamiltonian)
        .map_err(|_| LeadModeError::InvalidProjectors)?
        .contains(&SymmetryViolation::ConservationLaw)
        || symmetry
            .validate(inter_cell_hopping)
            .map_err(|_| LeadModeError::InvalidProjectors)?
            .contains(&SymmetryViolation::ConservationLaw)
    {
        return Err(LeadModeError::InvalidProjectors);
    }
    let projectors = symmetry
        .projectors()
        .ok_or(LeadModeError::InvalidProjectors)?;
    let cell = backend(cell_hamiltonian);
    let hopping = backend(inter_cell_hopping);
    let mut blocks = Vec::with_capacity(projectors.len());
    for projector in projectors {
        let projector_dense = backend(projector);
        let projected_cell = projector_dense.adjoint() * &cell * &projector_dense;
        let projected_hopping = projector_dense.adjoint() * &hopping * &projector_dense;
        let modes = propagating_modes(&owned(&projected_cell)?, &owned(&projected_hopping)?)?;
        if modes.wave_functions().columns() != 2 * modes.incoming_count() {
            return Err(LeadModeError::DecompositionFailure);
        }
        blocks.push((projector_dense, modes));
    }

    combine_projected_blocks(dimension, &blocks)
}

/// Solve conservation-law blocks under declared discrete-symmetry relations.
///
/// Operators are checked for unitary structure and canonical block mappings.
/// Their physical relation to the Hamiltonian is treated as a caller
/// declaration, matching workflows that construct a target energy block from
/// a symmetry-related source block.
pub fn propagating_modes_in_declared_symmetric_subspaces(
    cell_hamiltonian: &ComplexMatrix,
    inter_cell_hopping: &ComplexMatrix,
    projectors: &[ComplexMatrix],
    time_reversal: Option<&ComplexMatrix>,
    particle_hole: Option<&ComplexMatrix>,
    chiral: Option<&ComplexMatrix>,
) -> Result<ProjectedLeadModes, LeadModeError> {
    let dimension = cell_hamiltonian.rows();
    if cell_hamiltonian.shape() != (dimension, dimension)
        || inter_cell_hopping.shape() != (dimension, dimension)
        || projectors.is_empty()
    {
        return Err(LeadModeError::InvalidShape);
    }
    let projector_symmetry = DiscreteSymmetry::new(Some(projectors.to_vec()), None, None, None)
        .map_err(|_| LeadModeError::InvalidProjectors)?;
    for matrix in [cell_hamiltonian, inter_cell_hopping] {
        if projector_symmetry
            .validate(matrix)
            .map_err(|_| LeadModeError::InvalidProjectors)?
            .contains(&SymmetryViolation::ConservationLaw)
        {
            return Err(LeadModeError::InvalidProjectors);
        }
    }
    let projectors = projector_symmetry
        .projectors()
        .ok_or(LeadModeError::InvalidProjectors)?
        .iter()
        .map(backend)
        .collect::<Vec<_>>();
    let time_reversal = validate_symmetry_operator(time_reversal, SymmetryViolation::TimeReversal)?
        .map(|matrix| backend(&matrix));
    let particle_hole = validate_symmetry_operator(particle_hole, SymmetryViolation::ParticleHole)?
        .map(|matrix| backend(&matrix));
    let chiral = validate_symmetry_operator(chiral, SymmetryViolation::Chiral)?
        .map(|matrix| backend(&matrix));
    validate_canonical_block_maps(
        &projectors,
        [
            time_reversal.as_ref().map(|operator| (operator, true)),
            particle_hole.as_ref().map(|operator| (operator, true)),
            chiral.as_ref().map(|operator| (operator, false)),
        ],
    )?;

    let cell = backend(cell_hamiltonian);
    let hopping = backend(inter_cell_hopping);
    let projected_cells = projectors
        .iter()
        .map(|projector| projector.adjoint() * &cell * projector)
        .collect::<Vec<_>>();
    let projected_hoppings = projectors
        .iter()
        .map(|projector| projector.adjoint() * &hopping * projector)
        .collect::<Vec<_>>();
    let mut modes: Vec<Option<PropagatingLeadModes>> = vec![None; projectors.len()];

    for target in 0..projectors.len() {
        let mut transformed = None;
        for source in 0..target {
            let Some(source_modes) = modes[source].as_ref() else {
                continue;
            };
            if projected_cells[source].shape() != projected_cells[target].shape() {
                continue;
            }
            if matrices_are_close(&projected_cells[source], &projected_cells[target])
                && matrices_are_close(&projected_hoppings[source], &projected_hoppings[target])
            {
                transformed = Some(source_modes.clone());
                break;
            }
            for (operator, relation) in [
                (time_reversal.as_ref(), BlockRelation::TimeReversal),
                (particle_hole.as_ref(), BlockRelation::ParticleHole),
                (chiral.as_ref(), BlockRelation::Chiral),
            ] {
                let Some(operator) = operator else {
                    continue;
                };
                let block = projected_symmetry_block(
                    &projectors[target],
                    operator,
                    &projectors[source],
                    relation.is_antiunitary(),
                );
                if maximum_norm(&block) > 1.0e-7 {
                    transformed = Some(transform_block_modes(source_modes, &block, relation)?);
                    break;
                }
            }
            if transformed.is_some() {
                break;
            }
        }

        modes[target] = Some(if let Some(transformed) = transformed {
            transformed
        } else {
            let local_time_reversal = time_reversal.as_ref().and_then(|operator| {
                nonzero_projected_symmetry(&projectors[target], operator, true)
            });
            let local_particle_hole = particle_hole.as_ref().and_then(|operator| {
                nonzero_projected_symmetry(&projectors[target], operator, true)
            });
            let local_chiral = chiral.as_ref().and_then(|operator| {
                nonzero_projected_symmetry(&projectors[target], operator, false)
            });
            propagating_modes_with_symmetries(
                &owned(&projected_cells[target])?,
                &owned(&projected_hoppings[target])?,
                local_time_reversal.as_ref(),
                local_particle_hole.as_ref(),
                local_chiral.as_ref(),
            )?
        });
    }

    let blocks = projectors
        .into_iter()
        .zip(modes)
        .map(|(projector, modes)| {
            modes
                .map(|modes| (projector, modes))
                .ok_or(LeadModeError::DecompositionFailure)
        })
        .collect::<Result<Vec<_>, _>>()?;
    combine_projected_blocks(dimension, &blocks)
}

#[derive(Clone, Copy)]
enum BlockRelation {
    TimeReversal,
    ParticleHole,
    Chiral,
}

impl BlockRelation {
    const fn is_antiunitary(self) -> bool {
        matches!(self, Self::TimeReversal | Self::ParticleHole)
    }
}

fn transform_block_modes(
    source: &PropagatingLeadModes,
    operator: &DMatrix<Complex64>,
    relation: BlockRelation,
) -> Result<PropagatingLeadModes, LeadModeError> {
    let incoming = source.incoming_count();
    let mode_count = source.wave_functions().columns();
    if mode_count != 2 * incoming
        || operator.shape()
            != (
                source.wave_functions().rows(),
                source.wave_functions().rows(),
            )
    {
        return Err(LeadModeError::InvalidSymmetries);
    }
    let permutation = match relation {
        BlockRelation::TimeReversal => (0..mode_count).rev().collect::<Vec<_>>(),
        BlockRelation::ParticleHole => (0..mode_count)
            .map(|index| {
                mode_count - 1 - index - incoming * usize::from(index < incoming)
                    + incoming * usize::from(index >= incoming)
            })
            .collect(),
        BlockRelation::Chiral => (0..mode_count)
            .map(|index| {
                if index < incoming {
                    incoming + index
                } else {
                    index - incoming
                }
            })
            .collect(),
    };
    let conjugate = relation.is_antiunitary();
    let flip_energy = matches!(
        relation,
        BlockRelation::ParticleHole | BlockRelation::Chiral
    );
    let wave_functions = transform_block_matrix(
        source.wave_functions(),
        operator,
        &permutation,
        conjugate,
        false,
    )?;
    let stabilized_vectors =
        permute_block_matrix(source.stabilized_vectors(), &permutation, conjugate, false)?;
    let stabilized_vectors_lambda_inverse = permute_block_matrix(
        source.stabilized_vectors_lambda_inverse(),
        &permutation,
        conjugate,
        flip_energy,
    )?;
    let square_root_hopping =
        transform_square_root(source.square_root_hopping(), operator, conjugate)?;
    let velocity_sign = if flip_energy != conjugate { -1.0 } else { 1.0 };
    let velocities = permutation
        .iter()
        .map(|&index| velocity_sign * source.velocities()[index])
        .collect();
    let momenta = permutation
        .iter()
        .map(|&index| {
            canonical_momentum(if conjugate {
                -source.momenta()[index]
            } else {
                source.momenta()[index]
            })
        })
        .collect();
    Ok(PropagatingLeadModes {
        wave_functions,
        velocities,
        momenta,
        incoming_count: incoming,
        stabilized_vectors,
        stabilized_vectors_lambda_inverse,
        square_root_hopping,
    })
}

fn transform_block_matrix(
    matrix: &ComplexMatrix,
    operator: &DMatrix<Complex64>,
    permutation: &[usize],
    conjugate: bool,
    negate: bool,
) -> Result<ComplexMatrix, LeadModeError> {
    let transformed = permute_dense(&backend(matrix), permutation, conjugate, negate);
    owned(&(operator * transformed))
}

fn permute_block_matrix(
    matrix: &ComplexMatrix,
    permutation: &[usize],
    conjugate: bool,
    negate: bool,
) -> Result<ComplexMatrix, LeadModeError> {
    owned(&permute_dense(
        &backend(matrix),
        permutation,
        conjugate,
        negate,
    ))
}

fn permute_dense(
    matrix: &DMatrix<Complex64>,
    permutation: &[usize],
    conjugate: bool,
    negate: bool,
) -> DMatrix<Complex64> {
    DMatrix::from_fn(matrix.nrows(), permutation.len(), |row, column| {
        let mut value = matrix[(row, permutation[column])];
        if conjugate {
            value = value.conj();
        }
        if negate {
            value = -value;
        }
        value
    })
}

fn transform_square_root(
    square_root: &ComplexMatrix,
    operator: &DMatrix<Complex64>,
    conjugate: bool,
) -> Result<ComplexMatrix, LeadModeError> {
    let square_root = backend(square_root);
    let square_root = if conjugate {
        square_root.map(|value| value.conj())
    } else {
        square_root
    };
    owned(&(operator * square_root))
}

fn validate_canonical_block_maps(
    projectors: &[DMatrix<Complex64>],
    operators: [Option<(&DMatrix<Complex64>, bool)>; 3],
) -> Result<(), LeadModeError> {
    for (operator, antiunitary) in operators.into_iter().flatten() {
        for target in projectors {
            let nonzero_sources = projectors
                .iter()
                .filter(|source| {
                    maximum_norm(&projected_symmetry_block(
                        target,
                        operator,
                        source,
                        antiunitary,
                    )) > 1.0e-7
                })
                .count();
            if nonzero_sources != 1 {
                return Err(LeadModeError::InvalidSymmetries);
            }
        }
    }
    Ok(())
}

fn nonzero_projected_symmetry(
    projector: &DMatrix<Complex64>,
    operator: &DMatrix<Complex64>,
    antiunitary: bool,
) -> Option<ComplexMatrix> {
    let block = projected_symmetry_block(projector, operator, projector, antiunitary);
    (maximum_norm(&block) > 1.0e-7)
        .then(|| owned(&block).ok())
        .flatten()
}

fn projected_symmetry_block(
    target: &DMatrix<Complex64>,
    operator: &DMatrix<Complex64>,
    source: &DMatrix<Complex64>,
    antiunitary: bool,
) -> DMatrix<Complex64> {
    let source = if antiunitary {
        source.map(|value| value.conj())
    } else {
        source.clone()
    };
    target.adjoint() * operator * source
}

fn matrices_are_close(left: &DMatrix<Complex64>, right: &DMatrix<Complex64>) -> bool {
    left.shape() == right.shape() && maximum_norm(&(left - right)) <= 1.0e-8
}

fn maximum_norm(matrix: &DMatrix<Complex64>) -> f64 {
    matrix.iter().map(|value| value.norm()).fold(0.0, f64::max)
}

fn combine_projected_blocks(
    dimension: usize,
    blocks: &[(DMatrix<Complex64>, PropagatingLeadModes)],
) -> Result<ProjectedLeadModes, LeadModeError> {
    let block_incoming_counts = blocks
        .iter()
        .map(|(_, modes)| modes.incoming_count())
        .collect::<Vec<_>>();
    let total_incoming = block_incoming_counts.iter().sum::<usize>();
    let total_modes = 2 * total_incoming;
    let mut wave_functions = DMatrix::<Complex64>::zeros(dimension, total_modes);
    let mut stabilized_vectors = DMatrix::<Complex64>::zeros(dimension, total_modes);
    let mut stabilized_vectors_lambda_inverse = DMatrix::<Complex64>::zeros(dimension, total_modes);
    let mut square_root_hopping = DMatrix::<Complex64>::zeros(dimension, dimension);
    let mut velocities = vec![0.0; total_modes];
    let mut momenta = vec![0.0; total_modes];
    let mut row_offset = 0;
    let mut mode_offset = 0;

    for (projector, modes) in blocks {
        let block_dimension = projector.ncols();
        let block_incoming = modes.incoming_count();
        let lifted_wave_functions = projector * backend(modes.wave_functions());
        let block_vectors = backend(modes.stabilized_vectors());
        let block_vectors_lambda_inverse = backend(modes.stabilized_vectors_lambda_inverse());
        let lifted_square_root = projector * backend(modes.square_root_hopping());

        for row in 0..dimension {
            for column in 0..block_dimension {
                square_root_hopping[(row, row_offset + column)] = lifted_square_root[(row, column)];
            }
        }
        for direction in 0..2 {
            for local_mode in 0..block_incoming {
                let local_column = direction * block_incoming + local_mode;
                let global_column = direction * total_incoming + mode_offset + local_mode;
                velocities[global_column] = modes.velocities()[local_column];
                momenta[global_column] = modes.momenta()[local_column];
                for row in 0..dimension {
                    wave_functions[(row, global_column)] =
                        lifted_wave_functions[(row, local_column)];
                }
                for row in 0..block_dimension {
                    stabilized_vectors[(row_offset + row, global_column)] =
                        block_vectors[(row, local_column)];
                    stabilized_vectors_lambda_inverse[(row_offset + row, global_column)] =
                        block_vectors_lambda_inverse[(row, local_column)];
                }
            }
        }
        row_offset += block_dimension;
        mode_offset += block_incoming;
    }

    Ok(ProjectedLeadModes {
        modes: PropagatingLeadModes {
            wave_functions: owned(&wave_functions)?,
            velocities,
            momenta,
            incoming_count: total_incoming,
            stabilized_vectors: owned(&stabilized_vectors)?,
            stabilized_vectors_lambda_inverse: owned(&stabilized_vectors_lambda_inverse)?,
            square_root_hopping: owned(&square_root_hopping)?,
        },
        block_incoming_counts,
    })
}

fn adapt_opposite_direction(
    modes: &mut PropagatingLeadModes,
    operator: &ComplexMatrix,
    conjugate_source: bool,
    reverse_source: bool,
) -> Result<(), LeadModeError> {
    let incoming = modes.incoming_count;
    if modes.wave_functions.columns() != 2 * incoming {
        return Err(LeadModeError::DecompositionFailure);
    }
    let source_columns = if reverse_source {
        (0..incoming).rev().collect::<Vec<_>>()
    } else {
        (0..incoming).collect::<Vec<_>>()
    };
    let source = select_columns(&backend(&modes.wave_functions), &source_columns);
    let source = if conjugate_source {
        source.map(|value| value.conj())
    } else {
        source
    };
    let desired = backend(operator) * source;
    let target_columns = (incoming..2 * incoming).collect::<Vec<_>>();
    replace_mode_columns(modes, &target_columns, &desired)
}

fn adapt_particle_hole(
    modes: &mut PropagatingLeadModes,
    particle_hole: &ComplexMatrix,
) -> Result<(), LeadModeError> {
    let incoming = modes.incoming_count;
    if modes.wave_functions.columns() != 2 * incoming {
        return Err(LeadModeError::DecompositionFailure);
    }
    let momentum_tolerance = 1.0e-7;
    let velocity_tolerance = 1.0e-7;
    let particle_hole_dense = backend(particle_hole);

    for direction in [0..incoming, incoming..2 * incoming] {
        let positive = direction
            .clone()
            .filter(|&column| {
                modes.momenta[column] > momentum_tolerance
                    && modes.momenta[column] < std::f64::consts::PI - momentum_tolerance
            })
            .collect::<Vec<_>>();
        let negative = direction
            .clone()
            .filter(|&column| {
                modes.momenta[column] < -momentum_tolerance
                    && modes.momenta[column] > -std::f64::consts::PI + momentum_tolerance
            })
            .collect::<Vec<_>>();
        if positive.len() != negative.len() {
            return Err(LeadModeError::InvalidSymmetries);
        }
        if !negative.is_empty() {
            let wave_functions = backend(&modes.wave_functions);
            let mut used = vec![false; positive.len()];
            let mut desired = DMatrix::<Complex64>::zeros(wave_functions.nrows(), negative.len());
            for (target_position, &target_column) in negative.iter().enumerate() {
                let (source_position, &source_column) = positive
                    .iter()
                    .enumerate()
                    .filter(|(position, _)| !used[*position])
                    .min_by(|(_, left), (_, right)| {
                        (modes.momenta[**left] + modes.momenta[target_column])
                            .abs()
                            .total_cmp(
                                &(modes.momenta[**right] + modes.momenta[target_column]).abs(),
                            )
                    })
                    .ok_or(LeadModeError::InvalidSymmetries)?;
                used[source_position] = true;
                let source = wave_functions
                    .column(source_column)
                    .map(|value| value.conj());
                let transformed = &particle_hole_dense * source;
                desired.set_column(target_position, &transformed);
            }
            replace_mode_columns(modes, &negative, &desired)?;
        }

        let mut assigned = vec![false; modes.wave_functions.columns()];
        for seed in direction.clone() {
            let momentum = modes.momenta[seed];
            let at_trim = momentum.abs() <= momentum_tolerance
                || (momentum.abs() - std::f64::consts::PI).abs() <= momentum_tolerance;
            if !at_trim || assigned[seed] {
                continue;
            }
            let group = direction
                .clone()
                .filter(|&column| {
                    !assigned[column]
                        && trim_distance(modes.momenta[column], momentum) <= momentum_tolerance
                        && (modes.velocities[column] - modes.velocities[seed]).abs()
                            <= velocity_tolerance * modes.velocities[seed].abs().max(1.0)
                })
                .collect::<Vec<_>>();
            for &column in &group {
                assigned[column] = true;
            }
            let wave_functions = backend(&modes.wave_functions);
            let mut normalized = select_columns(&wave_functions, &group);
            for (local, &column) in group.iter().enumerate() {
                normalized
                    .column_mut(local)
                    .scale_mut(modes.velocities[column].abs().sqrt());
            }
            let normalized = owned(&normalized)?;
            let adapted = particle_hole_symmetric_basis(&normalized, particle_hole)
                .map_err(|_| LeadModeError::InvalidSymmetries)?;
            let mut desired = backend(adapted.wave_functions());
            for (local, &column) in group.iter().enumerate() {
                desired
                    .column_mut(local)
                    .scale_mut(modes.velocities[column].abs().sqrt().recip());
            }
            replace_mode_columns(modes, &group, &desired)?;
        }
    }
    Ok(())
}

fn trim_distance(left: f64, right: f64) -> f64 {
    if left.abs() <= 1.0e-7 && right.abs() <= 1.0e-7 {
        0.0
    } else {
        (left.abs() - right.abs()).abs()
    }
}

fn canonical_momentum(momentum: f64) -> f64 {
    if (momentum - std::f64::consts::PI).abs() <= 1.0e-12 {
        -std::f64::consts::PI
    } else {
        momentum
    }
}

fn replace_mode_columns(
    modes: &mut PropagatingLeadModes,
    columns: &[usize],
    desired: &DMatrix<Complex64>,
) -> Result<(), LeadModeError> {
    if columns.is_empty() {
        return Ok(());
    }
    let wave_functions = backend(&modes.wave_functions);
    let current = select_columns(&wave_functions, columns);
    if current.shape() != desired.shape() {
        return Err(LeadModeError::InvalidSymmetries);
    }
    let rotation = current
        .clone()
        .svd(true, true)
        .solve(desired, 1.0e-10)
        .map_err(|_| LeadModeError::DecompositionFailure)?;
    let residual = &current * &rotation - desired;
    if residual
        .iter()
        .map(|value| value.norm())
        .fold(0.0, f64::max)
        > 1.0e-6
    {
        return Err(LeadModeError::InvalidSymmetries);
    }

    modes.wave_functions = replace_columns(&wave_functions, columns, desired)?;
    let vectors = backend(&modes.stabilized_vectors);
    let transformed_vectors = select_columns(&vectors, columns) * &rotation;
    modes.stabilized_vectors = replace_columns(&vectors, columns, &transformed_vectors)?;
    let vectors_lambda_inverse = backend(&modes.stabilized_vectors_lambda_inverse);
    let transformed_vectors_lambda_inverse =
        select_columns(&vectors_lambda_inverse, columns) * rotation;
    modes.stabilized_vectors_lambda_inverse = replace_columns(
        &vectors_lambda_inverse,
        columns,
        &transformed_vectors_lambda_inverse,
    )?;
    Ok(())
}

fn select_columns(matrix: &DMatrix<Complex64>, columns: &[usize]) -> DMatrix<Complex64> {
    DMatrix::from_fn(matrix.nrows(), columns.len(), |row, column| {
        matrix[(row, columns[column])]
    })
}

fn replace_columns(
    matrix: &DMatrix<Complex64>,
    columns: &[usize],
    replacement: &DMatrix<Complex64>,
) -> Result<ComplexMatrix, LeadModeError> {
    if replacement.shape() != (matrix.nrows(), columns.len()) {
        return Err(LeadModeError::InvalidShape);
    }
    let mut result = matrix.clone();
    for (replacement_column, &target_column) in columns.iter().enumerate() {
        result.set_column(target_column, &replacement.column(replacement_column));
    }
    owned(&result)
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

    #[test]
    fn complex_projectors_produce_block_resolved_modes() {
        let inverse_sqrt_two = 2.0_f64.sqrt().recip();
        let projectors = vec![
            ComplexMatrix::new(
                2,
                1,
                vec![
                    Complex64::new(inverse_sqrt_two, 0.0),
                    Complex64::new(0.0, inverse_sqrt_two),
                ],
            )
            .unwrap(),
            ComplexMatrix::new(
                2,
                1,
                vec![
                    Complex64::new(0.0, inverse_sqrt_two),
                    Complex64::new(inverse_sqrt_two, 0.0),
                ],
            )
            .unwrap(),
        ];
        let cell = ComplexMatrix::new(
            2,
            2,
            vec![
                Complex64::new(0.3, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.3, 0.0),
            ],
        )
        .unwrap();
        let hopping = ComplexMatrix::new(
            2,
            2,
            vec![
                Complex64::new(0.7, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.7, 0.0),
            ],
        )
        .unwrap();

        let projected = propagating_modes_in_subspaces(&cell, &hopping, &projectors).unwrap();
        assert_eq!(projected.block_incoming_counts(), &[1, 1]);
        let modes = projected.modes();
        assert_eq!(modes.incoming_count(), 2);
        for column in [0, 2] {
            assert!(modes.stabilized_vectors().get(0, column).unwrap().norm() > 0.0);
            assert_eq!(
                modes.stabilized_vectors().get(1, column).unwrap(),
                Complex64::new(0.0, 0.0)
            );
        }
        for column in [1, 3] {
            assert!(modes.stabilized_vectors().get(1, column).unwrap().norm() > 0.0);
            assert_eq!(
                modes.stabilized_vectors().get(0, column).unwrap(),
                Complex64::new(0.0, 0.0)
            );
        }
        let square_root = modes.square_root_hopping();
        for row in 0..2 {
            for (column, projector) in projectors.iter().enumerate() {
                let expected = projector.get(row, 0).unwrap() * 0.7_f64.sqrt();
                assert!((square_root.get(row, column).unwrap() - expected).norm() < 1.0e-12);
            }
        }
    }

    #[test]
    fn time_reversal_fixes_the_relative_mode_gauge() {
        let modes = propagating_modes_with_symmetries(
            &ComplexMatrix::scalar(Complex64::new(0.3, 0.0)),
            &ComplexMatrix::scalar(Complex64::new(0.7, 0.0)),
            Some(&ComplexMatrix::identity(1)),
            None,
            None,
        )
        .unwrap();
        assert_eq!(modes.incoming_count(), 1);
        assert!(
            (modes.wave_functions().get(0, 1).unwrap()
                - modes.wave_functions().get(0, 0).unwrap().conj())
            .norm()
                < 1.0e-10
        );
    }

    #[test]
    fn square_plus_one_particle_hole_modes_are_real_at_trim() {
        let modes = propagating_modes_with_symmetries(
            &ComplexMatrix::scalar(Complex64::new(0.0, 0.0)),
            &ComplexMatrix::scalar(Complex64::new(0.0, 0.7)),
            None,
            Some(&ComplexMatrix::identity(1)),
            None,
        )
        .unwrap();
        assert_eq!(modes.incoming_count(), 1);
        for &value in modes.wave_functions().as_slice() {
            assert!(value.im.abs() < 1.0e-10);
        }
    }

    #[test]
    fn symmetry_related_projector_blocks_share_one_mode_solution() {
        let projectors = vec![
            ComplexMatrix::new(
                2,
                1,
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            )
            .unwrap(),
            ComplexMatrix::new(
                2,
                1,
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            )
            .unwrap(),
        ];
        let cell = ComplexMatrix::new(
            2,
            2,
            vec![
                Complex64::new(0.3, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.3, 0.0),
            ],
        )
        .unwrap();
        let hopping = ComplexMatrix::new(
            2,
            2,
            vec![
                Complex64::new(0.7, 0.2),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.7, -0.2),
            ],
        )
        .unwrap();
        let time_reversal = ComplexMatrix::new(
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

        let projected = propagating_modes_in_declared_symmetric_subspaces(
            &cell,
            &hopping,
            &projectors,
            Some(&time_reversal),
            None,
            None,
        )
        .unwrap();
        let wave_functions = projected.modes().wave_functions();
        for (source, target) in [(2, 1), (0, 3)] {
            for row in 0..2 {
                let transformed = (0..2)
                    .map(|column| {
                        time_reversal.get(row, column).unwrap()
                            * wave_functions.get(column, source).unwrap().conj()
                    })
                    .sum::<Complex64>();
                assert!((wave_functions.get(row, target).unwrap() - transformed).norm() < 1.0e-10);
            }
        }
    }
}
