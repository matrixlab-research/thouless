//! Steady-state coherent transport through finite devices and periodic leads.

use std::collections::BTreeMap;

use nalgebra::DMatrix;

use crate::linear_operator::{
    gmres_with_right_preconditioner, CsrMatrix, GmresOptions, Ilu0Preconditioner,
    IterativeSolveError, LinearOperatorError,
};
use crate::spectrum::hermitian_eigensystem;
use crate::{Complex64, ComplexMatrix, MatrixError};

const HERMITIAN_TOLERANCE: f64 = 1.0e-12;

/// One semi-infinite periodic lead coupled to a finite device.
///
/// `inter_cell_hopping` maps the next lead cell into the surface cell.
/// `coupling` maps the surface-cell basis into the device basis.
#[derive(Clone, Debug, PartialEq)]
pub struct LeadContact {
    cell_hamiltonian: ComplexMatrix,
    inter_cell_hopping: ComplexMatrix,
    coupling: ComplexMatrix,
}

impl LeadContact {
    /// Creates and validates a periodic lead contact.
    pub fn new(
        cell_hamiltonian: ComplexMatrix,
        inter_cell_hopping: ComplexMatrix,
        coupling: ComplexMatrix,
    ) -> Result<Self, TransportError> {
        let cell_count = cell_hamiltonian.rows();
        if cell_count == 0
            || cell_hamiltonian.columns() != cell_count
            || inter_cell_hopping.shape() != (cell_count, cell_count)
            || coupling.columns() != cell_count
        {
            return Err(TransportError::InvalidLeadShape);
        }
        if !cell_hamiltonian.is_hermitian(HERMITIAN_TOLERANCE)? {
            return Err(TransportError::NonHermitianLead);
        }
        Ok(Self {
            cell_hamiltonian,
            inter_cell_hopping,
            coupling,
        })
    }

    /// Returns the Hamiltonian of one principal lead cell.
    #[must_use]
    pub const fn cell_hamiltonian(&self) -> &ComplexMatrix {
        &self.cell_hamiltonian
    }

    /// Returns the hopping from the next cell into the surface cell.
    #[must_use]
    pub const fn inter_cell_hopping(&self) -> &ComplexMatrix {
        &self.inter_cell_hopping
    }

    /// Returns the device-by-lead coupling matrix.
    #[must_use]
    pub const fn coupling(&self) -> &ComplexMatrix {
        &self.coupling
    }
}

/// Numerical controls for retarded surface Green functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceGreenOptions {
    /// Positive imaginary broadening added to energy.
    pub broadening: f64,
    /// Frobenius-norm convergence threshold for renormalized hoppings.
    pub tolerance: f64,
    /// Maximum number of decimation iterations.
    pub max_iterations: usize,
}

impl Default for SurfaceGreenOptions {
    fn default() -> Self {
        Self {
            broadening: 1.0e-6,
            tolerance: 1.0e-13,
            max_iterations: 256,
        }
    }
}

/// Retarded open-system solution at one energy.
#[derive(Clone, Debug, PartialEq)]
pub struct ScatteringSolution {
    retarded_green: ComplexMatrix,
    self_energies: Vec<ComplexMatrix>,
    broadenings: Vec<ComplexMatrix>,
}

impl ScatteringSolution {
    /// Returns the retarded device Green function.
    #[must_use]
    pub const fn retarded_green(&self) -> &ComplexMatrix {
        &self.retarded_green
    }

    /// Returns embedded retarded lead self-energies.
    #[must_use]
    pub fn self_energies(&self) -> &[ComplexMatrix] {
        &self.self_energies
    }

    /// Returns embedded lead broadening matrices.
    #[must_use]
    pub fn broadenings(&self) -> &[ComplexMatrix] {
        &self.broadenings
    }

    /// Returns all Caroli transmissions as `[drain][source]`.
    pub fn transmission_matrix(&self) -> Result<Vec<Vec<f64>>, TransportError> {
        (0..self.broadenings.len())
            .map(|drain| {
                (0..self.broadenings.len())
                    .map(|source| self.transmission(drain, source))
                    .collect()
            })
            .collect()
    }

    /// Returns the Caroli transmission from `source` into `drain`.
    pub fn transmission(&self, drain: usize, source: usize) -> Result<f64, TransportError> {
        let drain_broadening = self
            .broadenings
            .get(drain)
            .ok_or(TransportError::UnknownLead { lead: drain })?;
        let source_broadening = self
            .broadenings
            .get(source)
            .ok_or(TransportError::UnknownLead { lead: source })?;
        let green = to_backend(&self.retarded_green);
        let product =
            to_backend(drain_broadening) * &green * to_backend(source_broadening) * green.adjoint();
        let value = product.trace().re;
        if value < 0.0 && value.abs() <= 1.0e-9 {
            Ok(0.0)
        } else {
            Ok(value)
        }
    }

    /// Returns `-Im G^r_ii / π` for every device orbital.
    #[must_use]
    pub fn local_density_of_states(&self) -> Vec<f64> {
        let dimension = self.retarded_green.rows();
        (0..dimension)
            .map(|index| {
                -self.retarded_green.as_slice()[index * dimension + index].im / std::f64::consts::PI
            })
            .collect()
    }

    /// Applies the retarded Green function to lead-injection factors.
    ///
    /// Every input matrix has shape `device orbitals × incoming channels`.
    /// Returned matrices use the source-compatible layout
    /// `incoming channels × device orbitals`.
    pub fn scattering_states(
        &self,
        injection_factors: &[ComplexMatrix],
    ) -> Result<Vec<ComplexMatrix>, TransportError> {
        let device_dimension = self.retarded_green.rows();
        let green = to_backend(&self.retarded_green);
        injection_factors
            .iter()
            .enumerate()
            .map(|(lead, factor)| {
                if factor.rows() != device_dimension {
                    return Err(TransportError::InvalidScatteringFactor { lead });
                }
                let states = &green * to_backend(factor);
                ComplexMatrix::new(
                    states.ncols(),
                    states.nrows(),
                    (0..states.ncols())
                        .flat_map(|channel| {
                            let states = &states;
                            (0..states.nrows()).map(move |orbital| states[(orbital, channel)])
                        })
                        .collect(),
                )
                .map_err(Into::into)
            })
            .collect()
    }

    /// Constructs the full Fisher-Lee scattering matrix.
    ///
    /// Incoming and outgoing factors for one lead must have the same channel
    /// count and use shape `device orbitals × channels`. The direct term is
    /// obtained from the Moore-Penrose inverse of the outgoing factor.
    pub fn scattering_matrix(
        &self,
        incoming_factors: &[ComplexMatrix],
        outgoing_factors: &[ComplexMatrix],
        relative_tolerance: f64,
    ) -> Result<ScatteringMatrix, TransportError> {
        let lead_count = self.broadenings.len();
        let device_dimension = self.retarded_green.rows();
        if incoming_factors.len() != lead_count
            || outgoing_factors.len() != lead_count
            || !relative_tolerance.is_finite()
            || relative_tolerance <= 0.0
        {
            return Err(TransportError::InvalidScatteringFactors);
        }
        let mut offsets: Vec<usize> = Vec::with_capacity(lead_count + 1);
        offsets.push(0);
        for lead in 0..lead_count {
            let incoming = &incoming_factors[lead];
            let outgoing = &outgoing_factors[lead];
            if incoming.rows() != device_dimension
                || outgoing.rows() != device_dimension
                || incoming.columns() != outgoing.columns()
            {
                return Err(TransportError::InvalidScatteringFactor { lead });
            }
            offsets.push(
                offsets
                    .last()
                    .copied()
                    .expect("offset zero was inserted")
                    .checked_add(incoming.columns())
                    .ok_or(TransportError::InvalidScatteringFactors)?,
            );
        }
        let channel_count = *offsets.last().expect("offset zero was inserted");
        let green = to_backend(&self.retarded_green);
        let incoming = incoming_factors.iter().map(to_backend).collect::<Vec<_>>();
        let outgoing = outgoing_factors.iter().map(to_backend).collect::<Vec<_>>();
        let mut matrix = DMatrix::<Complex64>::zeros(channel_count, channel_count);
        for drain in 0..lead_count {
            let drain_start = offsets[drain];
            let drain_channels = outgoing[drain].ncols();
            for source in 0..lead_count {
                let source_start = offsets[source];
                let source_channels = incoming[source].ncols();
                let mut block =
                    outgoing[drain].adjoint() * &green * &incoming[source] * -Complex64::i();
                if drain == source && source_channels > 0 {
                    let scale = frobenius_norm(&outgoing[drain]).max(1.0);
                    let inverse = outgoing[drain]
                        .clone()
                        .pseudo_inverse(relative_tolerance * scale)
                        .map_err(|_| TransportError::PseudoinverseFailure)?;
                    block += inverse * &incoming[source];
                }
                for row in 0..drain_channels {
                    for column in 0..source_channels {
                        matrix[(drain_start + row, source_start + column)] = block[(row, column)];
                    }
                }
            }
        }
        Ok(ScatteringMatrix {
            matrix: from_backend(&matrix)?,
            lead_offsets: offsets,
        })
    }

    /// Returns channel counts, preserving explicit counts and inferring the
    /// remaining values from positive broadening eigenchannels.
    pub fn channel_counts(
        &self,
        explicit: &[Option<usize>],
        relative_tolerance: f64,
    ) -> Result<Vec<usize>, TransportError> {
        if explicit.len() != self.broadenings.len()
            || !relative_tolerance.is_finite()
            || relative_tolerance <= 0.0
        {
            return Err(TransportError::InvalidChannelCounts);
        }
        explicit
            .iter()
            .zip(&self.broadenings)
            .map(|(count, broadening)| {
                if let Some(count) = count {
                    return Ok(*count);
                }
                let scale = infinity_norm(broadening).max(1.0);
                let spectrum = hermitian_eigensystem(broadening, HERMITIAN_TOLERANCE)
                    .map_err(|_| TransportError::DecompositionFailure)?;
                Ok(spectrum
                    .eigenvalues()
                    .iter()
                    .filter(|value| **value > relative_tolerance * scale)
                    .count())
            })
            .collect()
    }

    /// Returns lead-to-lead values with the reflection convention used by a
    /// retarded Green-function result.
    pub fn green_function_transmission_matrix(
        &self,
        channel_counts: &[usize],
    ) -> Result<Vec<Vec<f64>>, TransportError> {
        if channel_counts.len() != self.broadenings.len() {
            return Err(TransportError::InvalidChannelCounts);
        }
        let green = to_backend(&self.retarded_green);
        let green_adjoint = green.adjoint();
        (0..self.broadenings.len())
            .map(|drain| {
                (0..self.broadenings.len())
                    .map(|source| {
                        let drain_gamma = to_backend(&self.broadenings[drain]);
                        let source_gamma = to_backend(&self.broadenings[source]);
                        let mut value = (&drain_gamma * &green * &source_gamma * &green_adjoint)
                            .trace()
                            .re;
                        if drain == source {
                            value += 2.0 * (&source_gamma * &green).trace().im
                                + channel_counts[source] as f64;
                        }
                        Ok(value)
                    })
                    .collect()
            })
            .collect()
    }

    /// Returns a square root of one lead broadening with the largest
    /// `channel_count` eigenchannels.
    pub fn broadening_factor(
        &self,
        lead: usize,
        channel_count: usize,
    ) -> Result<ComplexMatrix, TransportError> {
        let broadening = self
            .broadenings
            .get(lead)
            .ok_or(TransportError::UnknownLead { lead })?;
        let dimension = broadening.rows();
        let spectrum = hermitian_eigensystem(broadening, HERMITIAN_TOLERANCE)
            .map_err(|_| TransportError::DecompositionFailure)?;
        let first = dimension.saturating_sub(channel_count.min(dimension));
        let vectors = spectrum.eigenvectors();
        let eigenvalues = spectrum.eigenvalues();
        ComplexMatrix::new(
            dimension,
            dimension - first,
            (0..dimension)
                .flat_map(|row| {
                    (first..dimension).map(move |column| {
                        vectors.as_slice()[row * dimension + column]
                            * eigenvalues[column].max(0.0).sqrt()
                    })
                })
                .collect(),
        )
        .map_err(Into::into)
    }
}

/// A channel-resolved scattering matrix with one contiguous block per lead.
#[derive(Clone, Debug, PartialEq)]
pub struct ScatteringMatrix {
    matrix: ComplexMatrix,
    lead_offsets: Vec<usize>,
}

impl ScatteringMatrix {
    /// Full channel-space scattering matrix.
    #[must_use]
    pub const fn matrix(&self) -> &ComplexMatrix {
        &self.matrix
    }

    /// Cumulative channel offsets, including the terminal offset.
    #[must_use]
    pub fn lead_offsets(&self) -> &[usize] {
        &self.lead_offsets
    }

    /// Copies one drain/source block.
    pub fn block(&self, drain: usize, source: usize) -> Result<ComplexMatrix, TransportError> {
        let lead_count = self.lead_offsets.len().saturating_sub(1);
        if drain >= lead_count || source >= lead_count {
            return Err(TransportError::UnknownLead {
                lead: drain.max(source),
            });
        }
        let row_start = self.lead_offsets[drain];
        let row_stop = self.lead_offsets[drain + 1];
        let column_start = self.lead_offsets[source];
        let column_stop = self.lead_offsets[source + 1];
        ComplexMatrix::new(
            row_stop - row_start,
            column_stop - column_start,
            (row_start..row_stop)
                .flat_map(|row| {
                    (column_start..column_stop).map(move |column| {
                        self.matrix.as_slice()[row * self.matrix.columns() + column]
                    })
                })
                .collect(),
        )
        .map_err(Into::into)
    }

    /// Returns squared block norms as `[drain][source]`.
    pub fn transmission_matrix(&self) -> Vec<Vec<f64>> {
        let lead_count = self.lead_offsets.len().saturating_sub(1);
        (0..lead_count)
            .map(|drain| {
                let row_start = self.lead_offsets[drain];
                let row_stop = self.lead_offsets[drain + 1];
                (0..lead_count)
                    .map(|source| {
                        let column_start = self.lead_offsets[source];
                        let column_stop = self.lead_offsets[source + 1];
                        (row_start..row_stop)
                            .flat_map(|row| {
                                (column_start..column_stop).map(move |column| {
                                    self.matrix.as_slice()[row * self.matrix.columns() + column]
                                        .norm_sqr()
                                })
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
}

/// A retarded self-energy acting only on selected device orbitals.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalizedSelfEnergy {
    orbitals: Vec<usize>,
    matrix: ComplexMatrix,
}

impl LocalizedSelfEnergy {
    /// Creates a local self-energy with an explicit orbital embedding.
    pub fn new(orbitals: Vec<usize>, matrix: ComplexMatrix) -> Result<Self, TransportError> {
        if orbitals.is_empty() || matrix.shape() != (orbitals.len(), orbitals.len()) {
            return Err(TransportError::InvalidLocalizedSelfEnergy);
        }
        let mut unique = orbitals.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != orbitals.len() {
            return Err(TransportError::InvalidLocalizedSelfEnergy);
        }
        Ok(Self { orbitals, matrix })
    }

    /// Device-orbital indices on which this self-energy acts.
    #[must_use]
    pub fn orbitals(&self) -> &[usize] {
        &self.orbitals
    }

    /// Dense contact-local self-energy matrix.
    #[must_use]
    pub const fn matrix(&self) -> &ComplexMatrix {
        &self.matrix
    }
}

/// Matrix-free open-system solver for a sparse finite Hamiltonian.
///
/// Lead self-energies remain dense only within their contact subspaces.
/// Device solves use restarted GMRES and scale with the Hamiltonian nonzero
/// count and the number of requested right-hand sides, never with a dense
/// device inverse.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseOpenSystem {
    hamiltonian: CsrMatrix,
    self_energies: Vec<LocalizedSelfEnergy>,
    broadenings: Vec<ComplexMatrix>,
    retarded_operator: CsrMatrix,
    preconditioner: Ilu0Preconditioner,
    options: GmresOptions,
}

impl SparseOpenSystem {
    /// Validates and constructs a sparse retarded open system.
    pub fn new(
        hamiltonian: CsrMatrix,
        self_energies: Vec<LocalizedSelfEnergy>,
        energy: f64,
        options: GmresOptions,
    ) -> Result<Self, TransportError> {
        let dimension = hamiltonian.rows();
        if dimension == 0 || hamiltonian.columns() != dimension || !energy.is_finite() {
            return Err(TransportError::InvalidDeviceShape);
        }
        if !hamiltonian.is_hermitian(HERMITIAN_TOLERANCE)? {
            return Err(TransportError::NonHermitianDevice);
        }
        if self_energies.iter().any(|contact| {
            contact
                .orbitals()
                .iter()
                .any(|orbital| *orbital >= dimension)
        }) {
            return Err(TransportError::InvalidLocalizedSelfEnergy);
        }
        validate_gmres_options(options)?;
        let broadenings = self_energies
            .iter()
            .map(|contact| broadening_from_self_energy(contact.matrix()))
            .collect::<Result<Vec<_>, _>>()?;
        let retarded_operator = assemble_retarded_operator(&hamiltonian, &self_energies, energy)?;
        let preconditioner = Ilu0Preconditioner::factor(&retarded_operator, 1.0e-12)?;
        Ok(Self {
            hamiltonian,
            self_energies,
            broadenings,
            retarded_operator,
            preconditioner,
            options,
        })
    }

    /// Device Hilbert-space dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.hamiltonian.rows()
    }

    /// Number of stored Hamiltonian entries.
    #[must_use]
    pub fn hamiltonian_nnz(&self) -> usize {
        self.hamiltonian.nnz()
    }

    /// Number of stored entries in the open-system operator and ILU factors.
    #[must_use]
    pub fn solver_nnz(&self) -> usize {
        self.retarded_operator.nnz() + self.preconditioner.nnz()
    }

    /// Contact-local retarded self-energies.
    #[must_use]
    pub fn self_energies(&self) -> &[LocalizedSelfEnergy] {
        &self.self_energies
    }

    /// Contact-local broadening matrices.
    #[must_use]
    pub fn broadenings(&self) -> &[ComplexMatrix] {
        &self.broadenings
    }

    /// Solves `(E - H - Σ) x = b` without materializing a dense device matrix.
    pub fn solve_vector(
        &self,
        right_hand_side: &[Complex64],
    ) -> Result<Vec<Complex64>, TransportError> {
        if right_hand_side.len() != self.dimension() {
            return Err(TransportError::InvalidScatteringFactors);
        }
        gmres_with_right_preconditioner(
            &self.retarded_operator,
            &self.preconditioner,
            right_hand_side,
            None,
            self.options,
        )
        .map(|solution| solution.into_vector())
        .map_err(Into::into)
    }

    /// Solves one right-hand side per input-matrix column.
    pub fn solve_matrix(
        &self,
        right_hand_sides: &ComplexMatrix,
    ) -> Result<ComplexMatrix, TransportError> {
        if right_hand_sides.rows() != self.dimension() {
            return Err(TransportError::InvalidScatteringFactors);
        }
        let column_count = right_hand_sides.columns();
        let mut output = ComplexMatrix::zeros(self.dimension(), column_count);
        for column in 0..column_count {
            let right = (0..self.dimension())
                .map(|row| right_hand_sides.as_slice()[row * column_count + column])
                .collect::<Vec<_>>();
            let solution = self.solve_vector(&right)?;
            for (row, value) in solution.into_iter().enumerate() {
                output.set(row, column, value)?;
            }
        }
        Ok(output)
    }

    /// Returns a selected Green-function block.
    ///
    /// Only columns named by `source_orbitals` are solved, and only rows named
    /// by `drain_orbitals` are returned.
    pub fn green_block(
        &self,
        drain_orbitals: &[usize],
        source_orbitals: &[usize],
    ) -> Result<ComplexMatrix, TransportError> {
        if drain_orbitals
            .iter()
            .chain(source_orbitals)
            .any(|orbital| *orbital >= self.dimension())
        {
            return Err(TransportError::InvalidLocalizedSelfEnergy);
        }
        let mut right = ComplexMatrix::zeros(self.dimension(), source_orbitals.len());
        for (column, orbital) in source_orbitals.iter().enumerate() {
            right.set(*orbital, column, Complex64::new(1.0, 0.0))?;
        }
        let solved = self.solve_matrix(&right)?;
        select_rows(&solved, drain_orbitals)
    }

    /// Returns exact diagonal local density of states values.
    ///
    /// This performs one sparse solve per device orbital. Scattering and
    /// transmission methods below require only contact-channel solves.
    pub fn local_density_of_states(&self) -> Result<Vec<f64>, TransportError> {
        let mut values = Vec::with_capacity(self.dimension());
        let mut right = vec![Complex64::new(0.0, 0.0); self.dimension()];
        for orbital in 0..self.dimension() {
            right[orbital] = Complex64::new(1.0, 0.0);
            let solution = self.solve_vector(&right)?;
            values.push(-solution[orbital].im / std::f64::consts::PI);
            right[orbital] = Complex64::new(0.0, 0.0);
        }
        Ok(values)
    }

    /// Applies the sparse retarded Green operator to injection factors.
    pub fn scattering_states(
        &self,
        injection_factors: &[ComplexMatrix],
    ) -> Result<Vec<ComplexMatrix>, TransportError> {
        injection_factors
            .iter()
            .enumerate()
            .map(|(lead, factor)| {
                if factor.rows() != self.dimension() {
                    return Err(TransportError::InvalidScatteringFactor { lead });
                }
                let states = self.solve_matrix(factor)?;
                let mut transposed = Vec::with_capacity(states.rows() * states.columns());
                for channel in 0..states.columns() {
                    for orbital in 0..states.rows() {
                        transposed.push(states.as_slice()[orbital * states.columns() + channel]);
                    }
                }
                ComplexMatrix::new(states.columns(), states.rows(), transposed).map_err(Into::into)
            })
            .collect()
    }

    /// Constructs a Fisher-Lee scattering matrix using sparse device solves.
    pub fn scattering_matrix(
        &self,
        incoming_factors: &[ComplexMatrix],
        outgoing_factors: &[ComplexMatrix],
        relative_tolerance: f64,
    ) -> Result<ScatteringMatrix, TransportError> {
        let lead_count = self.self_energies.len();
        if incoming_factors.len() != lead_count
            || outgoing_factors.len() != lead_count
            || !relative_tolerance.is_finite()
            || relative_tolerance <= 0.0
        {
            return Err(TransportError::InvalidScatteringFactors);
        }
        let mut offsets = Vec::with_capacity(lead_count + 1);
        offsets.push(0usize);
        for lead in 0..lead_count {
            if incoming_factors[lead].rows() != self.dimension()
                || outgoing_factors[lead].rows() != self.dimension()
                || incoming_factors[lead].columns() != outgoing_factors[lead].columns()
            {
                return Err(TransportError::InvalidScatteringFactor { lead });
            }
            offsets.push(
                offsets[lead]
                    .checked_add(incoming_factors[lead].columns())
                    .ok_or(TransportError::InvalidScatteringFactors)?,
            );
        }
        let solved = incoming_factors
            .iter()
            .map(|factor| self.solve_matrix(factor).map(|value| to_backend(&value)))
            .collect::<Result<Vec<_>, _>>()?;
        let incoming = incoming_factors.iter().map(to_backend).collect::<Vec<_>>();
        let outgoing = outgoing_factors.iter().map(to_backend).collect::<Vec<_>>();
        let channel_count = *offsets.last().expect("offset zero was inserted");
        let mut matrix = DMatrix::<Complex64>::zeros(channel_count, channel_count);
        for drain in 0..lead_count {
            for source in 0..lead_count {
                let mut block = outgoing[drain].adjoint() * &solved[source] * -Complex64::i();
                if drain == source && incoming[source].ncols() > 0 {
                    let scale = frobenius_norm(&outgoing[drain]).max(1.0);
                    let inverse = outgoing[drain]
                        .clone()
                        .pseudo_inverse(relative_tolerance * scale)
                        .map_err(|_| TransportError::PseudoinverseFailure)?;
                    block += inverse * &incoming[source];
                }
                for row in 0..outgoing[drain].ncols() {
                    for column in 0..incoming[source].ncols() {
                        matrix[(offsets[drain] + row, offsets[source] + column)] =
                            block[(row, column)];
                    }
                }
            }
        }
        Ok(ScatteringMatrix {
            matrix: from_backend(&matrix)?,
            lead_offsets: offsets,
        })
    }

    /// Returns channel counts from explicit values or local broadenings.
    pub fn channel_counts(
        &self,
        explicit: &[Option<usize>],
        relative_tolerance: f64,
    ) -> Result<Vec<usize>, TransportError> {
        if explicit.len() != self.broadenings.len()
            || !relative_tolerance.is_finite()
            || relative_tolerance <= 0.0
        {
            return Err(TransportError::InvalidChannelCounts);
        }
        explicit
            .iter()
            .zip(&self.broadenings)
            .map(|(count, broadening)| {
                count.map_or_else(
                    || broadening_channel_count(broadening, relative_tolerance),
                    Ok,
                )
            })
            .collect()
    }

    /// Embeds a square root of one contact broadening in the device basis.
    pub fn broadening_factor(
        &self,
        lead: usize,
        channel_count: usize,
    ) -> Result<ComplexMatrix, TransportError> {
        let broadening = self
            .broadenings
            .get(lead)
            .ok_or(TransportError::UnknownLead { lead })?;
        let local = broadening_factor_from_matrix(broadening, channel_count)?;
        embed_rows(
            self.dimension(),
            self.self_energies[lead].orbitals(),
            &local,
        )
    }

    /// Returns Green-function transmission and reflection values.
    ///
    /// Each source requires one sparse solve per nonzero broadening
    /// eigenchannel.
    pub fn green_function_transmission_matrix(
        &self,
        channel_counts: &[usize],
    ) -> Result<Vec<Vec<f64>>, TransportError> {
        if channel_counts.len() != self.self_energies.len() {
            return Err(TransportError::InvalidChannelCounts);
        }
        let local_factors = self
            .broadenings
            .iter()
            .zip(channel_counts)
            .map(|(broadening, count)| broadening_factor_from_matrix(broadening, *count))
            .collect::<Result<Vec<_>, _>>()?;
        let embedded_factors = local_factors
            .iter()
            .zip(&self.self_energies)
            .map(|(factor, contact)| embed_rows(self.dimension(), contact.orbitals(), factor))
            .collect::<Result<Vec<_>, _>>()?;
        let solved = embedded_factors
            .iter()
            .map(|factor| self.solve_matrix(factor))
            .collect::<Result<Vec<_>, _>>()?;
        (0..self.self_energies.len())
            .map(|drain| {
                (0..self.self_energies.len())
                    .map(|source| {
                        let selected =
                            select_rows(&solved[source], self.self_energies[drain].orbitals())?;
                        let amplitudes =
                            to_backend(&local_factors[drain]).adjoint() * to_backend(&selected);
                        let mut value = amplitudes.iter().map(Complex64::norm_sqr).sum();
                        if drain == source {
                            value += 2.0 * amplitudes.trace().im + channel_counts[source] as f64;
                        }
                        Ok(value)
                    })
                    .collect()
            })
            .collect()
    }
}

fn assemble_retarded_operator(
    hamiltonian: &CsrMatrix,
    self_energies: &[LocalizedSelfEnergy],
    energy: f64,
) -> Result<CsrMatrix, TransportError> {
    let dimension = hamiltonian.rows();
    let mut rows = (0..dimension)
        .map(|row| {
            let mut values = BTreeMap::new();
            values.insert(row, Complex64::new(energy, 0.0));
            values
        })
        .collect::<Vec<_>>();
    for (row, values) in rows.iter_mut().enumerate() {
        for entry in hamiltonian.row_offsets()[row]..hamiltonian.row_offsets()[row + 1] {
            *values
                .entry(hamiltonian.column_indices()[entry])
                .or_default() -= hamiltonian.values()[entry];
        }
    }
    for contact in self_energies {
        let local_dimension = contact.orbitals().len();
        for (local_row, device_row) in contact.orbitals().iter().copied().enumerate() {
            for (local_column, device_column) in contact.orbitals().iter().copied().enumerate() {
                *rows[device_row].entry(device_column).or_default() -=
                    contact.matrix().as_slice()[local_row * local_dimension + local_column];
            }
        }
    }
    let mut row_offsets = Vec::with_capacity(dimension + 1);
    let mut column_indices = Vec::new();
    let mut values = Vec::new();
    row_offsets.push(0);
    for (row, entries) in rows.into_iter().enumerate() {
        for (column, value) in entries {
            if column == row || value != Complex64::new(0.0, 0.0) {
                column_indices.push(column);
                values.push(value);
            }
        }
        row_offsets.push(values.len());
    }
    CsrMatrix::new(dimension, dimension, row_offsets, column_indices, values).map_err(Into::into)
}

fn select_rows(matrix: &ComplexMatrix, rows: &[usize]) -> Result<ComplexMatrix, TransportError> {
    if rows.iter().any(|row| *row >= matrix.rows()) {
        return Err(TransportError::InvalidLocalizedSelfEnergy);
    }
    ComplexMatrix::new(
        rows.len(),
        matrix.columns(),
        rows.iter()
            .flat_map(|row| {
                (0..matrix.columns())
                    .map(move |column| matrix.as_slice()[row * matrix.columns() + column])
            })
            .collect(),
    )
    .map_err(Into::into)
}

fn embed_rows(
    dimension: usize,
    rows: &[usize],
    local: &ComplexMatrix,
) -> Result<ComplexMatrix, TransportError> {
    if local.rows() != rows.len() || rows.iter().any(|row| *row >= dimension) {
        return Err(TransportError::InvalidLocalizedSelfEnergy);
    }
    let mut embedded = ComplexMatrix::zeros(dimension, local.columns());
    for (local_row, device_row) in rows.iter().enumerate() {
        for column in 0..local.columns() {
            embedded.set(
                *device_row,
                column,
                local.as_slice()[local_row * local.columns() + column],
            )?;
        }
    }
    Ok(embedded)
}

fn broadening_from_self_energy(
    self_energy: &ComplexMatrix,
) -> Result<ComplexMatrix, TransportError> {
    let dimension = self_energy.rows();
    if self_energy.columns() != dimension {
        return Err(TransportError::InvalidLocalizedSelfEnergy);
    }
    ComplexMatrix::new(
        dimension,
        dimension,
        (0..dimension)
            .flat_map(|row| {
                (0..dimension).map(move |column| {
                    Complex64::i()
                        * (self_energy.as_slice()[row * dimension + column]
                            - self_energy.as_slice()[column * dimension + row].conj())
                })
            })
            .collect(),
    )
    .map_err(Into::into)
}

fn broadening_channel_count(
    broadening: &ComplexMatrix,
    relative_tolerance: f64,
) -> Result<usize, TransportError> {
    let scale = infinity_norm(broadening).max(1.0);
    let spectrum = hermitian_eigensystem(broadening, HERMITIAN_TOLERANCE)
        .map_err(|_| TransportError::DecompositionFailure)?;
    Ok(spectrum
        .eigenvalues()
        .iter()
        .filter(|value| **value > relative_tolerance * scale)
        .count())
}

fn broadening_factor_from_matrix(
    broadening: &ComplexMatrix,
    channel_count: usize,
) -> Result<ComplexMatrix, TransportError> {
    let dimension = broadening.rows();
    let spectrum = hermitian_eigensystem(broadening, HERMITIAN_TOLERANCE)
        .map_err(|_| TransportError::DecompositionFailure)?;
    let first = dimension.saturating_sub(channel_count.min(dimension));
    let vectors = spectrum.eigenvectors();
    let eigenvalues = spectrum.eigenvalues();
    ComplexMatrix::new(
        dimension,
        dimension - first,
        (0..dimension)
            .flat_map(|row| {
                (first..dimension).map(move |column| {
                    vectors.as_slice()[row * dimension + column]
                        * eigenvalues[column].max(0.0).sqrt()
                })
            })
            .collect(),
    )
    .map_err(Into::into)
}

fn validate_gmres_options(options: GmresOptions) -> Result<(), TransportError> {
    if !options.relative_tolerance.is_finite()
        || options.relative_tolerance <= 0.0
        || !options.absolute_tolerance.is_finite()
        || options.absolute_tolerance < 0.0
        || options.restart == 0
        || options.max_iterations == 0
    {
        Err(TransportError::InvalidOptions)
    } else {
        Ok(())
    }
}

/// Errors raised by steady-state transport calculations.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransportError {
    /// The finite device Hamiltonian is empty or nonsquare.
    InvalidDeviceShape,
    /// The finite device Hamiltonian is not Hermitian.
    NonHermitianDevice,
    /// A lead block or coupling has an incompatible shape.
    InvalidLeadShape,
    /// A principal-cell Hamiltonian is not Hermitian.
    NonHermitianLead,
    /// Numerical controls are not finite and positive.
    InvalidOptions,
    /// A required matrix inverse does not exist.
    SingularGreenFunction,
    /// A Hermitian broadening eigendecomposition failed.
    DecompositionFailure,
    /// Surface decimation did not reach the requested tolerance.
    SurfaceNotConverged,
    /// A lead index is outside the attached-lead list.
    UnknownLead {
        /// Invalid lead index.
        lead: usize,
    },
    /// Injection or extraction factors have incompatible lead dimensions.
    InvalidScatteringFactors,
    /// One lead factor has an incompatible device or channel dimension.
    InvalidScatteringFactor {
        /// Lead containing the invalid factor.
        lead: usize,
    },
    /// A Moore-Penrose inverse required by the Fisher-Lee direct term failed.
    PseudoinverseFailure,
    /// Explicit or inferred lead channel counts are inconsistent.
    InvalidChannelCounts,
    /// A contact-local self-energy has invalid indices or dimensions.
    InvalidLocalizedSelfEnergy,
    /// A sparse linear-operator operation failed.
    LinearOperator(LinearOperatorError),
    /// A sparse iterative solve failed.
    Iterative(IterativeSolveError),
    /// A dense matrix operation failed.
    Matrix(MatrixError),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDeviceShape => {
                write!(formatter, "device Hamiltonian must be nonempty and square")
            }
            Self::NonHermitianDevice => write!(formatter, "device Hamiltonian is not Hermitian"),
            Self::InvalidLeadShape => write!(formatter, "lead matrices have incompatible shapes"),
            Self::NonHermitianLead => write!(formatter, "lead cell Hamiltonian is not Hermitian"),
            Self::InvalidOptions => write!(formatter, "surface Green options are invalid"),
            Self::SingularGreenFunction => write!(formatter, "retarded Green matrix is singular"),
            Self::DecompositionFailure => {
                write!(formatter, "lead broadening eigendecomposition failed")
            }
            Self::SurfaceNotConverged => {
                write!(formatter, "surface Green decimation did not converge")
            }
            Self::UnknownLead { lead } => write!(formatter, "lead index {lead} is out of range"),
            Self::InvalidScatteringFactors => {
                write!(
                    formatter,
                    "scattering factors are inconsistent with attached leads"
                )
            }
            Self::InvalidScatteringFactor { lead } => {
                write!(
                    formatter,
                    "scattering factor for lead {lead} has an invalid shape"
                )
            }
            Self::PseudoinverseFailure => {
                write!(formatter, "scattering-factor pseudoinverse failed")
            }
            Self::InvalidChannelCounts => write!(formatter, "lead channel counts are invalid"),
            Self::InvalidLocalizedSelfEnergy => {
                write!(formatter, "localized self-energy embedding is invalid")
            }
            Self::LinearOperator(error) => error.fmt(formatter),
            Self::Iterative(error) => error.fmt(formatter),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LinearOperator(error) => Some(error),
            Self::Iterative(error) => Some(error),
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatrixError> for TransportError {
    fn from(error: MatrixError) -> Self {
        Self::Matrix(error)
    }
}

impl From<LinearOperatorError> for TransportError {
    fn from(error: LinearOperatorError) -> Self {
        Self::LinearOperator(error)
    }
}

impl From<IterativeSolveError> for TransportError {
    fn from(error: IterativeSolveError) -> Self {
        Self::Iterative(error)
    }
}

/// Zero-temperature partition noise from one reflection-amplitude block.
///
/// The returned dimensionless value is `Tr[R - R²]` with `R = r rᴴ`.
pub fn partition_shot_noise(reflection_amplitudes: &ComplexMatrix) -> Result<f64, TransportError> {
    let channels = reflection_amplitudes.rows();
    let incoming = reflection_amplitudes.columns();
    let mut probabilities = vec![Complex64::new(0.0, 0.0); channels * channels];
    for row in 0..channels {
        for column in 0..channels {
            probabilities[row * channels + column] = (0..incoming)
                .map(|inner| {
                    reflection_amplitudes.as_slice()[row * incoming + inner]
                        * reflection_amplitudes.as_slice()[column * incoming + inner].conj()
                })
                .sum();
        }
    }
    let trace = (0..channels)
        .map(|index| probabilities[index * channels + index])
        .sum::<Complex64>();
    let squared_trace = (0..channels)
        .flat_map(|row| {
            let probabilities = &probabilities;
            (0..channels).map(move |column| {
                probabilities[row * channels + column] * probabilities[column * channels + row]
            })
        })
        .sum::<Complex64>();
    let noise = (trace - squared_trace).re;
    if noise < 0.0 && noise.abs() <= 1.0e-12 {
        Ok(0.0)
    } else {
        Ok(noise)
    }
}

/// Computes the retarded surface Green function by López-Sancho decimation.
pub fn surface_green_function(
    cell_hamiltonian: &ComplexMatrix,
    inter_cell_hopping: &ComplexMatrix,
    energy: f64,
    options: SurfaceGreenOptions,
) -> Result<ComplexMatrix, TransportError> {
    let size = cell_hamiltonian.rows();
    if size == 0 || cell_hamiltonian.columns() != size || inter_cell_hopping.shape() != (size, size)
    {
        return Err(TransportError::InvalidLeadShape);
    }
    if !cell_hamiltonian.is_hermitian(HERMITIAN_TOLERANCE)? {
        return Err(TransportError::NonHermitianLead);
    }
    if !energy.is_finite()
        || !options.broadening.is_finite()
        || options.broadening <= 0.0
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
        || options.max_iterations == 0
    {
        return Err(TransportError::InvalidOptions);
    }

    let z = Complex64::new(energy, options.broadening);
    let identity = DMatrix::<Complex64>::identity(size, size);
    let mut surface_cell = to_backend(cell_hamiltonian);
    let mut bulk_cell = surface_cell.clone();
    let mut forward = to_backend(inter_cell_hopping);
    let mut backward = forward.adjoint();
    let mut converged = false;
    for _ in 0..options.max_iterations {
        let green = (identity.clone() * z - &bulk_cell)
            .try_inverse()
            .ok_or(TransportError::SingularGreenFunction)?;
        let forward_green = &forward * &green;
        let backward_green = &backward * &green;
        let forward_correction = &forward_green * &backward;
        let backward_correction = &backward_green * &forward;
        surface_cell += &forward_correction;
        bulk_cell += &forward_correction + &backward_correction;
        forward = forward_green * forward;
        backward = backward_green * backward;
        if frobenius_norm(&forward).max(frobenius_norm(&backward)) <= options.tolerance {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(TransportError::SurfaceNotConverged);
    }
    let surface = (identity * z - surface_cell)
        .try_inverse()
        .ok_or(TransportError::SingularGreenFunction)?;
    from_backend(&surface)
}

/// Computes the retarded self-energy of a semi-infinite periodic lead.
///
/// `inter_cell_hopping` has shape `(N, M)`: its rows span the lead cell and
/// its columns span the interface on which the returned `(M, M)` self-energy
/// acts. Rectangular and rank-deficient interfaces are supported. The two
/// finite-broadening surface Green functions are linearly extrapolated to
/// zero broadening.
pub fn retarded_lead_self_energy(
    cell_hamiltonian: &ComplexMatrix,
    inter_cell_hopping: &ComplexMatrix,
    energy: f64,
    options: SurfaceGreenOptions,
) -> Result<ComplexMatrix, TransportError> {
    let cell_dimension = cell_hamiltonian.rows();
    let interface_dimension = inter_cell_hopping.columns();
    if cell_dimension == 0
        || cell_hamiltonian.columns() != cell_dimension
        || inter_cell_hopping.rows() != cell_dimension
        || interface_dimension > cell_dimension
    {
        return Err(TransportError::InvalidLeadShape);
    }

    let mut periodic_hopping = DMatrix::<Complex64>::zeros(cell_dimension, cell_dimension);
    for row in 0..cell_dimension {
        for column in 0..interface_dimension {
            periodic_hopping[(row, column)] =
                inter_cell_hopping.as_slice()[row * interface_dimension + column];
        }
    }
    let periodic_hopping = from_backend(&periodic_hopping.adjoint())?;
    let narrow = self_energy_at_broadening(
        cell_hamiltonian,
        &periodic_hopping,
        inter_cell_hopping,
        energy,
        options,
    )?;
    let mut wider_options = options;
    wider_options.broadening *= 2.0;
    let wider = self_energy_at_broadening(
        cell_hamiltonian,
        &periodic_hopping,
        inter_cell_hopping,
        energy,
        wider_options,
    )?;

    ComplexMatrix::new(
        interface_dimension,
        interface_dimension,
        narrow
            .as_slice()
            .iter()
            .zip(wider.as_slice())
            .map(|(narrow, wider)| 2.0 * narrow - wider)
            .collect(),
    )
    .map_err(Into::into)
}

/// Enforces the causal structure of a numerically evaluated retarded
/// self-energy.
///
/// The Hermitian part is preserved and the broadening
/// `i (Σ - Σᴴ)` is projected onto the positive semidefinite cone. When
/// `maximum_rank` is provided, only its largest eigenchannels are retained.
pub fn regularize_retarded_self_energy(
    self_energy: &ComplexMatrix,
    maximum_rank: Option<usize>,
) -> Result<ComplexMatrix, TransportError> {
    let dimension = self_energy.rows();
    if self_energy.columns() != dimension {
        return Err(TransportError::InvalidLeadShape);
    }
    if dimension == 0 {
        return Ok(self_energy.clone());
    }

    let sigma = to_backend(self_energy);
    let adjoint = sigma.adjoint();
    let hermitian_part = (&sigma + &adjoint) * Complex64::new(0.5, 0.0);
    let broadening = (&sigma - &adjoint) * Complex64::new(0.0, 1.0);
    let broadening = from_backend(&broadening)?;
    let decomposition = hermitian_eigensystem(&broadening, 1.0e-9)
        .map_err(|_| TransportError::DecompositionFailure)?;
    let vectors = to_backend(decomposition.eigenvectors());
    let keep_from = maximum_rank
        .map(|rank| dimension.saturating_sub(rank.min(dimension)))
        .unwrap_or(0);
    let mut positive = DMatrix::<Complex64>::zeros(dimension, dimension);
    for column in keep_from..dimension {
        let weight = decomposition.eigenvalues()[column].max(0.0);
        if weight == 0.0 {
            continue;
        }
        for row in 0..dimension {
            for other in 0..dimension {
                positive[(row, other)] +=
                    vectors[(row, column)] * weight * vectors[(other, column)].conj();
            }
        }
    }
    from_backend(&(hermitian_part - positive * Complex64::new(0.0, 0.5)))
}

/// Closed-form retarded self-energy of a nearest-neighbor square-lattice
/// strip with hard-wall transverse boundaries.
pub fn square_lattice_self_energy(
    width: usize,
    hopping: f64,
    fermi_energy: f64,
) -> Result<ComplexMatrix, TransportError> {
    if width == 0 || !hopping.is_finite() || hopping == 0.0 || !fermi_energy.is_finite() {
        return Err(TransportError::InvalidOptions);
    }

    let angle_step = std::f64::consts::PI / (width + 1) as f64;
    let normalization = (2.0 / (width + 1) as f64).sqrt();
    let transverse = (0..width)
        .map(|mode| {
            (0..width)
                .map(|site| {
                    normalization * (angle_step * (mode + 1) as f64 * (site + 1) as f64).sin()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let longitudinal = (0..width)
        .map(|mode| {
            let transverse_energy = 2.0 * hopping * (1.0 - (angle_step * (mode + 1) as f64).cos());
            let q = (fermi_energy - transverse_energy) / hopping - 2.0;
            if q.abs() <= 2.0 {
                Complex64::new(q / 2.0, -(1.0 - (q / 2.0).powi(2)).sqrt())
            } else {
                Complex64::new(q / 2.0 - ((q / 2.0).powi(2) - 1.0).sqrt().copysign(q), 0.0)
            }
        })
        .collect::<Vec<_>>();
    ComplexMatrix::new(
        width,
        width,
        (0..width)
            .flat_map(|row| {
                let transverse = &transverse;
                let longitudinal = &longitudinal;
                (0..width).map(move |column| {
                    hopping
                        * (0..width)
                            .map(|mode| {
                                transverse[mode][row]
                                    * transverse[mode][column]
                                    * longitudinal[mode]
                            })
                            .sum::<Complex64>()
                })
            })
            .collect(),
    )
    .map_err(Into::into)
}

fn self_energy_at_broadening(
    cell_hamiltonian: &ComplexMatrix,
    periodic_hopping: &ComplexMatrix,
    interface_hopping: &ComplexMatrix,
    energy: f64,
    options: SurfaceGreenOptions,
) -> Result<ComplexMatrix, TransportError> {
    let surface = surface_green_function(cell_hamiltonian, periodic_hopping, energy, options)?;
    let coupling = to_backend(interface_hopping).adjoint();
    let self_energy = &coupling * to_backend(&surface) * coupling.adjoint();
    from_backend(&self_energy)
}

fn embedded_contact_self_energy(
    lead: &LeadContact,
    energy: f64,
    options: SurfaceGreenOptions,
) -> Result<ComplexMatrix, TransportError> {
    let surface = surface_green_function(
        lead.cell_hamiltonian(),
        lead.inter_cell_hopping(),
        energy,
        options,
    )?;
    let coupling = to_backend(lead.coupling());
    let self_energy = &coupling * to_backend(&surface) * coupling.adjoint();
    from_backend(&self_energy)
}

/// Computes a contact-local zero-broadening self-energy.
///
/// Unlike [`open_system_extrapolated_self_energies`], the contact coupling
/// need only span the affected device-interface orbitals. The returned matrix
/// therefore scales with contact width rather than device size.
pub fn extrapolated_lead_self_energy(
    lead: &LeadContact,
    energy: f64,
    options: SurfaceGreenOptions,
) -> Result<ComplexMatrix, TransportError> {
    let narrow = embedded_contact_self_energy(lead, energy, options)?;
    let mut wider_options = options;
    wider_options.broadening *= 2.0;
    let wider = embedded_contact_self_energy(lead, energy, wider_options)?;
    ComplexMatrix::new(
        narrow.rows(),
        narrow.columns(),
        narrow
            .as_slice()
            .iter()
            .zip(wider.as_slice())
            .map(|(narrow, wider)| 2.0 * narrow - wider)
            .collect(),
    )
    .map_err(Into::into)
}

/// Computes embedded retarded self-energies without factoring the finite device.
///
/// This is the reusable boundary-condition step of [`solve_open_system`].
/// Callers that provide their own sparse or structured device solver can use
/// these matrices without paying for an unnecessary dense device inverse.
pub fn open_system_self_energies(
    device_hamiltonian: &ComplexMatrix,
    leads: &[LeadContact],
    energy: f64,
    options: SurfaceGreenOptions,
) -> Result<Vec<ComplexMatrix>, TransportError> {
    let device_count = device_hamiltonian.rows();
    if device_count == 0 || device_hamiltonian.columns() != device_count {
        return Err(TransportError::InvalidDeviceShape);
    }
    if !device_hamiltonian.is_hermitian(HERMITIAN_TOLERANCE)? {
        return Err(TransportError::NonHermitianDevice);
    }
    if leads
        .iter()
        .any(|lead| lead.coupling().rows() != device_count)
    {
        return Err(TransportError::InvalidLeadShape);
    }

    let mut embedded_self_energies = Vec::with_capacity(leads.len());
    for lead in leads {
        embedded_self_energies.push(embedded_contact_self_energy(lead, energy, options)?);
    }
    Ok(embedded_self_energies)
}

/// Computes zero-broadening self-energies by linear extrapolation from two
/// positive numerical broadenings.
pub fn open_system_extrapolated_self_energies(
    device_hamiltonian: &ComplexMatrix,
    leads: &[LeadContact],
    energy: f64,
    options: SurfaceGreenOptions,
) -> Result<Vec<ComplexMatrix>, TransportError> {
    let narrow = open_system_self_energies(device_hamiltonian, leads, energy, options)?;
    let mut wider_options = options;
    wider_options.broadening *= 2.0;
    let wider = open_system_self_energies(device_hamiltonian, leads, energy, wider_options)?;
    narrow
        .into_iter()
        .zip(wider)
        .map(|(narrow, wider)| {
            ComplexMatrix::new(
                narrow.rows(),
                narrow.columns(),
                narrow
                    .as_slice()
                    .iter()
                    .zip(wider.as_slice())
                    .map(|(narrow, wider)| 2.0 * narrow - wider)
                    .collect(),
            )
            .map_err(Into::into)
        })
        .collect()
}

/// Solves a finite Hermitian device from arbitrary embedded retarded
/// self-energies.
///
/// This is the narrow waist shared by periodic mode leads, custom
/// self-energy leads, and externally generated embedding methods.
pub fn solve_open_system_from_self_energies(
    device_hamiltonian: &ComplexMatrix,
    embedded_self_energies: &[ComplexMatrix],
    energy: f64,
) -> Result<ScatteringSolution, TransportError> {
    let device_count = device_hamiltonian.rows();
    if device_count == 0 || device_hamiltonian.columns() != device_count || !energy.is_finite() {
        return Err(TransportError::InvalidDeviceShape);
    }
    if !device_hamiltonian.is_hermitian(HERMITIAN_TOLERANCE)? {
        return Err(TransportError::NonHermitianDevice);
    }
    if embedded_self_energies
        .iter()
        .any(|self_energy| self_energy.shape() != (device_count, device_count))
    {
        return Err(TransportError::InvalidLeadShape);
    }

    let mut broadenings = Vec::with_capacity(embedded_self_energies.len());
    let mut inverse_green = DMatrix::<Complex64>::identity(device_count, device_count)
        * Complex64::new(energy, 0.0)
        - to_backend(device_hamiltonian);
    for self_energy in embedded_self_energies {
        let self_energy = to_backend(self_energy);
        inverse_green -= &self_energy;
        let broadening = (self_energy.clone() - self_energy.adjoint()) * Complex64::new(0.0, 1.0);
        broadenings.push(from_backend(&broadening)?);
    }
    let retarded_green = inverse_green
        .try_inverse()
        .ok_or(TransportError::SingularGreenFunction)?;
    Ok(ScatteringSolution {
        retarded_green: from_backend(&retarded_green)?,
        self_energies: embedded_self_energies.to_vec(),
        broadenings,
    })
}

/// Solves the retarded Green function of a finite device with periodic leads.
pub fn solve_open_system(
    device_hamiltonian: &ComplexMatrix,
    leads: &[LeadContact],
    energy: f64,
    options: SurfaceGreenOptions,
) -> Result<ScatteringSolution, TransportError> {
    let embedded_self_energies =
        open_system_self_energies(device_hamiltonian, leads, energy, options)?;
    solve_open_system_from_self_energies(device_hamiltonian, &embedded_self_energies, energy)
}

fn frobenius_norm(matrix: &DMatrix<Complex64>) -> f64 {
    matrix.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

fn infinity_norm(matrix: &ComplexMatrix) -> f64 {
    (0..matrix.rows())
        .map(|row| {
            (0..matrix.columns())
                .map(|column| matrix.as_slice()[row * matrix.columns() + column].norm())
                .sum::<f64>()
        })
        .fold(0.0, f64::max)
}

fn to_backend(matrix: &ComplexMatrix) -> DMatrix<Complex64> {
    DMatrix::from_row_slice(matrix.rows(), matrix.columns(), matrix.as_slice())
}

fn from_backend(matrix: &DMatrix<Complex64>) -> Result<ComplexMatrix, TransportError> {
    ComplexMatrix::new(
        matrix.nrows(),
        matrix.ncols(),
        (0..matrix.nrows())
            .flat_map(|row| (0..matrix.ncols()).map(move |column| matrix[(row, column)]))
            .collect(),
    )
    .map_err(Into::into)
}
