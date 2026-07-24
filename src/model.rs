//! Domain objects and invariants shared by periodic, finite, and open systems.

use std::collections::HashSet;
use std::f64::consts::TAU;

use nalgebra::DMatrix;

use crate::spectrum::{hermitian_eigensystem, Eigensystem};
use crate::{Complex64, ComplexMatrix, ModelError, SpectrumError};

const HERMITIAN_TOLERANCE: f64 = 1.0e-12;
const SINGULAR_TOLERANCE: f64 = 1.0e-12;

/// A complete primitive lattice together with the axes treated as periodic.
///
/// Primitive vectors are Cartesian row vectors. Orbital positions are stored
/// separately in reduced coordinates relative to these vectors. A finite
/// system has an empty `periodic_axes` list. Zero-dimensional models use an
/// empty primitive-vector matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Lattice {
    primitive_vectors: Vec<Vec<f64>>,
    periodic_axes: Vec<usize>,
}

impl Lattice {
    /// Creates a lattice from a square primitive-vector matrix and periodic axes.
    pub fn new(
        primitive_vectors: Vec<Vec<f64>>,
        periodic_axes: Vec<usize>,
    ) -> Result<Self, ModelError> {
        let dimension = primitive_vectors.len();
        for (vector, components) in primitive_vectors.iter().enumerate() {
            if components.len() != dimension {
                return Err(ModelError::InvalidPrimitiveVectors {
                    expected: dimension,
                    vector,
                    actual: components.len(),
                });
            }
            if components.iter().any(|value| !value.is_finite()) {
                return Err(ModelError::NonFiniteValue {
                    field: "primitive vector",
                });
            }
        }
        if dimension > 0 {
            let flattened: Vec<f64> = primitive_vectors
                .iter()
                .flat_map(|vector| vector.iter().copied())
                .collect();
            let matrix = DMatrix::from_row_slice(dimension, dimension, &flattened);
            if matrix.determinant().abs() <= SINGULAR_TOLERANCE {
                return Err(ModelError::SingularLattice);
            }
        }

        let mut seen = HashSet::new();
        for &axis in &periodic_axes {
            if axis >= dimension {
                return Err(ModelError::InvalidPeriodicAxis { axis, dimension });
            }
            if !seen.insert(axis) {
                return Err(ModelError::DuplicatePeriodicAxis { axis });
            }
        }

        Ok(Self {
            primitive_vectors,
            periodic_axes,
        })
    }

    /// Creates a finite model with an identity coordinate frame.
    pub fn finite(dimension: usize) -> Result<Self, ModelError> {
        let mut vectors = vec![vec![0.0; dimension]; dimension];
        for (index, vector) in vectors.iter_mut().enumerate() {
            vector[index] = 1.0;
        }
        Self::new(vectors, Vec::new())
    }

    /// Returns the real-space dimension.
    #[must_use]
    pub fn real_dimension(&self) -> usize {
        self.primitive_vectors.len()
    }

    /// Returns the number of periodic axes.
    #[must_use]
    pub fn periodic_dimension(&self) -> usize {
        self.periodic_axes.len()
    }

    /// Returns the primitive vectors as Cartesian row vectors.
    #[must_use]
    pub fn primitive_vectors(&self) -> &[Vec<f64>] {
        &self.primitive_vectors
    }

    /// Returns the real-space axes represented by reduced momentum components.
    #[must_use]
    pub fn periodic_axes(&self) -> &[usize] {
        &self.periodic_axes
    }
}

/// Stable identifier for a localized orbital subspace within one model.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct OrbitalId(usize);

impl OrbitalId {
    /// Returns the zero-based orbital index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// A localized orbital subspace and its reduced position.
#[derive(Clone, Debug, PartialEq)]
pub struct Orbital {
    label: String,
    reduced_position: Vec<f64>,
    degrees_of_freedom: usize,
}

impl Orbital {
    /// Returns the user-visible orbital label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the reduced position in the primitive-vector basis.
    #[must_use]
    pub fn reduced_position(&self) -> &[f64] {
        &self.reduced_position
    }

    /// Returns the number of internal basis states.
    #[must_use]
    pub const fn degrees_of_freedom(&self) -> usize {
        self.degrees_of_freedom
    }
}

/// A directed hopping block whose Hermitian conjugate is implicit.
///
/// `amplitude` maps the source subspace into the target subspace. Its shape is
/// `(target degrees of freedom, source degrees of freedom)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Hopping {
    target: OrbitalId,
    source: OrbitalId,
    cell_offset: Vec<i32>,
    amplitude: ComplexMatrix,
}

impl Hopping {
    /// Returns the target orbital.
    #[must_use]
    pub const fn target(&self) -> OrbitalId {
        self.target
    }

    /// Returns the source orbital.
    #[must_use]
    pub const fn source(&self) -> OrbitalId {
        self.source
    }

    /// Returns the integer displacement in full primitive-cell coordinates.
    #[must_use]
    pub fn cell_offset(&self) -> &[i32] {
        &self.cell_offset
    }

    /// Returns the hopping block.
    #[must_use]
    pub const fn amplitude(&self) -> &ComplexMatrix {
        &self.amplitude
    }
}

/// Builder for a tight-binding Hamiltonian.
#[derive(Clone, Debug)]
pub struct ModelBuilder {
    lattice: Lattice,
    orbitals: Vec<Orbital>,
    onsite: Vec<ComplexMatrix>,
    hoppings: Vec<Hopping>,
}

impl ModelBuilder {
    /// Starts a model in the supplied lattice.
    #[must_use]
    pub fn new(lattice: Lattice) -> Self {
        Self {
            lattice,
            orbitals: Vec::new(),
            onsite: Vec::new(),
            hoppings: Vec::new(),
        }
    }

    /// Adds a scalar orbital and returns its stable model-local identifier.
    pub fn add_orbital(
        &mut self,
        label: impl Into<String>,
        reduced_position: impl IntoIterator<Item = f64>,
    ) -> Result<OrbitalId, ModelError> {
        self.add_orbital_with_dof(label, reduced_position, 1)
    }

    /// Adds a localized subspace with an explicit number of internal states.
    pub fn add_orbital_with_dof(
        &mut self,
        label: impl Into<String>,
        reduced_position: impl IntoIterator<Item = f64>,
        degrees_of_freedom: usize,
    ) -> Result<OrbitalId, ModelError> {
        let label = label.into();
        if label.is_empty() {
            return Err(ModelError::EmptyOrbitalLabel);
        }
        if self.orbitals.iter().any(|orbital| orbital.label == label) {
            return Err(ModelError::DuplicateOrbitalLabel { label });
        }
        if degrees_of_freedom == 0 {
            return Err(ModelError::InvalidDegreesOfFreedom);
        }

        let reduced_position: Vec<f64> = reduced_position.into_iter().collect();
        if reduced_position.len() != self.lattice.real_dimension() {
            return Err(ModelError::InvalidOrbitalPosition {
                expected: self.lattice.real_dimension(),
                actual: reduced_position.len(),
            });
        }
        if reduced_position.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::NonFiniteValue {
                field: "orbital position",
            });
        }

        let id = OrbitalId(self.orbitals.len());
        self.orbitals.push(Orbital {
            label,
            reduced_position,
            degrees_of_freedom,
        });
        self.onsite
            .push(ComplexMatrix::zeros(degrees_of_freedom, degrees_of_freedom));
        Ok(id)
    }

    /// Sets a real scalar onsite energy for a scalar orbital.
    pub fn set_onsite(&mut self, orbital: OrbitalId, energy: f64) -> Result<(), ModelError> {
        if !energy.is_finite() {
            return Err(ModelError::NonFiniteValue {
                field: "onsite energy",
            });
        }
        self.set_onsite_block(orbital, ComplexMatrix::scalar(Complex64::new(energy, 0.0)))
    }

    /// Sets a Hermitian onsite block.
    pub fn set_onsite_block(
        &mut self,
        orbital: OrbitalId,
        block: ComplexMatrix,
    ) -> Result<(), ModelError> {
        let degrees = self.orbital(orbital)?.degrees_of_freedom;
        validate_shape(&block, degrees, degrees)?;
        if !block.is_hermitian(HERMITIAN_TOLERANCE)? {
            return Err(ModelError::NonHermitianOnsite);
        }
        self.onsite[orbital.index()] = block;
        Ok(())
    }

    /// Adds a scalar hopping between scalar orbitals.
    pub fn add_hopping(
        &mut self,
        target: OrbitalId,
        source: OrbitalId,
        cell_offset: impl IntoIterator<Item = i32>,
        amplitude: Complex64,
    ) -> Result<(), ModelError> {
        self.add_hopping_block(
            target,
            source,
            cell_offset,
            ComplexMatrix::scalar(amplitude),
        )
    }

    /// Adds a hopping block; its Hermitian partner is generated implicitly.
    pub fn add_hopping_block(
        &mut self,
        target: OrbitalId,
        source: OrbitalId,
        cell_offset: impl IntoIterator<Item = i32>,
        amplitude: ComplexMatrix,
    ) -> Result<(), ModelError> {
        let target_degrees = self.orbital(target)?.degrees_of_freedom;
        let source_degrees = self.orbital(source)?.degrees_of_freedom;
        validate_shape(&amplitude, target_degrees, source_degrees)?;

        let cell_offset: Vec<i32> = cell_offset.into_iter().collect();
        if cell_offset.len() != self.lattice.real_dimension() {
            return Err(ModelError::InvalidCellOffset {
                expected: self.lattice.real_dimension(),
                actual: cell_offset.len(),
            });
        }
        if target == source && cell_offset.iter().all(|component| *component == 0) {
            return Err(ModelError::SelfHoppingAtHome);
        }
        if self
            .hoppings
            .iter()
            .any(|term| is_same_or_hermitian_partner(term, target, source, &cell_offset))
        {
            return Err(ModelError::DuplicateHopping);
        }

        self.hoppings.push(Hopping {
            target,
            source,
            cell_offset,
            amplitude,
        });
        Ok(())
    }

    /// Adds a hopping block, summing it into an existing term or its partner.
    ///
    /// Exact structural transformations can map several source-cell terms onto
    /// one directed hopping. This operation preserves that sum while retaining
    /// the builder's single-term canonical representation.
    pub fn add_hopping_block_sum(
        &mut self,
        target: OrbitalId,
        source: OrbitalId,
        cell_offset: impl IntoIterator<Item = i32>,
        amplitude: ComplexMatrix,
    ) -> Result<(), ModelError> {
        let target_degrees = self.orbital(target)?.degrees_of_freedom;
        let source_degrees = self.orbital(source)?.degrees_of_freedom;
        validate_shape(&amplitude, target_degrees, source_degrees)?;
        let cell_offset: Vec<i32> = cell_offset.into_iter().collect();
        if cell_offset.len() != self.lattice.real_dimension() {
            return Err(ModelError::InvalidCellOffset {
                expected: self.lattice.real_dimension(),
                actual: cell_offset.len(),
            });
        }
        if target == source && cell_offset.iter().all(|component| *component == 0) {
            return Err(ModelError::SelfHoppingAtHome);
        }

        for existing in &mut self.hoppings {
            let same = existing.target == target
                && existing.source == source
                && existing.cell_offset == cell_offset;
            let partner = existing.target == source
                && existing.source == target
                && existing
                    .cell_offset
                    .iter()
                    .zip(&cell_offset)
                    .all(|(left, right)| *left == -*right);
            if !same && !partner {
                continue;
            }
            let contribution = if partner {
                amplitude.adjoint()
            } else {
                amplitude
            };
            for row in 0..existing.amplitude.rows() {
                for column in 0..existing.amplitude.columns() {
                    existing
                        .amplitude
                        .add_entry(row, column, contribution.get(row, column)?)?;
                }
            }
            return Ok(());
        }

        self.hoppings.push(Hopping {
            target,
            source,
            cell_offset,
            amplitude,
        });
        Ok(())
    }

    /// Finalizes structural validation and returns an immutable model.
    pub fn build(self) -> Result<TightBindingModel, ModelError> {
        if self.orbitals.is_empty() {
            return Err(ModelError::EmptyModel);
        }
        let mut basis_offsets = Vec::with_capacity(self.orbitals.len() + 1);
        basis_offsets.push(0);
        for orbital in &self.orbitals {
            basis_offsets.push(
                basis_offsets.last().copied().unwrap_or_default() + orbital.degrees_of_freedom,
            );
        }
        Ok(TightBindingModel {
            lattice: self.lattice,
            orbitals: self.orbitals,
            onsite: self.onsite,
            hoppings: self.hoppings,
            basis_offsets,
        })
    }

    fn orbital(&self, orbital: OrbitalId) -> Result<&Orbital, ModelError> {
        self.orbitals
            .get(orbital.index())
            .ok_or(ModelError::UnknownOrbital {
                index: orbital.index(),
            })
    }
}

/// Immutable tight-binding model shared by periodic and finite workflows.
#[derive(Clone, Debug, PartialEq)]
pub struct TightBindingModel {
    lattice: Lattice,
    orbitals: Vec<Orbital>,
    onsite: Vec<ComplexMatrix>,
    hoppings: Vec<Hopping>,
    basis_offsets: Vec<usize>,
}

impl TightBindingModel {
    /// Returns the complete lattice and periodic-axis selection.
    #[must_use]
    pub const fn lattice(&self) -> &Lattice {
        &self.lattice
    }

    /// Returns all localized subspaces in stable identifier order.
    #[must_use]
    pub fn orbitals(&self) -> &[Orbital] {
        &self.orbitals
    }

    /// Returns one onsite block.
    #[must_use]
    pub fn onsite(&self, orbital: OrbitalId) -> Option<&ComplexMatrix> {
        self.onsite.get(orbital.index())
    }

    /// Returns all onsite blocks in orbital order.
    #[must_use]
    pub fn onsite_blocks(&self) -> &[ComplexMatrix] {
        &self.onsite
    }

    /// Returns all explicit hopping blocks.
    #[must_use]
    pub fn hoppings(&self) -> &[Hopping] {
        &self.hoppings
    }

    /// Returns the total Hamiltonian dimension.
    #[must_use]
    pub fn state_count(&self) -> usize {
        self.basis_offsets.last().copied().unwrap_or_default()
    }

    /// Assembles the Bloch or finite Hamiltonian.
    ///
    /// Momentum components are reduced coordinates ordered by
    /// [`Lattice::periodic_axes`].
    pub fn hamiltonian(&self, momentum: &[f64]) -> Result<ComplexMatrix, ModelError> {
        self.validate_momentum(momentum)?;

        let mut hamiltonian = ComplexMatrix::zeros(self.state_count(), self.state_count());
        for (orbital, block) in self.onsite.iter().enumerate() {
            let start = self.basis_offsets[orbital];
            add_block(
                &mut hamiltonian,
                start,
                start,
                block,
                Complex64::new(1.0, 0.0),
            )?;
        }

        for hopping in &self.hoppings {
            let phase_argument = self
                .lattice
                .periodic_axes
                .iter()
                .zip(momentum)
                .map(|(axis, reduced_momentum)| {
                    let displacement = f64::from(hopping.cell_offset[*axis])
                        - self.orbitals[hopping.target.index()].reduced_position[*axis]
                        + self.orbitals[hopping.source.index()].reduced_position[*axis];
                    reduced_momentum * displacement
                })
                .sum::<f64>();
            let phase = Complex64::from_polar(1.0, TAU * phase_argument);

            let target_start = self.basis_offsets[hopping.target.index()];
            let source_start = self.basis_offsets[hopping.source.index()];
            add_block(
                &mut hamiltonian,
                target_start,
                source_start,
                &hopping.amplitude,
                phase,
            )?;
            add_block(
                &mut hamiltonian,
                source_start,
                target_start,
                &hopping.amplitude.adjoint(),
                phase.conj(),
            )?;
        }
        enforce_exact_hermiticity(&mut hamiltonian)?;
        Ok(hamiltonian)
    }

    /// Returns derivatives with respect to reduced momentum components.
    pub fn reduced_momentum_derivatives(
        &self,
        momentum: &[f64],
    ) -> Result<Vec<ComplexMatrix>, ModelError> {
        self.momentum_derivatives(momentum, false)
    }

    /// Returns derivatives with respect to Cartesian momentum components.
    pub fn cartesian_momentum_derivatives(
        &self,
        momentum: &[f64],
    ) -> Result<Vec<ComplexMatrix>, ModelError> {
        self.momentum_derivatives(momentum, true)
    }

    /// Diagonalizes the model at one reduced momentum.
    pub fn eigensystem(&self, momentum: &[f64]) -> Result<Eigensystem, ModelSolveError> {
        let hamiltonian = self.hamiltonian(momentum)?;
        Ok(hermitian_eigensystem(&hamiltonian, HERMITIAN_TOLERANCE)?)
    }

    /// Computes eigensystems at several reduced momenta.
    pub fn band_structure(
        &self,
        momenta: &[Vec<f64>],
    ) -> Result<Vec<Eigensystem>, ModelSolveError> {
        momenta
            .iter()
            .map(|momentum| self.eigensystem(momentum))
            .collect()
    }

    fn validate_momentum(&self, momentum: &[f64]) -> Result<(), ModelError> {
        if momentum.len() != self.lattice.periodic_dimension() {
            return Err(ModelError::InvalidMomentum {
                expected: self.lattice.periodic_dimension(),
                actual: momentum.len(),
            });
        }
        if momentum.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::NonFiniteValue { field: "momentum" });
        }
        Ok(())
    }

    fn momentum_derivatives(
        &self,
        momentum: &[f64],
        cartesian: bool,
    ) -> Result<Vec<ComplexMatrix>, ModelError> {
        self.validate_momentum(momentum)?;
        let component_count = if cartesian {
            self.lattice.real_dimension()
        } else {
            self.lattice.periodic_dimension()
        };
        let mut derivatives =
            vec![ComplexMatrix::zeros(self.state_count(), self.state_count()); component_count];

        for hopping in &self.hoppings {
            let periodic_displacements: Vec<f64> = self
                .lattice
                .periodic_axes
                .iter()
                .map(|axis| {
                    f64::from(hopping.cell_offset[*axis])
                        - self.orbitals[hopping.target.index()].reduced_position[*axis]
                        + self.orbitals[hopping.source.index()].reduced_position[*axis]
                })
                .collect();
            let phase_argument = momentum
                .iter()
                .zip(&periodic_displacements)
                .map(|(reduced_momentum, displacement)| reduced_momentum * displacement)
                .sum::<f64>();
            let phase = Complex64::from_polar(1.0, TAU * phase_argument);
            let coefficients: Vec<f64> = if cartesian {
                (0..component_count)
                    .map(|component| {
                        self.lattice
                            .periodic_axes
                            .iter()
                            .zip(&periodic_displacements)
                            .map(|(axis, displacement)| {
                                displacement * self.lattice.primitive_vectors[*axis][component]
                            })
                            .sum()
                    })
                    .collect()
            } else {
                periodic_displacements
                    .iter()
                    .map(|displacement| TAU * displacement)
                    .collect()
            };

            let target_start = self.basis_offsets[hopping.target.index()];
            let source_start = self.basis_offsets[hopping.source.index()];
            for (derivative, coefficient) in derivatives.iter_mut().zip(coefficients) {
                let factor = Complex64::new(0.0, coefficient) * phase;
                add_block(
                    derivative,
                    target_start,
                    source_start,
                    &hopping.amplitude,
                    factor,
                )?;
                add_block(
                    derivative,
                    source_start,
                    target_start,
                    &hopping.amplitude.adjoint(),
                    factor.conj(),
                )?;
            }
        }
        for derivative in &mut derivatives {
            enforce_exact_hermiticity(derivative)?;
        }
        Ok(derivatives)
    }
}

fn enforce_exact_hermiticity(matrix: &mut ComplexMatrix) -> Result<(), ModelError> {
    for index in 0..matrix.rows() {
        let diagonal = matrix.get(index, index)?;
        matrix.set(index, index, Complex64::new(diagonal.re, 0.0))?;
        for column in 0..index {
            let upper = matrix.get(column, index)?;
            matrix.set(index, column, upper.conj())?;
        }
    }
    Ok(())
}

/// Errors raised while assembling or diagonalizing a model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelSolveError {
    /// Model assembly failed.
    Model(ModelError),
    /// Hermitian diagonalization failed.
    Spectrum(SpectrumError),
}

impl std::fmt::Display for ModelSolveError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(error) => error.fmt(formatter),
            Self::Spectrum(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ModelSolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Spectrum(error) => Some(error),
        }
    }
}

impl From<ModelError> for ModelSolveError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

impl From<SpectrumError> for ModelSolveError {
    fn from(error: SpectrumError) -> Self {
        Self::Spectrum(error)
    }
}

fn validate_shape(
    block: &ComplexMatrix,
    expected_rows: usize,
    expected_columns: usize,
) -> Result<(), ModelError> {
    if block.shape() != (expected_rows, expected_columns) {
        return Err(ModelError::InvalidBlockShape {
            expected_rows,
            expected_columns,
            actual_rows: block.rows(),
            actual_columns: block.columns(),
        });
    }
    Ok(())
}

fn add_block(
    destination: &mut ComplexMatrix,
    row_start: usize,
    column_start: usize,
    block: &ComplexMatrix,
    factor: Complex64,
) -> Result<(), ModelError> {
    for row in 0..block.rows() {
        for column in 0..block.columns() {
            destination.add_entry(
                row_start + row,
                column_start + column,
                factor * block.get(row, column)?,
            )?;
        }
    }
    Ok(())
}

fn is_same_or_hermitian_partner(
    existing: &Hopping,
    target: OrbitalId,
    source: OrbitalId,
    cell_offset: &[i32],
) -> bool {
    let same = existing.target == target
        && existing.source == source
        && existing.cell_offset == cell_offset;
    let partner = existing.target == source
        && existing.source == target
        && existing.cell_offset.len() == cell_offset.len()
        && existing
            .cell_offset
            .iter()
            .zip(cell_offset)
            .all(|(left, right)| i64::from(*left) == -i64::from(*right));
    same || partner
}
