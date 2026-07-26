//! Intrinsic band-response quantities from Hamiltonian derivatives.
//!
//! This module exposes the geometric narrow waist shared by linear anomalous
//! Hall and Berry-curvature-dipole calculations. Quadrature weights, unit
//! conversions, charge conventions, and relaxation models remain explicit at
//! the caller boundary.

use nalgebra::DMatrix;

use crate::geometry::UniformReciprocalMesh;
use crate::model::TightBindingModel;
use crate::spectrum::hermitian_eigensystem;
use crate::{Complex64, ComplexMatrix, GeometryError, ModelError, RealMatrix};

const HERMITIAN_TOLERANCE: f64 = 1.0e-8;

/// Fermi-Dirac distribution with temperature expressed in energy units.
///
/// A zero temperature is supported for occupations. The pointwise
/// Fermi-surface derivative is intentionally unavailable at zero temperature,
/// where it is a Dirac distribution rather than a finite function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FermiDistribution {
    chemical_potential: f64,
    temperature: f64,
}

impl FermiDistribution {
    /// Creates a distribution with chemical potential `μ` and `k_B T`.
    pub fn new(chemical_potential: f64, temperature: f64) -> Result<Self, IntrinsicResponseError> {
        if !chemical_potential.is_finite() || !temperature.is_finite() || temperature < 0.0 {
            return Err(IntrinsicResponseError::InvalidFermiDistribution);
        }
        Ok(Self {
            chemical_potential,
            temperature,
        })
    }

    /// Returns the chemical potential.
    #[must_use]
    pub const fn chemical_potential(self) -> f64 {
        self.chemical_potential
    }

    /// Returns the temperature in energy units.
    #[must_use]
    pub const fn temperature(self) -> f64 {
        self.temperature
    }

    /// Returns the Fermi occupation of one finite energy.
    pub fn occupation(self, energy: f64) -> Result<f64, IntrinsicResponseError> {
        if !energy.is_finite() {
            return Err(IntrinsicResponseError::InvalidEnergy);
        }
        if self.temperature == 0.0 {
            return Ok(if energy < self.chemical_potential {
                1.0
            } else if energy > self.chemical_potential {
                0.0
            } else {
                0.5
            });
        }

        let scaled_energy = (energy - self.chemical_potential) / self.temperature;
        Ok(if scaled_energy >= 0.0 {
            let exponential = (-scaled_energy).exp();
            exponential / (1.0 + exponential)
        } else {
            1.0 / (1.0 + scaled_energy.exp())
        })
    }

    /// Returns `-∂f/∂E`, or `None` at zero temperature.
    pub fn negative_energy_derivative(
        self,
        energy: f64,
    ) -> Result<Option<f64>, IntrinsicResponseError> {
        let occupation = self.occupation(energy)?;
        Ok((self.temperature > 0.0).then_some(occupation * (1.0 - occupation) / self.temperature))
    }
}

/// Momentum coordinates used for tight-binding Hamiltonian derivatives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MomentumCoordinates {
    /// Dimensionless reduced reciprocal coordinates.
    Reduced,
    /// Cartesian reciprocal coordinates dual to the model's primitive vectors.
    Cartesian,
}

/// Band-resolved intrinsic response at one momentum.
#[derive(Clone, Debug, PartialEq)]
pub struct BandResponsePoint {
    direction_count: usize,
    energies: Vec<f64>,
    occupations: Vec<f64>,
    negative_occupation_derivatives: Option<Vec<f64>>,
    group_velocities: Vec<f64>,
    berry_curvatures: Vec<f64>,
}

impl BandResponsePoint {
    /// Returns the number of bands.
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.energies.len()
    }

    /// Returns the number of derivative directions.
    #[must_use]
    pub const fn direction_count(&self) -> usize {
        self.direction_count
    }

    /// Returns band energies in ascending order.
    #[must_use]
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// Returns Fermi occupations in band order.
    #[must_use]
    pub fn occupations(&self) -> &[f64] {
        &self.occupations
    }

    /// Returns `-∂f/∂E` in band order, or `None` at zero temperature.
    #[must_use]
    pub fn negative_occupation_derivatives(&self) -> Option<&[f64]> {
        self.negative_occupation_derivatives.as_deref()
    }

    /// Returns `∂E_n/∂k_a`.
    #[must_use]
    pub fn group_velocity(&self, band: usize, direction: usize) -> Option<f64> {
        if band >= self.band_count() || direction >= self.direction_count {
            return None;
        }
        self.group_velocities
            .get(band * self.direction_count + direction)
            .copied()
    }

    /// Returns the antisymmetric Berry-curvature component `Ω_n^{ab}`.
    ///
    /// The convention is `A = -i<u|∂u>` and therefore `Ω^{ab} = 2 Im Q^{ab}`.
    #[must_use]
    pub fn berry_curvature(&self, band: usize, first: usize, second: usize) -> Option<f64> {
        if band >= self.band_count()
            || first >= self.direction_count
            || second >= self.direction_count
        {
            return None;
        }
        self.berry_curvatures
            .get(
                band * self.direction_count * self.direction_count
                    + first * self.direction_count
                    + second,
            )
            .copied()
    }
}

/// Intrinsic band response sampled over one uniform primitive reciprocal cell.
///
/// The mesh and momentum-derivative coordinates remain attached to the
/// samples, so integrations can select the matching reduced or Cartesian
/// quadrature measure without caller-side unit ambiguity.
#[derive(Clone, Debug, PartialEq)]
pub struct UniformMeshBandResponse {
    mesh: UniformReciprocalMesh,
    coordinates: MomentumCoordinates,
    points: Vec<BandResponsePoint>,
}

impl UniformMeshBandResponse {
    /// Evaluates every point of a uniform mesh constructed from `model`.
    pub fn from_model(
        model: &TightBindingModel,
        shape: &[usize],
        fractional_offsets: &[f64],
        fermi: FermiDistribution,
        coordinates: MomentumCoordinates,
        degeneracy_tolerance: f64,
    ) -> Result<Self, IntrinsicResponseError> {
        let mesh = UniformReciprocalMesh::new(model.lattice(), shape, fractional_offsets)?;
        let points = mesh
            .reduced_points()
            .iter()
            .map(|momentum| {
                band_response_from_model(model, momentum, fermi, coordinates, degeneracy_tolerance)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            mesh,
            coordinates,
            points,
        })
    }

    /// Returns the reciprocal mesh and its coordinate measures.
    #[must_use]
    pub const fn mesh(&self) -> &UniformReciprocalMesh {
        &self.mesh
    }

    /// Returns the Hamiltonian-derivative coordinates.
    #[must_use]
    pub const fn coordinates(&self) -> MomentumCoordinates {
        self.coordinates
    }

    /// Returns band-response samples in mesh order.
    #[must_use]
    pub fn points(&self) -> &[BandResponsePoint] {
        &self.points
    }

    /// Integrates one occupation-weighted Berry-curvature component.
    ///
    /// Reduced-coordinate derivatives use the unit reduced-cell measure.
    /// Cartesian derivatives use the primitive reciprocal-cell volume.
    pub fn occupation_weighted_berry_curvature(
        &self,
        first: usize,
        second: usize,
    ) -> Result<f64, IntrinsicResponseError> {
        occupation_weighted_berry_curvature_with_weights(
            &self.points,
            std::iter::repeat(self.quadrature_weight()).take(self.points.len()),
            first,
            second,
        )
    }

    /// Integrates one finite-temperature Berry-curvature-dipole component.
    pub fn berry_curvature_dipole(
        &self,
        derivative_direction: usize,
        curvature_first: usize,
        curvature_second: usize,
    ) -> Result<f64, IntrinsicResponseError> {
        berry_curvature_dipole_with_weights(
            &self.points,
            std::iter::repeat(self.quadrature_weight()).take(self.points.len()),
            derivative_direction,
            curvature_first,
            curvature_second,
        )
    }

    fn quadrature_weight(&self) -> f64 {
        match self.coordinates {
            MomentumCoordinates::Reduced => self.mesh.normalized_weight(),
            MomentumCoordinates::Cartesian => self.mesh.cartesian_weight(),
        }
    }
}

/// Errors raised while evaluating intrinsic band response.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntrinsicResponseError {
    /// The Hamiltonian is not a finite, nonempty Hermitian square matrix.
    InvalidHamiltonian,
    /// Compatible finite Hermitian derivatives were not supplied.
    InvalidDerivatives,
    /// Chemical potential or temperature is invalid.
    InvalidFermiDistribution,
    /// A requested energy is not finite.
    InvalidEnergy,
    /// The absolute band-degeneracy tolerance is invalid.
    InvalidDegeneracyTolerance,
    /// Two bands touch within the requested tolerance.
    DegenerateBands {
        /// First band in ascending energy order.
        first: usize,
        /// Second band in ascending energy order.
        second: usize,
    },
    /// Diagonalization of a validated Hamiltonian failed.
    EigensystemFailure,
    /// Tight-binding model evaluation failed.
    Model(ModelError),
    /// Reciprocal-mesh construction failed.
    Geometry(GeometryError),
    /// Sample points and explicit quadrature weights are inconsistent.
    InvalidSamples,
    /// A requested momentum or curvature direction is unavailable.
    InvalidDirections,
    /// A pointwise Fermi-surface integral was requested at zero temperature.
    ZeroTemperatureFermiSurface,
    /// A finite input produced a non-finite response through numerical overflow.
    NonFiniteResult,
}

impl std::fmt::Display for IntrinsicResponseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHamiltonian => write!(
                formatter,
                "intrinsic response requires a finite, nonempty Hermitian square Hamiltonian"
            ),
            Self::InvalidDerivatives => write!(
                formatter,
                "intrinsic response requires compatible Hermitian Hamiltonian derivatives"
            ),
            Self::InvalidFermiDistribution => write!(
                formatter,
                "chemical potential must be finite and temperature must be finite and nonnegative"
            ),
            Self::InvalidEnergy => write!(formatter, "response energy must be finite"),
            Self::InvalidDegeneracyTolerance => {
                write!(
                    formatter,
                    "band-degeneracy tolerance must be finite and positive"
                )
            }
            Self::DegenerateBands { first, second } => write!(
                formatter,
                "band-resolved Abelian response is undefined for degenerate bands {first} and {second}"
            ),
            Self::EigensystemFailure => {
                write!(formatter, "failed to diagonalize the response Hamiltonian")
            }
            Self::Model(error) => error.fmt(formatter),
            Self::Geometry(error) => error.fmt(formatter),
            Self::InvalidSamples => write!(
                formatter,
                "response samples require one finite explicit quadrature weight per point"
            ),
            Self::InvalidDirections => {
                write!(formatter, "requested response direction is unavailable")
            }
            Self::ZeroTemperatureFermiSurface => write!(
                formatter,
                "pointwise Fermi-surface response requires positive temperature"
            ),
            Self::NonFiniteResult => {
                write!(formatter, "intrinsic-response evaluation overflowed")
            }
        }
    }
}

impl std::error::Error for IntrinsicResponseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Geometry(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModelError> for IntrinsicResponseError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

impl From<GeometryError> for IntrinsicResponseError {
    fn from(error: GeometryError) -> Self {
        Self::Geometry(error)
    }
}

/// Computes band energies, velocities, and Berry curvatures at one momentum.
///
/// For nondegenerate bands this evaluates
/// `Ω_n^{ab} = 2 Im Σ_{m≠n} <n|∂aH|m><m|∂bH|n> / (E_n-E_m)^2`.
/// The result is invariant under constant changes of the input basis. A
/// band-resolved Abelian curvature is deliberately rejected at degeneracies;
/// degenerate manifolds require a non-Abelian observable instead.
pub fn band_response_from_hamiltonian_derivatives(
    hamiltonian: &ComplexMatrix,
    derivatives: &[ComplexMatrix],
    fermi: FermiDistribution,
    degeneracy_tolerance: f64,
) -> Result<BandResponsePoint, IntrinsicResponseError> {
    let (energies, rotated_derivatives) =
        rotated_response_data(hamiltonian, derivatives, degeneracy_tolerance)?;
    let dimension = energies.len();
    for first in 0..dimension {
        for second in (first + 1)..dimension {
            if (energies[first] - energies[second]).abs() <= degeneracy_tolerance {
                return Err(IntrinsicResponseError::DegenerateBands { first, second });
            }
        }
    }

    let direction_count = derivatives.len();

    let mut occupations = Vec::with_capacity(dimension);
    let mut negative_occupation_derivatives =
        (fermi.temperature() > 0.0).then(|| Vec::with_capacity(dimension));
    for &energy in &energies {
        occupations.push(fermi.occupation(energy)?);
        if let Some(derivatives) = &mut negative_occupation_derivatives {
            derivatives.push(
                fermi
                    .negative_energy_derivative(energy)?
                    .expect("positive temperature has a finite Fermi derivative"),
            );
        }
    }

    let mut group_velocities = Vec::with_capacity(dimension * direction_count);
    for band in 0..dimension {
        for derivative in &rotated_derivatives {
            group_velocities.push(derivative[(band, band)].re);
        }
    }

    let mut berry_curvatures = vec![0.0; dimension * direction_count * direction_count];
    for band in 0..dimension {
        for first in 0..direction_count {
            for second in 0..direction_count {
                let curvature = (0..dimension)
                    .filter(|other| *other != band)
                    .map(|other| {
                        let gap = energies[band] - energies[other];
                        (rotated_derivatives[first][(band, other)]
                            * rotated_derivatives[second][(other, band)])
                            .im
                            * 2.0
                            / (gap * gap)
                    })
                    .sum();
                berry_curvatures
                    [band * direction_count * direction_count + first * direction_count + second] =
                    curvature;
            }
        }
    }

    Ok(BandResponsePoint {
        direction_count,
        energies,
        occupations,
        negative_occupation_derivatives,
        group_velocities,
        berry_curvatures,
    })
}

/// Evaluates the intrinsic band response of a tight-binding model.
///
/// `momentum` is always supplied in the model's reduced coordinates.
/// `coordinates` selects the basis of the returned derivative directions.
pub fn band_response_from_model(
    model: &TightBindingModel,
    momentum: &[f64],
    fermi: FermiDistribution,
    coordinates: MomentumCoordinates,
    degeneracy_tolerance: f64,
) -> Result<BandResponsePoint, IntrinsicResponseError> {
    let hamiltonian = model.hamiltonian(momentum)?;
    let derivatives = match coordinates {
        MomentumCoordinates::Reduced => model.reduced_momentum_derivatives(momentum)?,
        MomentumCoordinates::Cartesian => model.cartesian_momentum_derivatives(momentum)?,
    };
    band_response_from_hamiltonian_derivatives(
        &hamiltonian,
        &derivatives,
        fermi,
        degeneracy_tolerance,
    )
}

/// Computes the occupation-weighted intrinsic Berry-curvature tensor.
///
/// Unlike [`band_response_from_hamiltonian_derivatives`], this observable is
/// defined for exact degeneracies. Terms within a degenerate subspace cancel
/// pairwise, while couplings between distinct-energy subspaces are accumulated
/// as traces. The result is therefore invariant under arbitrary unitary basis
/// changes inside each degenerate subspace.
///
/// For nondegenerate bands this is exactly
/// `Σ_n f_n Ω_n^{ab}`. `degeneracy_tolerance` declares which numerically
/// indistinguishable levels belong to the same subspace.
pub fn intrinsic_berry_curvature_from_hamiltonian_derivatives(
    hamiltonian: &ComplexMatrix,
    derivatives: &[ComplexMatrix],
    fermi: FermiDistribution,
    degeneracy_tolerance: f64,
) -> Result<RealMatrix, IntrinsicResponseError> {
    let (energies, rotated_derivatives) =
        rotated_response_data(hamiltonian, derivatives, degeneracy_tolerance)?;
    let occupations = energies
        .iter()
        .map(|&energy| fermi.occupation(energy))
        .collect::<Result<Vec<_>, _>>()?;
    let direction_count = derivatives.len();
    let mut curvature = vec![0.0; direction_count * direction_count];

    for first_band in 0..energies.len() {
        for second_band in (first_band + 1)..energies.len() {
            let gap = energies[first_band] - energies[second_band];
            if gap.abs() <= degeneracy_tolerance {
                continue;
            }
            let occupation_difference = occupations[first_band] - occupations[second_band];
            for first_direction in 0..direction_count {
                for second_direction in 0..direction_count {
                    let matrix_element = rotated_derivatives[first_direction]
                        [(first_band, second_band)]
                        * rotated_derivatives[second_direction][(second_band, first_band)];
                    curvature[first_direction * direction_count + second_direction] +=
                        2.0 * occupation_difference * matrix_element.im / (gap * gap);
                }
            }
        }
    }

    RealMatrix::new(direction_count, direction_count, curvature)
        .map_err(|_| IntrinsicResponseError::NonFiniteResult)
}

/// Evaluates the gauge-invariant intrinsic curvature of a tight-binding model.
///
/// `momentum` is supplied in reduced coordinates; `coordinates` selects the
/// derivative basis of the returned tensor.
pub fn intrinsic_berry_curvature_from_model(
    model: &TightBindingModel,
    momentum: &[f64],
    fermi: FermiDistribution,
    coordinates: MomentumCoordinates,
    degeneracy_tolerance: f64,
) -> Result<RealMatrix, IntrinsicResponseError> {
    let hamiltonian = model.hamiltonian(momentum)?;
    let derivatives = match coordinates {
        MomentumCoordinates::Reduced => model.reduced_momentum_derivatives(momentum)?,
        MomentumCoordinates::Cartesian => model.cartesian_momentum_derivatives(momentum)?,
    };
    intrinsic_berry_curvature_from_hamiltonian_derivatives(
        &hamiltonian,
        &derivatives,
        fermi,
        degeneracy_tolerance,
    )
}

/// Integrates gauge-invariant intrinsic curvature on a uniform reciprocal mesh.
///
/// The returned tensor uses the unit reduced-cell measure for reduced
/// derivatives and the primitive reciprocal-cell volume for Cartesian
/// derivatives. It remains defined when occupied or empty bands are internally
/// degenerate, including spin-degenerate copies.
pub fn uniform_mesh_intrinsic_berry_curvature(
    model: &TightBindingModel,
    shape: &[usize],
    fractional_offsets: &[f64],
    fermi: FermiDistribution,
    coordinates: MomentumCoordinates,
    degeneracy_tolerance: f64,
) -> Result<RealMatrix, IntrinsicResponseError> {
    let mesh = UniformReciprocalMesh::new(model.lattice(), shape, fractional_offsets)?;
    let direction_count = model.lattice().periodic_dimension();
    let mut integrated = vec![0.0; direction_count * direction_count];
    let weight = match coordinates {
        MomentumCoordinates::Reduced => mesh.normalized_weight(),
        MomentumCoordinates::Cartesian => mesh.cartesian_weight(),
    };
    for momentum in mesh.reduced_points() {
        let point = intrinsic_berry_curvature_from_model(
            model,
            momentum,
            fermi,
            coordinates,
            degeneracy_tolerance,
        )?;
        for (total, value) in integrated.iter_mut().zip(point.as_slice()) {
            *total += weight * value;
        }
    }
    RealMatrix::new(direction_count, direction_count, integrated)
        .map_err(|_| IntrinsicResponseError::NonFiniteResult)
}

/// Integrates the occupation-weighted Berry curvature.
///
/// Each weight is used exactly as supplied and must include any Brillouin-zone
/// normalization and cell-volume factor chosen by the caller. Multiplying
/// this geometric integral by a charge and `ℏ` convention is also left to the
/// caller.
pub fn occupation_weighted_berry_curvature(
    points: &[BandResponsePoint],
    weights: &[f64],
    first: usize,
    second: usize,
) -> Result<f64, IntrinsicResponseError> {
    validate_samples(points, weights)?;
    occupation_weighted_berry_curvature_with_weights(points, weights.iter().copied(), first, second)
}

fn occupation_weighted_berry_curvature_with_weights(
    points: &[BandResponsePoint],
    weights: impl Iterator<Item = f64>,
    first: usize,
    second: usize,
) -> Result<f64, IntrinsicResponseError> {
    let mut integral = 0.0;
    for (point, weight) in points.iter().zip(weights) {
        if first >= point.direction_count() || second >= point.direction_count() {
            return Err(IntrinsicResponseError::InvalidDirections);
        }
        integral += weight
            * (0..point.band_count())
                .map(|band| {
                    point.occupations[band]
                        * point
                            .berry_curvature(band, first, second)
                            .expect("directions were validated")
                })
                .sum::<f64>();
    }
    Ok(integral)
}

/// Integrates one component of the Berry-curvature dipole.
///
/// This evaluates the Fermi-surface form
/// `D_{a;bc} = ∫ (-∂f/∂E) (∂E/∂k_a) Ω^{bc}`, equivalent on a periodic
/// Brillouin zone to `∫ f ∂_a Ω^{bc}`. The explicit weights follow the same
/// convention as [`occupation_weighted_berry_curvature`].
pub fn berry_curvature_dipole(
    points: &[BandResponsePoint],
    weights: &[f64],
    derivative_direction: usize,
    curvature_first: usize,
    curvature_second: usize,
) -> Result<f64, IntrinsicResponseError> {
    validate_samples(points, weights)?;
    berry_curvature_dipole_with_weights(
        points,
        weights.iter().copied(),
        derivative_direction,
        curvature_first,
        curvature_second,
    )
}

fn berry_curvature_dipole_with_weights(
    points: &[BandResponsePoint],
    weights: impl Iterator<Item = f64>,
    derivative_direction: usize,
    curvature_first: usize,
    curvature_second: usize,
) -> Result<f64, IntrinsicResponseError> {
    let mut integral = 0.0;
    for (point, weight) in points.iter().zip(weights) {
        if derivative_direction >= point.direction_count()
            || curvature_first >= point.direction_count()
            || curvature_second >= point.direction_count()
        {
            return Err(IntrinsicResponseError::InvalidDirections);
        }
        let negative_derivatives = point
            .negative_occupation_derivatives()
            .ok_or(IntrinsicResponseError::ZeroTemperatureFermiSurface)?;
        integral += weight
            * (0..point.band_count())
                .map(|band| {
                    negative_derivatives[band]
                        * point
                            .group_velocity(band, derivative_direction)
                            .expect("direction was validated")
                        * point
                            .berry_curvature(band, curvature_first, curvature_second)
                            .expect("directions were validated")
                })
                .sum::<f64>();
    }
    Ok(integral)
}

fn validate_samples(
    points: &[BandResponsePoint],
    weights: &[f64],
) -> Result<(), IntrinsicResponseError> {
    if points.is_empty()
        || points.len() != weights.len()
        || weights.iter().any(|weight| !weight.is_finite())
    {
        return Err(IntrinsicResponseError::InvalidSamples);
    }
    Ok(())
}

fn rotated_response_data(
    hamiltonian: &ComplexMatrix,
    derivatives: &[ComplexMatrix],
    degeneracy_tolerance: f64,
) -> Result<(Vec<f64>, Vec<DMatrix<Complex64>>), IntrinsicResponseError> {
    let dimension = hamiltonian.rows();
    if dimension == 0
        || hamiltonian.shape() != (dimension, dimension)
        || !hamiltonian
            .is_hermitian(HERMITIAN_TOLERANCE)
            .unwrap_or(false)
    {
        return Err(IntrinsicResponseError::InvalidHamiltonian);
    }
    if derivatives.is_empty()
        || derivatives.iter().any(|derivative| {
            derivative.shape() != (dimension, dimension)
                || !derivative
                    .is_hermitian(HERMITIAN_TOLERANCE)
                    .unwrap_or(false)
        })
    {
        return Err(IntrinsicResponseError::InvalidDerivatives);
    }
    if !degeneracy_tolerance.is_finite() || degeneracy_tolerance <= 0.0 {
        return Err(IntrinsicResponseError::InvalidDegeneracyTolerance);
    }

    let eigensystem = hermitian_eigensystem(hamiltonian, HERMITIAN_TOLERANCE)
        .map_err(|_| IntrinsicResponseError::EigensystemFailure)?;
    let energies = eigensystem.eigenvalues().to_vec();
    let eigenvectors = dmatrix(eigensystem.eigenvectors());
    let eigenvectors_adjoint = eigenvectors.adjoint();
    let rotated_derivatives = derivatives
        .iter()
        .map(|derivative| &eigenvectors_adjoint * dmatrix(derivative) * &eigenvectors)
        .collect();
    Ok((energies, rotated_derivatives))
}

fn dmatrix(matrix: &ComplexMatrix) -> DMatrix<Complex64> {
    DMatrix::from_row_slice(matrix.rows(), matrix.columns(), matrix.as_slice())
}
