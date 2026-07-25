//! Steady-state coherent transport through finite devices and periodic leads.

use nalgebra::DMatrix;

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
    /// Surface decimation did not reach the requested tolerance.
    SurfaceNotConverged,
    /// A lead index is outside the attached-lead list.
    UnknownLead {
        /// Invalid lead index.
        lead: usize,
    },
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
            Self::SurfaceNotConverged => {
                write!(formatter, "surface Green decimation did not converge")
            }
            Self::UnknownLead { lead } => write!(formatter, "lead index {lead} is out of range"),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
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

/// Solves the retarded Green function of a finite device with periodic leads.
pub fn solve_open_system(
    device_hamiltonian: &ComplexMatrix,
    leads: &[LeadContact],
    energy: f64,
    options: SurfaceGreenOptions,
) -> Result<ScatteringSolution, TransportError> {
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
    let mut broadenings = Vec::with_capacity(leads.len());
    // The lead self-energies provide the retarded boundary condition. The
    // finite device must not receive the surface-decimation broadening again:
    // doing so would introduce artificial absorption proportional to device
    // length and violate current conservation.
    let mut inverse_green = DMatrix::<Complex64>::identity(device_count, device_count)
        * Complex64::new(energy, 0.0)
        - to_backend(device_hamiltonian);
    for lead in leads {
        let surface = surface_green_function(
            lead.cell_hamiltonian(),
            lead.inter_cell_hopping(),
            energy,
            options,
        )?;
        let coupling = to_backend(lead.coupling());
        let self_energy = &coupling * to_backend(&surface) * coupling.adjoint();
        inverse_green -= &self_energy;
        let broadening = (self_energy.clone() - self_energy.adjoint()) * Complex64::new(0.0, 1.0);
        embedded_self_energies.push(from_backend(&self_energy)?);
        broadenings.push(from_backend(&broadening)?);
    }
    let retarded_green = inverse_green
        .try_inverse()
        .ok_or(TransportError::SingularGreenFunction)?;
    Ok(ScatteringSolution {
        retarded_green: from_backend(&retarded_green)?,
        self_energies: embedded_self_energies,
        broadenings,
    })
}

fn frobenius_norm(matrix: &DMatrix<Complex64>) -> f64 {
    matrix.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
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
