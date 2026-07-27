//! Rust-native automatic-differentiation rules for scientific workflows.
//!
//! This module uses [`chainrules_core`] as a narrow first-order protocol.  It
//! owns Thouless-specific parameter spaces, constrained matrix semantics,
//! solver rules, diagnostics, and explicit workflow composition.  Finite
//! differences remain validation oracles and are never used to produce a
//! gradient returned by this module.

use std::error::Error;
use std::fmt;

pub use chainrules_core::{Differentiable, JvpRule, NoTangent, Pullback, VjpRule};
use nalgebra::{DMatrix, DVector};

use crate::linear_operator::{
    gmres, AdjointLinearOperator, CsrMatrix, GmresOptions, IterativeSolveError, LinearOperator,
    LinearOperatorError,
};
use crate::spectrum::hermitian_eigensystem;
use crate::transport::{
    solve_open_system_from_self_energies, surface_green_function, SurfaceGreenOptions,
    TransportError,
};
use crate::{Complex64, ComplexMatrix, MatrixError, SpectrumError};

const HERMITIAN_TOLERANCE: f64 = 1.0e-10;

/// Errors raised while constructing or applying a native derivative rule.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum AdError {
    /// A parameter, direction, or gradient has an unexpected number of
    /// coordinates.
    ParameterCount {
        /// Required coordinate count.
        expected: usize,
        /// Supplied coordinate count.
        actual: usize,
    },
    /// A parameter-space value contains NaN or infinity.
    NonFiniteParameter {
        /// Index of the invalid coordinate.
        index: usize,
    },
    /// A matrix or vector has an incompatible shape.
    Shape {
        /// Description of the expected shape.
        expected: String,
        /// Description of the supplied shape.
        actual: String,
    },
    /// A matrix advertised as Hermitian violates that invariant.
    NonHermitian,
    /// A projector does not satisfy the documented Hermitian/idempotent
    /// contract.
    InvalidProjector,
    /// A requested eigenvalue index or occupied-subspace rank is invalid.
    InvalidSubspace,
    /// The separating spectral gap is too small for the requested derivative.
    GapTooSmall {
        /// Observed separating gap.
        gap: f64,
        /// Minimum accepted gap.
        minimum: f64,
    },
    /// A primal dense solve is singular.
    SingularPrimal,
    /// An adjoint dense solve is singular.
    SingularAdjoint,
    /// A derivative produced NaN or infinity.
    NonFiniteDerivative,
    /// Dense matrix validation or construction failed.
    Matrix(MatrixError),
    /// Matrix-free operator validation or application failed.
    LinearOperator(LinearOperatorError),
    /// A sparse primal iterative solve failed.
    PrimalIterative(IterativeSolveError),
    /// A sparse adjoint iterative solve failed.
    AdjointIterative(IterativeSolveError),
    /// Hermitian spectral decomposition failed.
    Spectrum(SpectrumError),
    /// Open-system transport evaluation failed.
    Transport(TransportError),
}

impl fmt::Display for AdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParameterCount { expected, actual } => {
                write!(
                    formatter,
                    "parameter space has {actual} coordinates; expected {expected}"
                )
            }
            Self::NonFiniteParameter { index } => {
                write!(formatter, "parameter coordinate {index} is not finite")
            }
            Self::Shape { expected, actual } => {
                write!(formatter, "shape {actual} does not match {expected}")
            }
            Self::NonHermitian => write!(formatter, "matrix is not Hermitian"),
            Self::InvalidProjector => {
                write!(formatter, "target is not a Hermitian idempotent projector")
            }
            Self::InvalidSubspace => write!(formatter, "requested spectral subspace is invalid"),
            Self::GapTooSmall { gap, minimum } => write!(
                formatter,
                "separating spectral gap {gap:e} is below the required {minimum:e}"
            ),
            Self::SingularPrimal => write!(formatter, "primal linear system is singular"),
            Self::SingularAdjoint => write!(formatter, "adjoint linear system is singular"),
            Self::NonFiniteDerivative => write!(formatter, "derivative is not finite"),
            Self::Matrix(error) => error.fmt(formatter),
            Self::LinearOperator(error) => error.fmt(formatter),
            Self::PrimalIterative(error) => {
                write!(formatter, "primal iterative solve failed: {error}")
            }
            Self::AdjointIterative(error) => {
                write!(formatter, "adjoint iterative solve failed: {error}")
            }
            Self::Spectrum(error) => error.fmt(formatter),
            Self::Transport(error) => error.fmt(formatter),
        }
    }
}

impl Error for AdError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            Self::LinearOperator(error) => Some(error),
            Self::PrimalIterative(error) | Self::AdjointIterative(error) => Some(error),
            Self::Spectrum(error) => Some(error),
            Self::Transport(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatrixError> for AdError {
    fn from(error: MatrixError) -> Self {
        Self::Matrix(error)
    }
}

impl From<LinearOperatorError> for AdError {
    fn from(error: LinearOperatorError) -> Self {
        Self::LinearOperator(error)
    }
}

impl From<SpectrumError> for AdError {
    fn from(error: SpectrumError) -> Self {
        Self::Spectrum(error)
    }
}

impl From<TransportError> for AdError {
    fn from(error: TransportError) -> Self {
        Self::Transport(error)
    }
}

/// Independent continuous coordinates of one physical model family.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelParameters {
    values: Vec<f64>,
}

impl ModelParameters {
    /// Creates a parameter vector after validating every coordinate.
    pub fn new(values: Vec<f64>) -> Result<Self, AdError> {
        validate_real_coordinates(&values)?;
        Ok(Self { values })
    }

    /// Returns the independent physical coordinates.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Number of independent coordinates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the parameter vector has no continuous coordinates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns a displaced parameter vector `self + scale * direction`.
    pub fn displaced(&self, direction: &ModelDirection, scale: f64) -> Result<Self, AdError> {
        validate_count(self.len(), direction.len())?;
        if !scale.is_finite() {
            return Err(AdError::NonFiniteParameter { index: 0 });
        }
        Self::new(
            self.values
                .iter()
                .zip(direction.as_slice())
                .map(|(value, tangent)| value + scale * tangent)
                .collect(),
        )
    }
}

/// Forward perturbation of [`ModelParameters`].
#[derive(Clone, Debug, PartialEq)]
pub struct ModelDirection {
    values: Vec<f64>,
}

impl ModelDirection {
    /// Creates a finite parameter direction.
    pub fn new(values: Vec<f64>) -> Result<Self, AdError> {
        validate_real_coordinates(&values)?;
        Ok(Self { values })
    }

    /// Returns the direction coordinates.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Number of direction coordinates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the direction has no coordinates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Reverse sensitivity in the physical coordinate space.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelGradient {
    values: Vec<f64>,
}

impl ModelGradient {
    /// Creates a finite physical gradient.
    pub fn new(values: Vec<f64>) -> Result<Self, AdError> {
        validate_real_coordinates(&values)?;
        Ok(Self { values })
    }

    /// Creates an exact zero in a known physical cotangent space.
    #[must_use]
    pub fn zeros(parameter_count: usize) -> Self {
        Self {
            values: vec![0.0; parameter_count],
        }
    }

    /// Returns the gradient coordinates.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Number of gradient coordinates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the gradient has no coordinates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Euclidean norm in the declared physical coordinates.
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.values
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    }

    /// Adds another contribution in place.
    pub fn accumulate(&mut self, contribution: &Self) -> Result<(), AdError> {
        validate_count(self.len(), contribution.len())?;
        for (target, value) in self.values.iter_mut().zip(&contribution.values) {
            *target += value;
        }
        validate_real_coordinates(&self.values)
    }

    /// Returns a scaled gradient.
    pub fn scaled(&self, factor: f64) -> Result<Self, AdError> {
        if !factor.is_finite() {
            return Err(AdError::NonFiniteDerivative);
        }
        Self::new(self.values.iter().map(|value| factor * value).collect())
    }
}

impl Differentiable for ModelParameters {
    type Tangent = ModelDirection;
    type Cotangent = ModelGradient;
}

impl Differentiable for ComplexMatrix {
    type Tangent = ComplexMatrix;
    type Cotangent = ComplexMatrix;
}

/// Real Frobenius pairing `Re tr(left† right)`.
pub fn real_frobenius_pairing(left: &ComplexMatrix, right: &ComplexMatrix) -> Result<f64, AdError> {
    validate_matrix_shape(left.shape(), right.shape())?;
    let value = left
        .as_slice()
        .iter()
        .zip(right.as_slice())
        .map(|(left, right)| (left.conj() * right).re)
        .sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AdError::NonFiniteDerivative)
    }
}

/// An affine Hermitian matrix family `H(theta) = H0 + Σ theta_i H_i`.
///
/// The matrices are owned and validated once.  This provides a general
/// operation-level boundary for Hamiltonians, device matrices, and local
/// parameterizations without exposing an ambient dense Jacobian.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineHermitianFamily {
    base: ComplexMatrix,
    directions: Vec<ComplexMatrix>,
}

impl AffineHermitianFamily {
    /// Creates a family after validating shape and Hermiticity of every
    /// physical direction.
    pub fn new(base: ComplexMatrix, directions: Vec<ComplexMatrix>) -> Result<Self, AdError> {
        if base.rows() == 0 || base.rows() != base.columns() {
            return Err(AdError::Shape {
                expected: "a nonempty square matrix".to_owned(),
                actual: format!("{}x{}", base.rows(), base.columns()),
            });
        }
        if !base.is_hermitian(HERMITIAN_TOLERANCE)? {
            return Err(AdError::NonHermitian);
        }
        for direction in &directions {
            validate_matrix_shape(base.shape(), direction.shape())?;
            if !direction.is_hermitian(HERMITIAN_TOLERANCE)? {
                return Err(AdError::NonHermitian);
            }
        }
        Ok(Self { base, directions })
    }

    /// Matrix dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.base.rows()
    }

    /// Number of independent physical coordinates.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.directions.len()
    }

    /// Constant matrix.
    #[must_use]
    pub const fn base(&self) -> &ComplexMatrix {
        &self.base
    }

    /// Parameter derivative matrices.
    #[must_use]
    pub fn directions(&self) -> &[ComplexMatrix] {
        &self.directions
    }

    /// Evaluates the Hermitian matrix.
    pub fn value(&self, parameters: &ModelParameters) -> Result<ComplexMatrix, AdError> {
        validate_count(self.parameter_count(), parameters.len())?;
        let mut result = self.base.clone();
        for (&coefficient, direction) in parameters.as_slice().iter().zip(&self.directions) {
            add_scaled_matrix(&mut result, direction, coefficient)?;
        }
        Ok(result)
    }

    /// Evaluates the directional matrix derivative.
    pub fn directional_value(&self, direction: &ModelDirection) -> Result<ComplexMatrix, AdError> {
        validate_count(self.parameter_count(), direction.len())?;
        let mut result = ComplexMatrix::zeros(self.dimension(), self.dimension());
        for (&coefficient, basis) in direction.as_slice().iter().zip(&self.directions) {
            add_scaled_matrix(&mut result, basis, coefficient)?;
        }
        Ok(result)
    }

    /// Contracts a matrix cotangent directly into physical coordinates.
    pub fn parameter_vjp(
        &self,
        matrix_cotangent: &ComplexMatrix,
    ) -> Result<ModelGradient, AdError> {
        validate_matrix_shape(self.base.shape(), matrix_cotangent.shape())?;
        ModelGradient::new(
            self.directions
                .iter()
                .map(|direction| real_frobenius_pairing(direction, matrix_cotangent))
                .collect::<Result<Vec<_>, _>>()?,
        )
    }
}

impl JvpRule<ModelParameters> for AffineHermitianFamily {
    type Output = ComplexMatrix;
    type Error = AdError;

    fn jvp(
        &self,
        parameters: &ModelParameters,
        tangent: &ModelDirection,
    ) -> Result<(ComplexMatrix, ComplexMatrix), Self::Error> {
        Ok((self.value(parameters)?, self.directional_value(tangent)?))
    }
}

/// Pullback of an affine Hermitian family.
pub struct AffineHermitianPullback<'a> {
    family: &'a AffineHermitianFamily,
}

impl Pullback<ComplexMatrix, ModelGradient> for AffineHermitianPullback<'_> {
    type Error = AdError;

    fn apply(self, cotangent: ComplexMatrix) -> Result<ModelGradient, Self::Error> {
        self.family.parameter_vjp(&cotangent)
    }
}

impl VjpRule<ModelParameters> for AffineHermitianFamily {
    type Output = ComplexMatrix;
    type Error = AdError;
    type Pullback<'a> = AffineHermitianPullback<'a>;

    fn vjp<'a>(
        &'a self,
        parameters: &'a ModelParameters,
    ) -> Result<(ComplexMatrix, Self::Pullback<'a>), Self::Error> {
        Ok((
            self.value(parameters)?,
            AffineHermitianPullback { family: self },
        ))
    }
}

/// Primal arguments of a dense complex linear solve `A x = b`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearSolveArguments {
    /// Square system matrix.
    pub matrix: ComplexMatrix,
    /// Right-hand side.
    pub right_hand_side: Vec<Complex64>,
}

/// Forward perturbation of a dense linear solve.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearSolveTangent {
    /// Matrix perturbation.
    pub matrix: ComplexMatrix,
    /// Right-hand-side perturbation.
    pub right_hand_side: Vec<Complex64>,
}

/// Reverse sensitivities of a dense linear solve.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearSolveCotangent {
    /// Matrix sensitivity.
    pub matrix: ComplexMatrix,
    /// Right-hand-side sensitivity.
    pub right_hand_side: Vec<Complex64>,
}

impl Differentiable for LinearSolveArguments {
    type Tangent = LinearSolveTangent;
    type Cotangent = LinearSolveCotangent;
}

/// Mathematical rule for a dense complex linear solve.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearSolve;

impl JvpRule<LinearSolveArguments> for LinearSolve {
    type Output = Vec<Complex64>;
    type Error = AdError;

    fn jvp(
        &self,
        arguments: &LinearSolveArguments,
        tangent: &LinearSolveTangent,
    ) -> Result<(Vec<Complex64>, Vec<Complex64>), Self::Error> {
        validate_linear_solve_arguments(arguments)?;
        validate_matrix_shape(arguments.matrix.shape(), tangent.matrix.shape())?;
        validate_vector_len(arguments.matrix.rows(), tangent.right_hand_side.len())?;
        let inverse = inverse(&arguments.matrix, false)?;
        let solution = apply_dense(&inverse, &arguments.right_hand_side)?;
        let matrix_action = apply_dense(&tangent.matrix, &solution)?;
        let residual_tangent = tangent
            .right_hand_side
            .iter()
            .zip(matrix_action)
            .map(|(right, matrix)| *right - matrix)
            .collect::<Vec<_>>();
        let solution_tangent = apply_dense(&inverse, &residual_tangent)?;
        Ok((solution, solution_tangent))
    }
}

/// One-shot pullback of a dense solve.
pub struct LinearSolvePullback {
    inverse: ComplexMatrix,
    solution: Vec<Complex64>,
}

impl Pullback<Vec<Complex64>, LinearSolveCotangent> for LinearSolvePullback {
    type Error = AdError;

    fn apply(self, cotangent: Vec<Complex64>) -> Result<LinearSolveCotangent, Self::Error> {
        validate_vector_len(self.solution.len(), cotangent.len())?;
        let adjoint_inverse = self.inverse.adjoint();
        let right_hand_side =
            apply_dense(&adjoint_inverse, &cotangent).map_err(|error| match error {
                AdError::SingularPrimal => AdError::SingularAdjoint,
                other => other,
            })?;
        let dimension = self.solution.len();
        let mut matrix = ComplexMatrix::zeros(dimension, dimension);
        for (row, right_cotangent) in right_hand_side.iter().enumerate() {
            for column in 0..dimension {
                matrix.set(
                    row,
                    column,
                    -*right_cotangent * self.solution[column].conj(),
                )?;
            }
        }
        Ok(LinearSolveCotangent {
            matrix,
            right_hand_side,
        })
    }
}

impl VjpRule<LinearSolveArguments> for LinearSolve {
    type Output = Vec<Complex64>;
    type Error = AdError;
    type Pullback<'a> = LinearSolvePullback;

    fn vjp<'a>(
        &'a self,
        arguments: &'a LinearSolveArguments,
    ) -> Result<(Vec<Complex64>, Self::Pullback<'a>), Self::Error> {
        validate_linear_solve_arguments(arguments)?;
        let inverse = inverse(&arguments.matrix, false)?;
        let solution = apply_dense(&inverse, &arguments.right_hand_side)?;
        Ok((solution.clone(), LinearSolvePullback { inverse, solution }))
    }
}

/// Primal inputs of a retarded surface Green function.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceGreenArguments {
    /// Hermitian Hamiltonian of one principal lead cell.
    pub cell_hamiltonian: ComplexMatrix,
    /// Hopping from the next cell into the surface cell.
    pub inter_cell_hopping: ComplexMatrix,
    /// Real energy.
    pub energy: f64,
    /// Positive retarded broadening.
    pub broadening: f64,
}

/// Forward perturbation of a retarded surface Green function.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceGreenTangent {
    /// Hermitian cell-Hamiltonian perturbation.
    pub cell_hamiltonian: ComplexMatrix,
    /// General complex hopping perturbation.
    pub inter_cell_hopping: ComplexMatrix,
    /// Energy perturbation.
    pub energy: f64,
    /// Broadening perturbation.
    pub broadening: f64,
}

/// Reverse sensitivities of the physical surface-Green inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceGreenCotangent {
    /// Hermitian-projected cell-Hamiltonian sensitivity.
    pub cell_hamiltonian: ComplexMatrix,
    /// General complex hopping sensitivity.
    pub inter_cell_hopping: ComplexMatrix,
    /// Energy sensitivity.
    pub energy: f64,
    /// Broadening sensitivity.
    pub broadening: f64,
}

impl Differentiable for SurfaceGreenArguments {
    type Tangent = SurfaceGreenTangent;
    type Cotangent = SurfaceGreenCotangent;
}

/// Implicit native derivative rule for the converged surface Green function.
///
/// The derivative is obtained from the fixed-point equation
///
/// `g^-1 - z I + H + V g V† = 0`
///
/// after the López-Sancho primal has converged. The rule therefore does not
/// differentiate through, or retain a tape of, the decimation iterations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceGreenRule {
    tolerance: f64,
    max_iterations: usize,
}

impl SurfaceGreenRule {
    /// Creates a rule with the same convergence controls as the primal.
    pub fn new(tolerance: f64, max_iterations: usize) -> Result<Self, AdError> {
        if !tolerance.is_finite() || tolerance <= 0.0 || max_iterations == 0 {
            return Err(AdError::Transport(TransportError::InvalidOptions));
        }
        Ok(Self {
            tolerance,
            max_iterations,
        })
    }

    /// Evaluates the retarded surface Green function.
    pub fn value(&self, arguments: &SurfaceGreenArguments) -> Result<ComplexMatrix, AdError> {
        Ok(surface_green_function(
            &arguments.cell_hamiltonian,
            &arguments.inter_cell_hopping,
            arguments.energy,
            SurfaceGreenOptions {
                broadening: arguments.broadening,
                tolerance: self.tolerance,
                max_iterations: self.max_iterations,
            },
        )?)
    }
}

impl Default for SurfaceGreenRule {
    fn default() -> Self {
        let options = SurfaceGreenOptions::default();
        Self {
            tolerance: options.tolerance,
            max_iterations: options.max_iterations,
        }
    }
}

impl JvpRule<SurfaceGreenArguments> for SurfaceGreenRule {
    type Output = ComplexMatrix;
    type Error = AdError;

    fn jvp(
        &self,
        arguments: &SurfaceGreenArguments,
        tangent: &SurfaceGreenTangent,
    ) -> Result<(ComplexMatrix, ComplexMatrix), Self::Error> {
        validate_surface_green_tangent(arguments, tangent)?;
        let green = self.value(arguments)?;
        let green_backend = to_backend(&green);
        let hopping = to_backend(&arguments.inter_cell_hopping);
        let hopping_tangent = to_backend(&tangent.inter_cell_hopping);
        let mut right_hand_side = to_backend(&tangent.cell_hamiltonian)
            + &hopping_tangent * &green_backend * hopping.adjoint()
            + &hopping * &green_backend * hopping_tangent.adjoint();
        let spectral_tangent = Complex64::new(tangent.energy, tangent.broadening);
        for index in 0..green.rows() {
            right_hand_side[(index, index)] -= spectral_tangent;
        }
        let green_tangent =
            solve_surface_green_linearization(&green_backend, &hopping, &right_hand_side, false)?;
        Ok((green, from_backend(&green_tangent)?))
    }
}

/// One-shot implicit pullback for a converged surface Green function.
pub struct SurfaceGreenPullback {
    green: DMatrix<Complex64>,
    hopping: DMatrix<Complex64>,
}

impl Pullback<ComplexMatrix, SurfaceGreenCotangent> for SurfaceGreenPullback {
    type Error = AdError;

    fn apply(self, cotangent: ComplexMatrix) -> Result<SurfaceGreenCotangent, Self::Error> {
        validate_matrix_shape((self.green.nrows(), self.green.ncols()), cotangent.shape())?;
        let lambda = solve_surface_green_linearization(
            &self.green,
            &self.hopping,
            &to_backend(&cotangent),
            true,
        )?;
        let hamiltonian = from_backend(&hermitian_part(&lambda))?;
        let hopping = &lambda * &self.hopping * self.green.adjoint()
            + lambda.adjoint() * &self.hopping * &self.green;
        let trace = lambda.trace();
        if !trace.re.is_finite() || !trace.im.is_finite() {
            return Err(AdError::NonFiniteDerivative);
        }
        Ok(SurfaceGreenCotangent {
            cell_hamiltonian: hamiltonian,
            inter_cell_hopping: from_backend(&hopping)?,
            energy: -trace.re,
            broadening: -trace.im,
        })
    }
}

impl VjpRule<SurfaceGreenArguments> for SurfaceGreenRule {
    type Output = ComplexMatrix;
    type Error = AdError;
    type Pullback<'a> = SurfaceGreenPullback;

    fn vjp<'a>(
        &'a self,
        arguments: &'a SurfaceGreenArguments,
    ) -> Result<(ComplexMatrix, Self::Pullback<'a>), Self::Error> {
        let green = self.value(arguments)?;
        Ok((
            green.clone(),
            SurfaceGreenPullback {
                green: to_backend(&green),
                hopping: to_backend(&arguments.inter_cell_hopping),
            },
        ))
    }
}

/// Rule for one spectrally isolated Hermitian eigenvalue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsolatedEigenvalue {
    index: usize,
    minimum_gap: f64,
}

impl IsolatedEigenvalue {
    /// Creates a rule with an explicit validity threshold.
    pub fn new(index: usize, minimum_gap: f64) -> Result<Self, AdError> {
        if !minimum_gap.is_finite() || minimum_gap <= 0.0 {
            return Err(AdError::GapTooSmall {
                gap: minimum_gap,
                minimum: f64::EPSILON,
            });
        }
        Ok(Self { index, minimum_gap })
    }
}

impl JvpRule<ComplexMatrix> for IsolatedEigenvalue {
    type Output = f64;
    type Error = AdError;

    fn jvp(
        &self,
        matrix: &ComplexMatrix,
        tangent: &ComplexMatrix,
    ) -> Result<(f64, f64), Self::Error> {
        validate_matrix_shape(matrix.shape(), tangent.shape())?;
        let (value, gradient) = isolated_eigenvalue_gradient(matrix, self.index, self.minimum_gap)?;
        Ok((value, real_frobenius_pairing(&gradient, tangent)?))
    }
}

/// Pullback for one isolated eigenvalue.
pub struct IsolatedEigenvaluePullback {
    matrix_gradient: ComplexMatrix,
}

impl Pullback<f64, ComplexMatrix> for IsolatedEigenvaluePullback {
    type Error = AdError;

    fn apply(self, cotangent: f64) -> Result<ComplexMatrix, Self::Error> {
        scale_matrix(&self.matrix_gradient, cotangent)
    }
}

impl VjpRule<ComplexMatrix> for IsolatedEigenvalue {
    type Output = f64;
    type Error = AdError;
    type Pullback<'a> = IsolatedEigenvaluePullback;

    fn vjp<'a>(
        &'a self,
        matrix: &'a ComplexMatrix,
    ) -> Result<(f64, Self::Pullback<'a>), Self::Error> {
        let (value, matrix_gradient) =
            isolated_eigenvalue_gradient(matrix, self.index, self.minimum_gap)?;
        Ok((value, IsolatedEigenvaluePullback { matrix_gradient }))
    }
}

/// Gauge-invariant distance from a separated occupied subspace to a target
/// projector.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectorDistance {
    occupied: usize,
    target: ComplexMatrix,
    minimum_gap: f64,
}

impl ProjectorDistance {
    /// Creates a projector objective with an explicit separating-gap
    /// threshold.
    pub fn new(occupied: usize, target: ComplexMatrix, minimum_gap: f64) -> Result<Self, AdError> {
        if target.rows() == 0 || target.rows() != target.columns() {
            return Err(AdError::InvalidProjector);
        }
        if !minimum_gap.is_finite() || minimum_gap <= 0.0 {
            return Err(AdError::GapTooSmall {
                gap: minimum_gap,
                minimum: f64::EPSILON,
            });
        }
        validate_projector(&target)?;
        if occupied == 0 || occupied >= target.rows() {
            return Err(AdError::InvalidSubspace);
        }
        Ok(Self {
            occupied,
            target,
            minimum_gap,
        })
    }

    /// Evaluates the objective and its Hermitian matrix gradient.
    pub fn value_and_gradient(
        &self,
        matrix: &ComplexMatrix,
    ) -> Result<(f64, ComplexMatrix), AdError> {
        validate_matrix_shape(self.target.shape(), matrix.shape())?;
        let decomposition = separated_projector(matrix, self.occupied, self.minimum_gap)?;
        let difference = subtract_matrices(&decomposition.projector, &self.target)?;
        let value = 0.5 * real_frobenius_pairing(&difference, &difference)?;
        let gradient = projector_input_cotangent(&decomposition, &difference)?;
        Ok((value, gradient))
    }
}

impl JvpRule<ComplexMatrix> for ProjectorDistance {
    type Output = f64;
    type Error = AdError;

    fn jvp(
        &self,
        matrix: &ComplexMatrix,
        tangent: &ComplexMatrix,
    ) -> Result<(f64, f64), Self::Error> {
        validate_matrix_shape(matrix.shape(), tangent.shape())?;
        let (value, gradient) = self.value_and_gradient(matrix)?;
        Ok((value, real_frobenius_pairing(&gradient, tangent)?))
    }
}

/// Pullback of a gauge-invariant projector distance.
pub struct ProjectorDistancePullback {
    matrix_gradient: ComplexMatrix,
}

impl Pullback<f64, ComplexMatrix> for ProjectorDistancePullback {
    type Error = AdError;

    fn apply(self, cotangent: f64) -> Result<ComplexMatrix, Self::Error> {
        scale_matrix(&self.matrix_gradient, cotangent)
    }
}

impl VjpRule<ComplexMatrix> for ProjectorDistance {
    type Output = f64;
    type Error = AdError;
    type Pullback<'a> = ProjectorDistancePullback;

    fn vjp<'a>(
        &'a self,
        matrix: &'a ComplexMatrix,
    ) -> Result<(f64, Self::Pullback<'a>), Self::Error> {
        let (value, matrix_gradient) = self.value_and_gradient(matrix)?;
        Ok((value, ProjectorDistancePullback { matrix_gradient }))
    }
}

/// Complete parameter-to-projector-distance workflow.
pub struct SpectralProjectorObjective<'a> {
    family: &'a AffineHermitianFamily,
    objective: ProjectorDistance,
}

impl<'a> SpectralProjectorObjective<'a> {
    /// Creates the complete spectral workflow.
    pub fn new(
        family: &'a AffineHermitianFamily,
        occupied: usize,
        target: ComplexMatrix,
        minimum_gap: f64,
    ) -> Result<Self, AdError> {
        validate_matrix_shape((family.dimension(), family.dimension()), target.shape())?;
        Ok(Self {
            family,
            objective: ProjectorDistance::new(occupied, target, minimum_gap)?,
        })
    }

    /// Primal objective value.
    pub fn value(&self, parameters: &ModelParameters) -> Result<f64, AdError> {
        Ok(self
            .objective
            .value_and_gradient(&self.family.value(parameters)?)?
            .0)
    }

    /// Value and forward directional derivative.
    pub fn jvp(
        &self,
        parameters: &ModelParameters,
        direction: &ModelDirection,
    ) -> Result<(f64, f64), AdError> {
        <Self as JvpRule<ModelParameters>>::jvp(self, parameters, direction)
    }

    /// Value and physical gradient for a real scalar objective.
    pub fn value_and_grad(
        &self,
        parameters: &ModelParameters,
    ) -> Result<(f64, ModelGradient), AdError> {
        let (value, pullback) = <Self as VjpRule<ModelParameters>>::vjp(self, parameters)?;
        Ok((value, pullback.apply(1.0)?))
    }
}

impl JvpRule<ModelParameters> for SpectralProjectorObjective<'_> {
    type Output = f64;
    type Error = AdError;

    fn jvp(
        &self,
        parameters: &ModelParameters,
        tangent: &ModelDirection,
    ) -> Result<(f64, f64), Self::Error> {
        let (matrix, matrix_tangent) = self.family.jvp(parameters, tangent)?;
        self.objective.jvp(&matrix, &matrix_tangent)
    }
}

/// Pullback of a complete scalar workflow after direct contraction into model
/// coordinates.
pub struct ScalarModelPullback {
    gradient: ModelGradient,
}

impl Pullback<f64, ModelGradient> for ScalarModelPullback {
    type Error = AdError;

    fn apply(self, cotangent: f64) -> Result<ModelGradient, Self::Error> {
        self.gradient.scaled(cotangent)
    }
}

impl VjpRule<ModelParameters> for SpectralProjectorObjective<'_> {
    type Output = f64;
    type Error = AdError;
    type Pullback<'a>
        = ScalarModelPullback
    where
        Self: 'a,
        ModelParameters: 'a;

    fn vjp<'a>(
        &'a self,
        parameters: &'a ModelParameters,
    ) -> Result<(f64, Self::Pullback<'a>), Self::Error> {
        let matrix = self.family.value(parameters)?;
        let (value, matrix_gradient) = self.objective.value_and_gradient(&matrix)?;
        let gradient = self.family.parameter_vjp(&matrix_gradient)?;
        Ok((value, ScalarModelPullback { gradient }))
    }
}

/// Gauge-invariant one-dimensional quantum-metric objective on a periodic
/// momentum mesh.
///
/// For occupied projectors `P_i` on an evenly spaced closed mesh, the
/// objective is
///
/// `sum_i ||P_(i+1) - P_i||_F^2 / (2 N delta_k^2)`.
///
/// It is a finite-difference discretization of the Brillouin-zone-averaged
/// quantum metric. Working with projectors makes the value and derivative
/// invariant under arbitrary rotations inside the occupied eigenspace.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantumMetricMeshObjective {
    families: Vec<AffineHermitianFamily>,
    occupied: usize,
    momentum_step: f64,
    minimum_gap: f64,
    parameter_count: usize,
}

impl QuantumMetricMeshObjective {
    /// Creates a periodic mesh objective from Hamiltonian families that share
    /// one physical parameter space.
    pub fn new(
        families: Vec<AffineHermitianFamily>,
        occupied: usize,
        momentum_step: f64,
        minimum_gap: f64,
    ) -> Result<Self, AdError> {
        if families.len() < 3 {
            return Err(AdError::Shape {
                expected: "at least three points on a periodic momentum mesh".to_owned(),
                actual: format!("{} points", families.len()),
            });
        }
        if !momentum_step.is_finite() || momentum_step <= 0.0 {
            return Err(AdError::NonFiniteParameter { index: 0 });
        }
        if !minimum_gap.is_finite() || minimum_gap <= 0.0 {
            return Err(AdError::GapTooSmall {
                gap: minimum_gap,
                minimum: f64::EPSILON,
            });
        }
        let dimension = families[0].dimension();
        let parameter_count = families[0].parameter_count();
        if occupied == 0 || occupied >= dimension {
            return Err(AdError::InvalidSubspace);
        }
        for family in &families[1..] {
            validate_matrix_shape(
                (dimension, dimension),
                (family.dimension(), family.dimension()),
            )?;
            validate_count(parameter_count, family.parameter_count())?;
        }
        Ok(Self {
            families,
            occupied,
            momentum_step,
            minimum_gap,
            parameter_count,
        })
    }

    /// Number of points on the closed momentum mesh.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.families.len()
    }

    /// Scalar quantum-metric objective.
    pub fn value(&self, parameters: &ModelParameters) -> Result<f64, AdError> {
        Ok(self.value_and_grad(parameters)?.0)
    }

    /// Value and directional derivative.
    pub fn jvp(
        &self,
        parameters: &ModelParameters,
        direction: &ModelDirection,
    ) -> Result<(f64, f64), AdError> {
        <Self as JvpRule<ModelParameters>>::jvp(self, parameters, direction)
    }

    /// Value and gradient in the shared physical parameter space.
    pub fn value_and_grad(
        &self,
        parameters: &ModelParameters,
    ) -> Result<(f64, ModelGradient), AdError> {
        validate_count(self.parameter_count, parameters.len())?;
        let decompositions = self
            .families
            .iter()
            .map(|family| {
                separated_projector(&family.value(parameters)?, self.occupied, self.minimum_gap)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dimension = self.families[0].dimension();
        let mut projector_cotangents =
            vec![ComplexMatrix::zeros(dimension, dimension); self.families.len()];
        let edge_scale =
            1.0 / (self.families.len() as f64 * self.momentum_step * self.momentum_step);
        let mut value = 0.0;
        for point in 0..self.families.len() {
            let next = (point + 1) % self.families.len();
            let difference = subtract_matrices(
                &decompositions[next].projector,
                &decompositions[point].projector,
            )?;
            value += 0.5 * edge_scale * real_frobenius_pairing(&difference, &difference)?;
            add_scaled_matrix(&mut projector_cotangents[point], &difference, -edge_scale)?;
            add_scaled_matrix(&mut projector_cotangents[next], &difference, edge_scale)?;
        }
        let mut gradient = ModelGradient::zeros(self.parameter_count);
        for ((family, decomposition), projector_cotangent) in self
            .families
            .iter()
            .zip(&decompositions)
            .zip(&projector_cotangents)
        {
            let matrix_cotangent = projector_input_cotangent(decomposition, projector_cotangent)?;
            gradient.accumulate(&family.parameter_vjp(&matrix_cotangent)?)?;
        }
        if value.is_finite() {
            Ok((value, gradient))
        } else {
            Err(AdError::NonFiniteDerivative)
        }
    }
}

impl JvpRule<ModelParameters> for QuantumMetricMeshObjective {
    type Output = f64;
    type Error = AdError;

    fn jvp(
        &self,
        parameters: &ModelParameters,
        tangent: &ModelDirection,
    ) -> Result<(f64, f64), Self::Error> {
        validate_count(self.parameter_count, tangent.len())?;
        let (value, gradient) = self.value_and_grad(parameters)?;
        let derivative = gradient
            .as_slice()
            .iter()
            .zip(tangent.as_slice())
            .map(|(gradient, tangent)| gradient * tangent)
            .sum::<f64>();
        Ok((value, derivative))
    }
}

impl VjpRule<ModelParameters> for QuantumMetricMeshObjective {
    type Output = f64;
    type Error = AdError;
    type Pullback<'a> = ScalarModelPullback;

    fn vjp<'a>(
        &'a self,
        parameters: &'a ModelParameters,
    ) -> Result<(f64, Self::Pullback<'a>), Self::Error> {
        let (value, gradient) = self.value_and_grad(parameters)?;
        Ok((value, ScalarModelPullback { gradient }))
    }
}

/// Complete parameter-to-Caroli-transmission workflow with fixed retarded
/// lead self-energies.
pub struct OpenTransmissionObjective<'a> {
    family: &'a AffineHermitianFamily,
    self_energies: Vec<ComplexMatrix>,
    energy: f64,
    drain: usize,
    source: usize,
}

impl<'a> OpenTransmissionObjective<'a> {
    /// Creates a differentiable transmission objective.
    pub fn new(
        family: &'a AffineHermitianFamily,
        self_energies: Vec<ComplexMatrix>,
        energy: f64,
        drain: usize,
        source: usize,
    ) -> Result<Self, AdError> {
        if !energy.is_finite() {
            return Err(AdError::NonFiniteParameter { index: 0 });
        }
        for self_energy in &self_energies {
            validate_matrix_shape(
                (family.dimension(), family.dimension()),
                self_energy.shape(),
            )?;
        }
        if drain >= self_energies.len() || source >= self_energies.len() {
            return Err(AdError::Shape {
                expected: format!("lead index below {}", self_energies.len()),
                actual: drain.max(source).to_string(),
            });
        }
        Ok(Self {
            family,
            self_energies,
            energy,
            drain,
            source,
        })
    }

    /// Transmission value.
    pub fn value(&self, parameters: &ModelParameters) -> Result<f64, AdError> {
        Ok(self.value_and_matrix_gradient(parameters)?.0)
    }

    /// Value and forward directional derivative.
    pub fn jvp(
        &self,
        parameters: &ModelParameters,
        direction: &ModelDirection,
    ) -> Result<(f64, f64), AdError> {
        <Self as JvpRule<ModelParameters>>::jvp(self, parameters, direction)
    }

    /// Value and physical gradient.
    pub fn value_and_grad(
        &self,
        parameters: &ModelParameters,
    ) -> Result<(f64, ModelGradient), AdError> {
        let (value, pullback) = <Self as VjpRule<ModelParameters>>::vjp(self, parameters)?;
        Ok((value, pullback.apply(1.0)?))
    }

    fn value_and_matrix_gradient(
        &self,
        parameters: &ModelParameters,
    ) -> Result<(f64, ComplexMatrix), AdError> {
        let hamiltonian = self.family.value(parameters)?;
        let solution =
            solve_open_system_from_self_energies(&hamiltonian, &self.self_energies, self.energy)?;
        let value = solution.transmission(self.drain, self.source)?;
        let green = to_backend(solution.retarded_green());
        let gamma_drain = to_backend(&solution.broadenings()[self.drain]);
        let gamma_source = to_backend(&solution.broadenings()[self.source]);
        let raw_gradient = green.adjoint()
            * gamma_drain
            * &green
            * gamma_source
            * green.adjoint()
            * Complex64::new(2.0, 0.0);
        let gradient = from_backend(&hermitian_part(&raw_gradient))?;
        Ok((value, gradient))
    }
}

impl JvpRule<ModelParameters> for OpenTransmissionObjective<'_> {
    type Output = f64;
    type Error = AdError;

    fn jvp(
        &self,
        parameters: &ModelParameters,
        tangent: &ModelDirection,
    ) -> Result<(f64, f64), Self::Error> {
        let (value, matrix_gradient) = self.value_and_matrix_gradient(parameters)?;
        let matrix_tangent = self.family.directional_value(tangent)?;
        Ok((
            value,
            real_frobenius_pairing(&matrix_gradient, &matrix_tangent)?,
        ))
    }
}

impl VjpRule<ModelParameters> for OpenTransmissionObjective<'_> {
    type Output = f64;
    type Error = AdError;
    type Pullback<'a>
        = ScalarModelPullback
    where
        Self: 'a,
        ModelParameters: 'a;

    fn vjp<'a>(
        &'a self,
        parameters: &'a ModelParameters,
    ) -> Result<(f64, Self::Pullback<'a>), Self::Error> {
        let (value, matrix_gradient) = self.value_and_matrix_gradient(parameters)?;
        let gradient = self.family.parameter_vjp(&matrix_gradient)?;
        Ok((value, ScalarModelPullback { gradient }))
    }
}

/// Continuous values of one periodic lead and its device interface.
#[derive(Clone, Debug, PartialEq)]
pub struct DifferentiableLead {
    /// Hermitian principal-cell Hamiltonian.
    pub cell_hamiltonian: ComplexMatrix,
    /// Periodic inter-cell hopping.
    pub inter_cell_hopping: ComplexMatrix,
    /// Device-by-lead interface coupling.
    pub coupling: ComplexMatrix,
    /// Positive surface-Green broadening.
    pub broadening: f64,
}

/// Forward perturbation of one differentiable lead.
#[derive(Clone, Debug, PartialEq)]
pub struct LeadDirection {
    /// Hermitian principal-cell perturbation.
    pub cell_hamiltonian: ComplexMatrix,
    /// Periodic-hopping perturbation.
    pub inter_cell_hopping: ComplexMatrix,
    /// Interface-coupling perturbation.
    pub coupling: ComplexMatrix,
    /// Broadening perturbation.
    pub broadening: f64,
}

/// Reverse sensitivities of one differentiable lead.
#[derive(Clone, Debug, PartialEq)]
pub struct LeadGradient {
    /// Hermitian principal-cell sensitivity.
    pub cell_hamiltonian: ComplexMatrix,
    /// Periodic-hopping sensitivity.
    pub inter_cell_hopping: ComplexMatrix,
    /// Interface-coupling sensitivity.
    pub coupling: ComplexMatrix,
    /// Broadening sensitivity.
    pub broadening: f64,
}

impl Differentiable for DifferentiableLead {
    type Tangent = LeadDirection;
    type Cotangent = LeadGradient;
}

/// Continuous values of a finite device with periodic leads.
#[derive(Clone, Debug, PartialEq)]
pub struct DifferentiableOpenSystem {
    /// Hermitian finite-device Hamiltonian.
    pub device_hamiltonian: ComplexMatrix,
    /// Periodic leads in a fixed discrete order.
    pub leads: Vec<DifferentiableLead>,
    /// Shared real scattering energy.
    pub energy: f64,
}

/// Forward perturbation of a differentiable open system.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenSystemDirection {
    /// Hermitian finite-device perturbation.
    pub device_hamiltonian: ComplexMatrix,
    /// One perturbation per fixed lead.
    pub leads: Vec<LeadDirection>,
    /// Scattering-energy perturbation.
    pub energy: f64,
}

/// Reverse sensitivities of a differentiable open system.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenSystemGradient {
    /// Hermitian finite-device sensitivity.
    pub device_hamiltonian: ComplexMatrix,
    /// One physical sensitivity per fixed lead.
    pub leads: Vec<LeadGradient>,
    /// Shared scattering-energy sensitivity.
    pub energy: f64,
}

impl Differentiable for DifferentiableOpenSystem {
    type Tangent = OpenSystemDirection;
    type Cotangent = OpenSystemGradient;
}

/// Complete device-and-lead transmission rule.
///
/// This composes implicit surface-Green pullbacks, interface self-energy
/// products, the device resolvent, lead broadenings, and the Caroli
/// transmission into one Rust-owned VJP.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpenSystemTransmission {
    drain: usize,
    source: usize,
    surface_rule: SurfaceGreenRule,
}

impl OpenSystemTransmission {
    /// Creates a complete transmission rule.
    pub fn new(
        drain: usize,
        source: usize,
        surface_tolerance: f64,
        surface_max_iterations: usize,
    ) -> Result<Self, AdError> {
        Ok(Self {
            drain,
            source,
            surface_rule: SurfaceGreenRule::new(surface_tolerance, surface_max_iterations)?,
        })
    }

    /// Transmission value.
    pub fn value(&self, system: &DifferentiableOpenSystem) -> Result<f64, AdError> {
        Ok(self.value_and_grad(system)?.0)
    }

    /// Value and directional derivative.
    pub fn jvp(
        &self,
        system: &DifferentiableOpenSystem,
        direction: &OpenSystemDirection,
    ) -> Result<(f64, f64), AdError> {
        <Self as JvpRule<DifferentiableOpenSystem>>::jvp(self, system, direction)
    }

    /// Value and complete device-and-lead cotangent.
    pub fn value_and_grad(
        &self,
        system: &DifferentiableOpenSystem,
    ) -> Result<(f64, OpenSystemGradient), AdError> {
        validate_differentiable_open_system(system)?;
        if self.drain >= system.leads.len() || self.source >= system.leads.len() {
            return Err(AdError::Shape {
                expected: format!("lead index below {}", system.leads.len()),
                actual: self.drain.max(self.source).to_string(),
            });
        }

        let mut surfaces = Vec::with_capacity(system.leads.len());
        let mut surface_pullbacks = Vec::with_capacity(system.leads.len());
        let mut self_energies = Vec::with_capacity(system.leads.len());
        for lead in &system.leads {
            let arguments = SurfaceGreenArguments {
                cell_hamiltonian: lead.cell_hamiltonian.clone(),
                inter_cell_hopping: lead.inter_cell_hopping.clone(),
                energy: system.energy,
                broadening: lead.broadening,
            };
            let (surface, pullback) = self.surface_rule.vjp(&arguments)?;
            let coupling = to_backend(&lead.coupling);
            let self_energy = &coupling * to_backend(&surface) * coupling.adjoint();
            surfaces.push(surface);
            surface_pullbacks.push(pullback);
            self_energies.push(from_backend(&self_energy)?);
        }

        let (value, device_cotangent, self_energy_cotangents, mut energy_cotangent) =
            caroli_input_cotangents(
                &system.device_hamiltonian,
                &self_energies,
                system.energy,
                self.drain,
                self.source,
            )?;
        let mut lead_gradients = Vec::with_capacity(system.leads.len());
        for (((lead, surface), surface_pullback), self_energy_cotangent) in system
            .leads
            .iter()
            .zip(&surfaces)
            .zip(surface_pullbacks)
            .zip(self_energy_cotangents)
        {
            let coupling = to_backend(&lead.coupling);
            let surface = to_backend(surface);
            let self_energy_cotangent = to_backend(&self_energy_cotangent);
            let surface_cotangent = coupling.adjoint() * &self_energy_cotangent * &coupling;
            let coupling_cotangent = &self_energy_cotangent * &coupling * surface.adjoint()
                + self_energy_cotangent.adjoint() * &coupling * &surface;
            let surface_gradient = surface_pullback.apply(from_backend(&surface_cotangent)?)?;
            energy_cotangent += surface_gradient.energy;
            lead_gradients.push(LeadGradient {
                cell_hamiltonian: surface_gradient.cell_hamiltonian,
                inter_cell_hopping: surface_gradient.inter_cell_hopping,
                coupling: from_backend(&coupling_cotangent)?,
                broadening: surface_gradient.broadening,
            });
        }
        if energy_cotangent.is_finite() {
            Ok((
                value,
                OpenSystemGradient {
                    device_hamiltonian: device_cotangent,
                    leads: lead_gradients,
                    energy: energy_cotangent,
                },
            ))
        } else {
            Err(AdError::NonFiniteDerivative)
        }
    }
}

impl JvpRule<DifferentiableOpenSystem> for OpenSystemTransmission {
    type Output = f64;
    type Error = AdError;

    fn jvp(
        &self,
        system: &DifferentiableOpenSystem,
        tangent: &OpenSystemDirection,
    ) -> Result<(f64, f64), Self::Error> {
        validate_open_system_direction(system, tangent)?;
        let (value, gradient) = self.value_and_grad(system)?;
        let mut derivative =
            real_frobenius_pairing(&gradient.device_hamiltonian, &tangent.device_hamiltonian)?
                + gradient.energy * tangent.energy;
        for (gradient, tangent) in gradient.leads.iter().zip(&tangent.leads) {
            derivative +=
                real_frobenius_pairing(&gradient.cell_hamiltonian, &tangent.cell_hamiltonian)?
                    + real_frobenius_pairing(
                        &gradient.inter_cell_hopping,
                        &tangent.inter_cell_hopping,
                    )?
                    + real_frobenius_pairing(&gradient.coupling, &tangent.coupling)?
                    + gradient.broadening * tangent.broadening;
        }
        if derivative.is_finite() {
            Ok((value, derivative))
        } else {
            Err(AdError::NonFiniteDerivative)
        }
    }
}

/// One-shot complete open-system transmission pullback.
pub struct OpenSystemTransmissionPullback {
    gradient: OpenSystemGradient,
}

impl Pullback<f64, OpenSystemGradient> for OpenSystemTransmissionPullback {
    type Error = AdError;

    fn apply(self, cotangent: f64) -> Result<OpenSystemGradient, Self::Error> {
        scale_open_system_gradient(self.gradient, cotangent)
    }
}

impl VjpRule<DifferentiableOpenSystem> for OpenSystemTransmission {
    type Output = f64;
    type Error = AdError;
    type Pullback<'a> = OpenSystemTransmissionPullback;

    fn vjp<'a>(
        &'a self,
        system: &'a DifferentiableOpenSystem,
    ) -> Result<(f64, Self::Pullback<'a>), Self::Error> {
        let (value, gradient) = self.value_and_grad(system)?;
        Ok((value, OpenSystemTransmissionPullback { gradient }))
    }
}

/// One physical Hermitian contribution to a canonical sparse matrix.
///
/// An off-diagonal contribution automatically includes its conjugate partner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseHermitianTerm {
    /// Physical parameter index.
    pub parameter: usize,
    /// Matrix row.
    pub row: usize,
    /// Matrix column.
    pub column: usize,
    /// Coefficient at `(row, column)`.
    pub coefficient: Complex64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SparseContribution {
    parameter: usize,
    row: usize,
    column: usize,
    coefficient: Complex64,
}

/// Hermitian sparse affine operator with fixed canonical sparsity.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseAffineOperator {
    base: CsrMatrix,
    parameter_count: usize,
    contributions: Vec<SparseContribution>,
}

impl SparseAffineOperator {
    /// Creates a sparse family, validating the base and every physical term.
    pub fn new(
        base: CsrMatrix,
        parameter_count: usize,
        terms: Vec<SparseHermitianTerm>,
    ) -> Result<Self, AdError> {
        if base.rows() != base.columns() {
            return Err(AdError::Shape {
                expected: "a square sparse matrix".to_owned(),
                actual: format!("{}x{}", base.rows(), base.columns()),
            });
        }
        if !base.is_hermitian(HERMITIAN_TOLERANCE)? {
            return Err(AdError::NonHermitian);
        }
        let mut contributions = Vec::with_capacity(2 * terms.len());
        for term in terms {
            if term.parameter >= parameter_count {
                return Err(AdError::ParameterCount {
                    expected: parameter_count,
                    actual: term.parameter + 1,
                });
            }
            if term.row >= base.rows() || term.column >= base.columns() {
                return Err(AdError::Shape {
                    expected: format!("indices below {}", base.rows()),
                    actual: format!("({}, {})", term.row, term.column),
                });
            }
            if !is_finite_complex(term.coefficient) {
                return Err(AdError::NonFiniteParameter {
                    index: term.parameter,
                });
            }
            find_csr_entry(&base, term.row, term.column)?;
            if term.row == term.column {
                if term.coefficient.im.abs() > HERMITIAN_TOLERANCE {
                    return Err(AdError::NonHermitian);
                }
                contributions.push(SparseContribution {
                    parameter: term.parameter,
                    row: term.row,
                    column: term.column,
                    coefficient: Complex64::new(term.coefficient.re, 0.0),
                });
            } else {
                find_csr_entry(&base, term.column, term.row)?;
                contributions.push(SparseContribution {
                    parameter: term.parameter,
                    row: term.row,
                    column: term.column,
                    coefficient: term.coefficient,
                });
                contributions.push(SparseContribution {
                    parameter: term.parameter,
                    row: term.column,
                    column: term.row,
                    coefficient: term.coefficient.conj(),
                });
            }
        }
        Ok(Self {
            base,
            parameter_count,
            contributions,
        })
    }

    /// Matrix dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.base.rows()
    }

    /// Number of physical coordinates.
    #[must_use]
    pub const fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    /// Binds parameters without materializing a new sparse matrix.
    pub fn bind<'a>(
        &'a self,
        parameters: &'a ModelParameters,
    ) -> Result<SparseAffineSnapshot<'a>, AdError> {
        validate_count(self.parameter_count, parameters.len())?;
        Ok(SparseAffineSnapshot {
            family: self,
            parameters,
        })
    }

    /// Materializes the canonical CSR values for interoperability or
    /// validation.
    pub fn value(&self, parameters: &ModelParameters) -> Result<CsrMatrix, AdError> {
        validate_count(self.parameter_count, parameters.len())?;
        let mut values = self.base.values().to_vec();
        for contribution in &self.contributions {
            let entry = find_csr_entry(&self.base, contribution.row, contribution.column)?;
            values[entry] +=
                parameters.as_slice()[contribution.parameter] * contribution.coefficient;
        }
        Ok(CsrMatrix::new(
            self.base.rows(),
            self.base.columns(),
            self.base.row_offsets().to_vec(),
            self.base.column_indices().to_vec(),
            values,
        )?)
    }

    /// Applies `dA(theta)[direction]` to a state.
    pub fn parameter_jvp_into(
        &self,
        direction: &ModelDirection,
        state: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), AdError> {
        validate_count(self.parameter_count, direction.len())?;
        validate_vector_len(self.base.columns(), state.len())?;
        validate_vector_len(self.base.rows(), output.len())?;
        output.fill(Complex64::new(0.0, 0.0));
        for contribution in &self.contributions {
            output[contribution.row] += direction.as_slice()[contribution.parameter]
                * contribution.coefficient
                * state[contribution.column];
        }
        validate_complex_vector(output)
    }

    /// Contracts `Re <output_cotangent, dA state>` into physical coordinates.
    pub fn parameter_vjp(
        &self,
        state: &[Complex64],
        output_cotangent: &[Complex64],
    ) -> Result<ModelGradient, AdError> {
        validate_vector_len(self.base.columns(), state.len())?;
        validate_vector_len(self.base.rows(), output_cotangent.len())?;
        let mut gradient = vec![0.0; self.parameter_count];
        for contribution in &self.contributions {
            gradient[contribution.parameter] += (output_cotangent[contribution.row].conj()
                * contribution.coefficient
                * state[contribution.column])
                .re;
        }
        ModelGradient::new(gradient)
    }
}

/// A borrowed parameter binding of a sparse affine operator.
pub struct SparseAffineSnapshot<'a> {
    family: &'a SparseAffineOperator,
    parameters: &'a ModelParameters,
}

impl LinearOperator for SparseAffineSnapshot<'_> {
    fn rows(&self) -> usize {
        self.family.base.rows()
    }

    fn columns(&self) -> usize {
        self.family.base.columns()
    }

    fn apply_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), LinearOperatorError> {
        self.family.base.apply_into(input, output)?;
        for contribution in &self.family.contributions {
            output[contribution.row] += self.parameters.as_slice()[contribution.parameter]
                * contribution.coefficient
                * input[contribution.column];
        }
        if output.iter().all(|value| is_finite_complex(*value)) {
            Ok(())
        } else {
            Err(LinearOperatorError::NonFiniteValue)
        }
    }
}

impl AdjointLinearOperator for SparseAffineSnapshot<'_> {
    fn apply_adjoint_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), LinearOperatorError> {
        self.family.base.apply_adjoint_into(input, output)?;
        for contribution in &self.family.contributions {
            output[contribution.column] += self.parameters.as_slice()[contribution.parameter]
                * contribution.coefficient.conj()
                * input[contribution.row];
        }
        if output.iter().all(|value| is_finite_complex(*value)) {
            Ok(())
        } else {
            Err(LinearOperatorError::NonFiniteValue)
        }
    }
}

struct AdjointOperatorView<'a, O>(&'a O);

impl<O: AdjointLinearOperator> LinearOperator for AdjointOperatorView<'_, O> {
    fn rows(&self) -> usize {
        self.0.columns()
    }

    fn columns(&self) -> usize {
        self.0.rows()
    }

    fn apply_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), LinearOperatorError> {
        self.0.apply_adjoint_into(input, output)
    }
}

/// Diagnostics from one sparse primal-and-adjoint scalar solve.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseSolveReport {
    value: f64,
    gradient: ModelGradient,
    primal_iterations: usize,
    primal_residual_norm: f64,
    adjoint_iterations: usize,
    adjoint_residual_norm: f64,
}

impl SparseSolveReport {
    /// Scalar linear functional of the solution.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Gradient in physical parameter coordinates.
    #[must_use]
    pub const fn gradient(&self) -> &ModelGradient {
        &self.gradient
    }

    /// Primal GMRES iterations.
    #[must_use]
    pub const fn primal_iterations(&self) -> usize {
        self.primal_iterations
    }

    /// True unpreconditioned primal residual norm.
    #[must_use]
    pub const fn primal_residual_norm(&self) -> f64 {
        self.primal_residual_norm
    }

    /// Adjoint GMRES iterations.
    #[must_use]
    pub const fn adjoint_iterations(&self) -> usize {
        self.adjoint_iterations
    }

    /// True unpreconditioned adjoint residual norm.
    #[must_use]
    pub const fn adjoint_residual_norm(&self) -> f64 {
        self.adjoint_residual_norm
    }
}

/// Sparse matrix-free linear-solve objective
/// `Re <output_cotangent, A(theta)^-1 right_hand_side>`.
///
/// The VJP performs one primal and one adjoint GMRES solve regardless of the
/// number of physical parameters, and contracts directly through
/// [`SparseAffineOperator::parameter_vjp`].
pub struct SparseLinearFunctionalObjective<'a> {
    operator: &'a SparseAffineOperator,
    right_hand_side: Vec<Complex64>,
    output_cotangent: Vec<Complex64>,
    options: GmresOptions,
}

impl<'a> SparseLinearFunctionalObjective<'a> {
    /// Creates a sparse scalar solve objective.
    pub fn new(
        operator: &'a SparseAffineOperator,
        right_hand_side: Vec<Complex64>,
        output_cotangent: Vec<Complex64>,
        options: GmresOptions,
    ) -> Result<Self, AdError> {
        validate_vector_len(operator.dimension(), right_hand_side.len())?;
        validate_vector_len(operator.dimension(), output_cotangent.len())?;
        validate_complex_vector(&right_hand_side)?;
        validate_complex_vector(&output_cotangent)?;
        Ok(Self {
            operator,
            right_hand_side,
            output_cotangent,
            options,
        })
    }

    /// Scalar primal value.
    pub fn value(&self, parameters: &ModelParameters) -> Result<f64, AdError> {
        let bound = self.operator.bind(parameters)?;
        let solution = gmres(&bound, &self.right_hand_side, None, self.options)
            .map_err(AdError::PrimalIterative)?;
        real_vector_pairing(&self.output_cotangent, solution.vector())
    }

    /// Value and forward directional derivative.
    pub fn jvp(
        &self,
        parameters: &ModelParameters,
        direction: &ModelDirection,
    ) -> Result<(f64, f64), AdError> {
        <Self as JvpRule<ModelParameters>>::jvp(self, parameters, direction)
    }

    /// Value, gradient, and independently measured primal/adjoint residuals.
    pub fn value_and_grad_with_report(
        &self,
        parameters: &ModelParameters,
    ) -> Result<SparseSolveReport, AdError> {
        let bound = self.operator.bind(parameters)?;
        let primal = gmres(&bound, &self.right_hand_side, None, self.options)
            .map_err(AdError::PrimalIterative)?;
        let adjoint = gmres(
            &AdjointOperatorView(&bound),
            &self.output_cotangent,
            None,
            self.options,
        )
        .map_err(AdError::AdjointIterative)?;
        let value = real_vector_pairing(&self.output_cotangent, primal.vector())?;
        let gradient = self
            .operator
            .parameter_vjp(primal.vector(), adjoint.vector())?
            .scaled(-1.0)?;
        Ok(SparseSolveReport {
            value,
            gradient,
            primal_iterations: primal.iterations(),
            primal_residual_norm: primal.residual_norm(),
            adjoint_iterations: adjoint.iterations(),
            adjoint_residual_norm: adjoint.residual_norm(),
        })
    }

    /// Value and physical gradient.
    pub fn value_and_grad(
        &self,
        parameters: &ModelParameters,
    ) -> Result<(f64, ModelGradient), AdError> {
        let report = self.value_and_grad_with_report(parameters)?;
        Ok((report.value, report.gradient))
    }
}

impl JvpRule<ModelParameters> for SparseLinearFunctionalObjective<'_> {
    type Output = f64;
    type Error = AdError;

    fn jvp(
        &self,
        parameters: &ModelParameters,
        tangent: &ModelDirection,
    ) -> Result<(f64, f64), Self::Error> {
        let bound = self.operator.bind(parameters)?;
        let primal = gmres(&bound, &self.right_hand_side, None, self.options)
            .map_err(AdError::PrimalIterative)?;
        let value = real_vector_pairing(&self.output_cotangent, primal.vector())?;
        let mut matrix_tangent = vec![Complex64::new(0.0, 0.0); self.operator.dimension()];
        self.operator
            .parameter_jvp_into(tangent, primal.vector(), &mut matrix_tangent)?;
        for value in &mut matrix_tangent {
            *value = -*value;
        }
        let solution_tangent =
            gmres(&bound, &matrix_tangent, None, self.options).map_err(AdError::PrimalIterative)?;
        Ok((
            value,
            real_vector_pairing(&self.output_cotangent, solution_tangent.vector())?,
        ))
    }
}

impl VjpRule<ModelParameters> for SparseLinearFunctionalObjective<'_> {
    type Output = f64;
    type Error = AdError;
    type Pullback<'a>
        = ScalarModelPullback
    where
        Self: 'a,
        ModelParameters: 'a;

    fn vjp<'a>(
        &'a self,
        parameters: &'a ModelParameters,
    ) -> Result<(f64, Self::Pullback<'a>), Self::Error> {
        let (value, gradient) = self.value_and_grad(parameters)?;
        Ok((value, ScalarModelPullback { gradient }))
    }
}

/// Diagnostics for one KPM value-and-gradient evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct KpmMomentReport {
    value: f64,
    gradient: ModelGradient,
    forward_operator_applications: usize,
    recomputed_operator_applications: usize,
    adjoint_operator_applications: usize,
    parameter_contractions: usize,
    checkpoint_count: usize,
    peak_stored_vectors: usize,
}

impl KpmMomentReport {
    /// Scalar Chebyshev objective.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Physical parameter gradient.
    #[must_use]
    pub const fn gradient(&self) -> &ModelGradient {
        &self.gradient
    }

    /// Number of primal sparse operator applications.
    #[must_use]
    pub const fn forward_operator_applications(&self) -> usize {
        self.forward_operator_applications
    }

    /// Number of sparse primal actions used to reconstruct checkpoint
    /// segments during the reverse pass.
    #[must_use]
    pub const fn recomputed_operator_applications(&self) -> usize {
        self.recomputed_operator_applications
    }

    /// Number of reverse sparse adjoint applications.
    #[must_use]
    pub const fn adjoint_operator_applications(&self) -> usize {
        self.adjoint_operator_applications
    }

    /// Number of direct contractions into the physical parameter space.
    #[must_use]
    pub const fn parameter_contractions(&self) -> usize {
        self.parameter_contractions
    }

    /// Number of retained two-vector recurrence checkpoints.
    #[must_use]
    pub const fn checkpoint_count(&self) -> usize {
        self.checkpoint_count
    }

    /// Upper bound on simultaneously retained state vectors.
    #[must_use]
    pub const fn peak_stored_vectors(&self) -> usize {
        self.peak_stored_vectors
    }
}

#[derive(Clone, Debug)]
struct KpmCheckpoint {
    index: usize,
    previous: Vec<Complex64>,
    current: Vec<Complex64>,
}

/// A scalar KPM/Chebyshev objective with a checkpointed reverse recurrence.
///
/// For a fixed probe `r`, this evaluates
///
/// `sum_n coefficients[n] Re(r† T_n(H(theta)) r)`.
///
/// The reverse pass never materializes a dense Hamiltonian and retains
/// `O((N / K + K) * dimension)` state for `N` moments and checkpoint interval
/// `K`, rather than all `N` Chebyshev vectors.
pub struct KpmMomentObjective<'a> {
    operator: &'a SparseAffineOperator,
    probe: Vec<Complex64>,
    coefficients: Vec<f64>,
    checkpoint_interval: usize,
}

impl<'a> KpmMomentObjective<'a> {
    /// Creates a sparse KPM objective.
    pub fn new(
        operator: &'a SparseAffineOperator,
        probe: Vec<Complex64>,
        coefficients: Vec<f64>,
        checkpoint_interval: usize,
    ) -> Result<Self, AdError> {
        validate_vector_len(operator.dimension(), probe.len())?;
        validate_complex_vector(&probe)?;
        if coefficients.is_empty() {
            return Err(AdError::Shape {
                expected: "at least one Chebyshev coefficient".to_owned(),
                actual: "zero coefficients".to_owned(),
            });
        }
        validate_real_coordinates(&coefficients)?;
        if checkpoint_interval == 0 {
            return Err(AdError::Shape {
                expected: "a positive checkpoint interval".to_owned(),
                actual: "zero".to_owned(),
            });
        }
        Ok(Self {
            operator,
            probe,
            coefficients,
            checkpoint_interval,
        })
    }

    /// Number of Chebyshev moments.
    #[must_use]
    pub fn moment_count(&self) -> usize {
        self.coefficients.len()
    }

    /// Scalar primal value.
    pub fn value(&self, parameters: &ModelParameters) -> Result<f64, AdError> {
        Ok(self.forward(parameters)?.0)
    }

    /// Value and directional derivative.
    pub fn jvp(
        &self,
        parameters: &ModelParameters,
        direction: &ModelDirection,
    ) -> Result<(f64, f64), AdError> {
        <Self as JvpRule<ModelParameters>>::jvp(self, parameters, direction)
    }

    /// Value, gradient, and sparse/checkpoint diagnostics.
    pub fn value_and_grad_with_report(
        &self,
        parameters: &ModelParameters,
    ) -> Result<KpmMomentReport, AdError> {
        let (value, checkpoints) = self.forward(parameters)?;
        self.reverse(parameters, value, checkpoints, 1.0)
    }

    /// Value and physical gradient.
    pub fn value_and_grad(
        &self,
        parameters: &ModelParameters,
    ) -> Result<(f64, ModelGradient), AdError> {
        let report = self.value_and_grad_with_report(parameters)?;
        Ok((report.value, report.gradient))
    }

    fn forward(&self, parameters: &ModelParameters) -> Result<(f64, Vec<KpmCheckpoint>), AdError> {
        let bound = self.operator.bind(parameters)?;
        let mut value = self.coefficients[0] * real_vector_pairing(&self.probe, &self.probe)?;
        if self.moment_count() == 1 {
            return Ok((value, Vec::new()));
        }

        let mut previous = self.probe.clone();
        let mut current = bound.apply(&previous)?;
        value += self.coefficients[1] * real_vector_pairing(&self.probe, &current)?;
        let mut checkpoints = vec![KpmCheckpoint {
            index: 1,
            previous: previous.clone(),
            current: current.clone(),
        }];
        let mut index = 1;
        while index + 1 < self.moment_count() {
            let applied = bound.apply(&current)?;
            let next = applied
                .into_iter()
                .zip(&previous)
                .map(|(applied, previous)| applied * 2.0 - previous)
                .collect::<Vec<_>>();
            index += 1;
            value += self.coefficients[index] * real_vector_pairing(&self.probe, &next)?;
            previous = current;
            current = next;
            if (index - 1) % self.checkpoint_interval == 0 {
                checkpoints.push(KpmCheckpoint {
                    index,
                    previous: previous.clone(),
                    current: current.clone(),
                });
            }
        }
        if value.is_finite() {
            Ok((value, checkpoints))
        } else {
            Err(AdError::NonFiniteDerivative)
        }
    }

    fn reverse(
        &self,
        parameters: &ModelParameters,
        value: f64,
        checkpoints: Vec<KpmCheckpoint>,
        output_cotangent: f64,
    ) -> Result<KpmMomentReport, AdError> {
        if !output_cotangent.is_finite() {
            return Err(AdError::NonFiniteDerivative);
        }
        if self.moment_count() == 1 {
            return Ok(KpmMomentReport {
                value,
                gradient: ModelGradient::zeros(self.operator.parameter_count()),
                forward_operator_applications: 0,
                recomputed_operator_applications: 0,
                adjoint_operator_applications: 0,
                parameter_contractions: 0,
                checkpoint_count: 0,
                peak_stored_vectors: 1,
            });
        }

        let bound = self.operator.bind(parameters)?;
        let last = self.moment_count() - 1;
        let mut bar_next = scaled_vector(&self.probe, output_cotangent * self.coefficients[last])?;
        let mut bar_current =
            scaled_vector(&self.probe, output_cotangent * self.coefficients[last - 1])?;
        let mut gradient = ModelGradient::zeros(self.operator.parameter_count());
        let mut peak_segment_vectors = 0;

        for checkpoint_index in (0..checkpoints.len()).rev() {
            let checkpoint = &checkpoints[checkpoint_index];
            let end = checkpoints
                .get(checkpoint_index + 1)
                .map_or(last, |next| next.index);
            let mut segment = vec![checkpoint.current.clone()];
            let mut previous = checkpoint.previous.clone();
            let mut current = checkpoint.current.clone();
            let mut index = checkpoint.index;
            while index < end {
                let applied = bound.apply(&current)?;
                let next = applied
                    .into_iter()
                    .zip(&previous)
                    .map(|(applied, previous)| applied * 2.0 - previous)
                    .collect::<Vec<_>>();
                previous = current;
                current = next;
                index += 1;
                segment.push(current.clone());
            }
            peak_segment_vectors = peak_segment_vectors.max(segment.len());

            for recurrence_index in (checkpoint.index..end).rev() {
                let current_state = &segment[recurrence_index - checkpoint.index];
                let doubled_bar = scaled_vector(&bar_next, 2.0)?;
                gradient.accumulate(&self.operator.parameter_vjp(current_state, &doubled_bar)?)?;

                let adjoint = bound.apply_adjoint(&bar_next)?;
                let updated_current = bar_current
                    .iter()
                    .zip(adjoint)
                    .map(|(current, adjoint)| current + adjoint * 2.0)
                    .collect::<Vec<_>>();
                let new_previous = scaled_vector(
                    &self.probe,
                    output_cotangent * self.coefficients[recurrence_index - 1],
                )?
                .into_iter()
                .zip(&bar_next)
                .map(|(seed, propagated)| seed - propagated)
                .collect::<Vec<_>>();
                bar_next = updated_current;
                bar_current = new_previous;
            }
        }

        gradient.accumulate(&self.operator.parameter_vjp(&self.probe, &bar_next)?)?;
        Ok(KpmMomentReport {
            value,
            gradient,
            forward_operator_applications: self.moment_count() - 1,
            recomputed_operator_applications: self.moment_count().saturating_sub(2),
            adjoint_operator_applications: self.moment_count().saturating_sub(2),
            parameter_contractions: self.moment_count() - 1,
            checkpoint_count: checkpoints.len(),
            peak_stored_vectors: 2 * checkpoints.len() + peak_segment_vectors.max(1),
        })
    }
}

impl JvpRule<ModelParameters> for KpmMomentObjective<'_> {
    type Output = f64;
    type Error = AdError;

    fn jvp(
        &self,
        parameters: &ModelParameters,
        tangent: &ModelDirection,
    ) -> Result<(f64, f64), Self::Error> {
        validate_count(self.operator.parameter_count(), tangent.len())?;
        let bound = self.operator.bind(parameters)?;
        let mut value = self.coefficients[0] * real_vector_pairing(&self.probe, &self.probe)?;
        let mut derivative = 0.0;
        if self.moment_count() == 1 {
            return Ok((value, derivative));
        }

        let mut previous = self.probe.clone();
        let mut previous_tangent = vec![Complex64::new(0.0, 0.0); self.probe.len()];
        let mut current = bound.apply(&previous)?;
        let mut current_tangent = vec![Complex64::new(0.0, 0.0); self.probe.len()];
        self.operator
            .parameter_jvp_into(tangent, &previous, &mut current_tangent)?;
        value += self.coefficients[1] * real_vector_pairing(&self.probe, &current)?;
        derivative += self.coefficients[1] * real_vector_pairing(&self.probe, &current_tangent)?;

        for index in 2..self.moment_count() {
            let applied = bound.apply(&current)?;
            let applied_tangent = bound.apply(&current_tangent)?;
            let mut parameter_tangent = vec![Complex64::new(0.0, 0.0); self.probe.len()];
            self.operator
                .parameter_jvp_into(tangent, &current, &mut parameter_tangent)?;
            let next = applied
                .into_iter()
                .zip(&previous)
                .map(|(applied, previous)| applied * 2.0 - previous)
                .collect::<Vec<_>>();
            let next_tangent = applied_tangent
                .into_iter()
                .zip(parameter_tangent)
                .zip(&previous_tangent)
                .map(|((operator, parameter), previous)| (operator + parameter) * 2.0 - previous)
                .collect::<Vec<_>>();
            value += self.coefficients[index] * real_vector_pairing(&self.probe, &next)?;
            derivative +=
                self.coefficients[index] * real_vector_pairing(&self.probe, &next_tangent)?;
            previous = current;
            current = next;
            previous_tangent = current_tangent;
            current_tangent = next_tangent;
        }
        if value.is_finite() && derivative.is_finite() {
            Ok((value, derivative))
        } else {
            Err(AdError::NonFiniteDerivative)
        }
    }
}

/// One-shot checkpointed KPM pullback.
pub struct KpmMomentPullback<'a> {
    objective: &'a KpmMomentObjective<'a>,
    parameters: &'a ModelParameters,
    value: f64,
    checkpoints: Vec<KpmCheckpoint>,
}

impl Pullback<f64, ModelGradient> for KpmMomentPullback<'_> {
    type Error = AdError;

    fn apply(self, cotangent: f64) -> Result<ModelGradient, Self::Error> {
        Ok(self
            .objective
            .reverse(self.parameters, self.value, self.checkpoints, cotangent)?
            .gradient)
    }
}

impl VjpRule<ModelParameters> for KpmMomentObjective<'_> {
    type Output = f64;
    type Error = AdError;
    type Pullback<'a>
        = KpmMomentPullback<'a>
    where
        Self: 'a,
        ModelParameters: 'a;

    fn vjp<'a>(
        &'a self,
        parameters: &'a ModelParameters,
    ) -> Result<(f64, Self::Pullback<'a>), Self::Error> {
        let (value, checkpoints) = self.forward(parameters)?;
        Ok((
            value,
            KpmMomentPullback {
                objective: self,
                parameters,
                value,
                checkpoints,
            },
        ))
    }
}

struct SeparatedProjector {
    eigenvalues: Vec<f64>,
    eigenvectors: ComplexMatrix,
    projector: ComplexMatrix,
    occupied: usize,
}

fn validate_real_coordinates(values: &[f64]) -> Result<(), AdError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        Err(AdError::NonFiniteParameter { index })
    } else {
        Ok(())
    }
}

fn validate_count(expected: usize, actual: usize) -> Result<(), AdError> {
    if expected == actual {
        Ok(())
    } else {
        Err(AdError::ParameterCount { expected, actual })
    }
}

fn validate_matrix_shape(expected: (usize, usize), actual: (usize, usize)) -> Result<(), AdError> {
    if expected == actual {
        Ok(())
    } else {
        Err(AdError::Shape {
            expected: format!("{}x{}", expected.0, expected.1),
            actual: format!("{}x{}", actual.0, actual.1),
        })
    }
}

fn validate_vector_len(expected: usize, actual: usize) -> Result<(), AdError> {
    if expected == actual {
        Ok(())
    } else {
        Err(AdError::Shape {
            expected: format!("a vector of length {expected}"),
            actual: format!("a vector of length {actual}"),
        })
    }
}

fn validate_complex_vector(values: &[Complex64]) -> Result<(), AdError> {
    if values.iter().all(|value| is_finite_complex(*value)) {
        Ok(())
    } else {
        Err(AdError::NonFiniteDerivative)
    }
}

fn is_finite_complex(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn real_vector_pairing(left: &[Complex64], right: &[Complex64]) -> Result<f64, AdError> {
    validate_vector_len(left.len(), right.len())?;
    let value = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left.conj() * right).re)
        .sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AdError::NonFiniteDerivative)
    }
}

fn scaled_vector(vector: &[Complex64], scale: f64) -> Result<Vec<Complex64>, AdError> {
    if !scale.is_finite() {
        return Err(AdError::NonFiniteDerivative);
    }
    let result = vector.iter().map(|value| value * scale).collect::<Vec<_>>();
    validate_complex_vector(&result)?;
    Ok(result)
}

fn add_scaled_matrix(
    target: &mut ComplexMatrix,
    source: &ComplexMatrix,
    scale: f64,
) -> Result<(), AdError> {
    validate_matrix_shape(target.shape(), source.shape())?;
    if !scale.is_finite() {
        return Err(AdError::NonFiniteDerivative);
    }
    for row in 0..target.rows() {
        for column in 0..target.columns() {
            target.add_entry(row, column, source.get(row, column)? * scale)?;
        }
    }
    Ok(())
}

fn scale_matrix(matrix: &ComplexMatrix, scale: f64) -> Result<ComplexMatrix, AdError> {
    if !scale.is_finite() {
        return Err(AdError::NonFiniteDerivative);
    }
    Ok(ComplexMatrix::new(
        matrix.rows(),
        matrix.columns(),
        matrix
            .as_slice()
            .iter()
            .map(|value| value * scale)
            .collect(),
    )?)
}

fn subtract_matrices(
    left: &ComplexMatrix,
    right: &ComplexMatrix,
) -> Result<ComplexMatrix, AdError> {
    validate_matrix_shape(left.shape(), right.shape())?;
    Ok(ComplexMatrix::new(
        left.rows(),
        left.columns(),
        left.as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(left, right)| left - right)
            .collect(),
    )?)
}

fn validate_linear_solve_arguments(arguments: &LinearSolveArguments) -> Result<(), AdError> {
    if arguments.matrix.rows() == 0 || arguments.matrix.rows() != arguments.matrix.columns() {
        return Err(AdError::Shape {
            expected: "a nonempty square matrix".to_owned(),
            actual: format!("{}x{}", arguments.matrix.rows(), arguments.matrix.columns()),
        });
    }
    validate_vector_len(arguments.matrix.rows(), arguments.right_hand_side.len())?;
    validate_complex_vector(&arguments.right_hand_side)
}

fn validate_surface_green_tangent(
    arguments: &SurfaceGreenArguments,
    tangent: &SurfaceGreenTangent,
) -> Result<(), AdError> {
    validate_matrix_shape(
        arguments.cell_hamiltonian.shape(),
        tangent.cell_hamiltonian.shape(),
    )?;
    validate_matrix_shape(
        arguments.inter_cell_hopping.shape(),
        tangent.inter_cell_hopping.shape(),
    )?;
    if !tangent.cell_hamiltonian.is_hermitian(HERMITIAN_TOLERANCE)? {
        return Err(AdError::NonHermitian);
    }
    if !tangent.energy.is_finite() || !tangent.broadening.is_finite() {
        return Err(AdError::NonFiniteDerivative);
    }
    Ok(())
}

fn validate_differentiable_open_system(system: &DifferentiableOpenSystem) -> Result<(), AdError> {
    let dimension = system.device_hamiltonian.rows();
    if dimension == 0 || system.device_hamiltonian.columns() != dimension {
        return Err(AdError::Shape {
            expected: "a nonempty square device Hamiltonian".to_owned(),
            actual: format!(
                "{}x{}",
                system.device_hamiltonian.rows(),
                system.device_hamiltonian.columns()
            ),
        });
    }
    if !system
        .device_hamiltonian
        .is_hermitian(HERMITIAN_TOLERANCE)?
    {
        return Err(AdError::NonHermitian);
    }
    if system.leads.is_empty() || !system.energy.is_finite() {
        return Err(AdError::Shape {
            expected: "at least one lead and a finite energy".to_owned(),
            actual: format!("{} leads at energy {}", system.leads.len(), system.energy),
        });
    }
    for lead in &system.leads {
        let lead_dimension = lead.cell_hamiltonian.rows();
        if lead_dimension == 0
            || lead.cell_hamiltonian.columns() != lead_dimension
            || lead.inter_cell_hopping.shape() != (lead_dimension, lead_dimension)
            || lead.coupling.shape() != (dimension, lead_dimension)
            || !lead.broadening.is_finite()
            || lead.broadening <= 0.0
        {
            return Err(AdError::Shape {
                expected: format!("a square lead and a {dimension}x{lead_dimension} coupling"),
                actual: format!(
                    "cell {}x{}, hopping {}x{}, coupling {}x{}",
                    lead.cell_hamiltonian.rows(),
                    lead.cell_hamiltonian.columns(),
                    lead.inter_cell_hopping.rows(),
                    lead.inter_cell_hopping.columns(),
                    lead.coupling.rows(),
                    lead.coupling.columns()
                ),
            });
        }
        if !lead.cell_hamiltonian.is_hermitian(HERMITIAN_TOLERANCE)? {
            return Err(AdError::NonHermitian);
        }
    }
    Ok(())
}

fn validate_open_system_direction(
    system: &DifferentiableOpenSystem,
    direction: &OpenSystemDirection,
) -> Result<(), AdError> {
    validate_differentiable_open_system(system)?;
    validate_matrix_shape(
        system.device_hamiltonian.shape(),
        direction.device_hamiltonian.shape(),
    )?;
    if !direction
        .device_hamiltonian
        .is_hermitian(HERMITIAN_TOLERANCE)?
    {
        return Err(AdError::NonHermitian);
    }
    if !direction.energy.is_finite() {
        return Err(AdError::NonFiniteDerivative);
    }
    validate_count(system.leads.len(), direction.leads.len())?;
    for (lead, tangent) in system.leads.iter().zip(&direction.leads) {
        validate_matrix_shape(
            lead.cell_hamiltonian.shape(),
            tangent.cell_hamiltonian.shape(),
        )?;
        validate_matrix_shape(
            lead.inter_cell_hopping.shape(),
            tangent.inter_cell_hopping.shape(),
        )?;
        validate_matrix_shape(lead.coupling.shape(), tangent.coupling.shape())?;
        if !tangent.cell_hamiltonian.is_hermitian(HERMITIAN_TOLERANCE)? {
            return Err(AdError::NonHermitian);
        }
        if !tangent.broadening.is_finite() {
            return Err(AdError::NonFiniteDerivative);
        }
    }
    Ok(())
}

fn scale_open_system_gradient(
    gradient: OpenSystemGradient,
    scale: f64,
) -> Result<OpenSystemGradient, AdError> {
    if !scale.is_finite() {
        return Err(AdError::NonFiniteDerivative);
    }
    Ok(OpenSystemGradient {
        device_hamiltonian: scale_matrix(&gradient.device_hamiltonian, scale)?,
        leads: gradient
            .leads
            .into_iter()
            .map(|lead| {
                Ok(LeadGradient {
                    cell_hamiltonian: scale_matrix(&lead.cell_hamiltonian, scale)?,
                    inter_cell_hopping: scale_matrix(&lead.inter_cell_hopping, scale)?,
                    coupling: scale_matrix(&lead.coupling, scale)?,
                    broadening: lead.broadening * scale,
                })
            })
            .collect::<Result<Vec<_>, AdError>>()?,
        energy: gradient.energy * scale,
    })
}

fn caroli_input_cotangents(
    device_hamiltonian: &ComplexMatrix,
    self_energies: &[ComplexMatrix],
    energy: f64,
    drain: usize,
    source: usize,
) -> Result<(f64, ComplexMatrix, Vec<ComplexMatrix>, f64), AdError> {
    let solution = solve_open_system_from_self_energies(device_hamiltonian, self_energies, energy)?;
    let value = solution.transmission(drain, source)?;
    let green = to_backend(solution.retarded_green());
    let gamma_drain = to_backend(&solution.broadenings()[drain]);
    let gamma_source = to_backend(&solution.broadenings()[source]);
    let common = green.adjoint()
        * &gamma_drain
        * &green
        * &gamma_source
        * green.adjoint()
        * Complex64::new(2.0, 0.0);
    let device_cotangent = from_backend(&hermitian_part(&common))?;
    let mut self_energy_cotangents = vec![common.clone(); self_energies.len()];
    self_energy_cotangents[drain] +=
        &green * &gamma_source * green.adjoint() * Complex64::new(0.0, -2.0);
    self_energy_cotangents[source] +=
        green.adjoint() * &gamma_drain * &green * Complex64::new(0.0, -2.0);
    let energy_cotangent = -common.trace().re;
    Ok((
        value,
        device_cotangent,
        self_energy_cotangents
            .iter()
            .map(from_backend)
            .collect::<Result<Vec<_>, _>>()?,
        energy_cotangent,
    ))
}

fn solve_surface_green_linearization(
    green: &DMatrix<Complex64>,
    hopping: &DMatrix<Complex64>,
    right_hand_side: &DMatrix<Complex64>,
    adjoint: bool,
) -> Result<DMatrix<Complex64>, AdError> {
    let dimension = green.nrows();
    if dimension == 0
        || green.ncols() != dimension
        || hopping.shape() != (dimension, dimension)
        || right_hand_side.shape() != (dimension, dimension)
    {
        return Err(AdError::Shape {
            expected: format!("three {dimension}x{dimension} matrices"),
            actual: format!(
                "green {}x{}, hopping {}x{}, right-hand side {}x{}",
                green.nrows(),
                green.ncols(),
                hopping.nrows(),
                hopping.ncols(),
                right_hand_side.nrows(),
                right_hand_side.ncols()
            ),
        });
    }
    let inverse = green.clone().try_inverse().ok_or(AdError::SingularPrimal)?;
    let coordinate_count = dimension * dimension;
    let mut operator = DMatrix::<Complex64>::zeros(coordinate_count, coordinate_count);
    for input_row in 0..dimension {
        for input_column in 0..dimension {
            let input_index = input_row * dimension + input_column;
            let mut basis = DMatrix::<Complex64>::zeros(dimension, dimension);
            basis[(input_row, input_column)] = Complex64::new(1.0, 0.0);
            let applied = if adjoint {
                inverse.adjoint() * &basis * inverse.adjoint()
                    - hopping.adjoint() * &basis * hopping
            } else {
                &inverse * &basis * &inverse - hopping * &basis * hopping.adjoint()
            };
            for output_row in 0..dimension {
                for output_column in 0..dimension {
                    let output_index = output_row * dimension + output_column;
                    operator[(output_index, input_index)] = applied[(output_row, output_column)];
                }
            }
        }
    }
    let right_hand_side = DVector::from_iterator(
        coordinate_count,
        (0..dimension)
            .flat_map(|row| (0..dimension).map(move |column| right_hand_side[(row, column)])),
    );
    let solution = operator.lu().solve(&right_hand_side).ok_or(if adjoint {
        AdError::SingularAdjoint
    } else {
        AdError::SingularPrimal
    })?;
    let matrix = DMatrix::from_fn(dimension, dimension, |row, column| {
        solution[row * dimension + column]
    });
    if matrix.iter().all(|value| is_finite_complex(*value)) {
        Ok(matrix)
    } else {
        Err(AdError::NonFiniteDerivative)
    }
}

fn inverse(matrix: &ComplexMatrix, adjoint: bool) -> Result<ComplexMatrix, AdError> {
    if matrix.rows() == 0 || matrix.rows() != matrix.columns() {
        return Err(AdError::Shape {
            expected: "a nonempty square matrix".to_owned(),
            actual: format!("{}x{}", matrix.rows(), matrix.columns()),
        });
    }
    let backend = if adjoint {
        to_backend(matrix).adjoint()
    } else {
        to_backend(matrix)
    };
    let inverse = backend.try_inverse().ok_or(if adjoint {
        AdError::SingularAdjoint
    } else {
        AdError::SingularPrimal
    })?;
    from_backend(&inverse)
}

fn apply_dense(matrix: &ComplexMatrix, vector: &[Complex64]) -> Result<Vec<Complex64>, AdError> {
    validate_vector_len(matrix.columns(), vector.len())?;
    let output = (0..matrix.rows())
        .map(|row| {
            (0..matrix.columns())
                .map(|column| matrix.as_slice()[row * matrix.columns() + column] * vector[column])
                .sum()
        })
        .collect::<Vec<_>>();
    validate_complex_vector(&output)?;
    Ok(output)
}

fn isolated_eigenvalue_gradient(
    matrix: &ComplexMatrix,
    index: usize,
    minimum_gap: f64,
) -> Result<(f64, ComplexMatrix), AdError> {
    if matrix.rows() == 0 || matrix.rows() != matrix.columns() {
        return Err(AdError::InvalidSubspace);
    }
    let eigensystem = hermitian_eigensystem(matrix, HERMITIAN_TOLERANCE)?;
    if index >= eigensystem.eigenvalues().len() {
        return Err(AdError::InvalidSubspace);
    }
    let gap = adjacent_gap(eigensystem.eigenvalues(), index);
    if gap < minimum_gap {
        return Err(AdError::GapTooSmall {
            gap,
            minimum: minimum_gap,
        });
    }
    let dimension = matrix.rows();
    let mut gradient = ComplexMatrix::zeros(dimension, dimension);
    for row in 0..dimension {
        let left = eigensystem.eigenvectors().as_slice()[row * dimension + index];
        for column in 0..dimension {
            let right = eigensystem.eigenvectors().as_slice()[column * dimension + index];
            gradient.set(row, column, left * right.conj())?;
        }
    }
    Ok((eigensystem.eigenvalues()[index], gradient))
}

fn adjacent_gap(eigenvalues: &[f64], index: usize) -> f64 {
    if eigenvalues.len() <= 1 {
        return f64::INFINITY;
    }
    let lower = if index > 0 {
        eigenvalues[index] - eigenvalues[index - 1]
    } else {
        f64::INFINITY
    };
    let upper = if index + 1 < eigenvalues.len() {
        eigenvalues[index + 1] - eigenvalues[index]
    } else {
        f64::INFINITY
    };
    lower.min(upper)
}

fn validate_projector(projector: &ComplexMatrix) -> Result<(), AdError> {
    if !projector.is_hermitian(HERMITIAN_TOLERANCE)? {
        return Err(AdError::InvalidProjector);
    }
    let backend = to_backend(projector);
    let residual = &backend * &backend - backend;
    let norm = residual.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    if norm <= HERMITIAN_TOLERANCE * projector.rows() as f64 {
        Ok(())
    } else {
        Err(AdError::InvalidProjector)
    }
}

fn separated_projector(
    matrix: &ComplexMatrix,
    occupied: usize,
    minimum_gap: f64,
) -> Result<SeparatedProjector, AdError> {
    if matrix.rows() == 0
        || matrix.rows() != matrix.columns()
        || occupied == 0
        || occupied >= matrix.rows()
    {
        return Err(AdError::InvalidSubspace);
    }
    let eigensystem = hermitian_eigensystem(matrix, HERMITIAN_TOLERANCE)?;
    let gap = eigensystem.eigenvalues()[occupied] - eigensystem.eigenvalues()[occupied - 1];
    if gap < minimum_gap {
        return Err(AdError::GapTooSmall {
            gap,
            minimum: minimum_gap,
        });
    }
    let dimension = matrix.rows();
    let mut projector = ComplexMatrix::zeros(dimension, dimension);
    for state in 0..occupied {
        for row in 0..dimension {
            let left = eigensystem.eigenvectors().as_slice()[row * dimension + state];
            for column in 0..dimension {
                let right = eigensystem.eigenvectors().as_slice()[column * dimension + state];
                projector.add_entry(row, column, left * right.conj())?;
            }
        }
    }
    Ok(SeparatedProjector {
        eigenvalues: eigensystem.eigenvalues().to_vec(),
        eigenvectors: eigensystem.eigenvectors().clone(),
        projector,
        occupied,
    })
}

fn projector_input_cotangent(
    decomposition: &SeparatedProjector,
    projector_cotangent: &ComplexMatrix,
) -> Result<ComplexMatrix, AdError> {
    let dimension = decomposition.eigenvalues.len();
    validate_matrix_shape((dimension, dimension), projector_cotangent.shape())?;
    let unitary = to_backend(&decomposition.eigenvectors);
    let cotangent = hermitian_part(&to_backend(projector_cotangent));
    let eigenbasis = unitary.adjoint() * cotangent * &unitary;
    let mut matrix_cotangent = DMatrix::<Complex64>::zeros(dimension, dimension);
    for occupied in 0..decomposition.occupied {
        for empty in decomposition.occupied..dimension {
            let denominator =
                decomposition.eigenvalues[occupied] - decomposition.eigenvalues[empty];
            let value = eigenbasis[(empty, occupied)] / denominator;
            matrix_cotangent[(empty, occupied)] = value;
            matrix_cotangent[(occupied, empty)] = value.conj();
        }
    }
    from_backend(&(unitary.clone() * matrix_cotangent * unitary.adjoint()))
}

fn hermitian_part(matrix: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    (matrix + matrix.adjoint()) * Complex64::new(0.5, 0.0)
}

fn find_csr_entry(matrix: &CsrMatrix, row: usize, column: usize) -> Result<usize, AdError> {
    let range = matrix.row_offsets()[row]..matrix.row_offsets()[row + 1];
    matrix.column_indices()[range.clone()]
        .binary_search(&column)
        .map(|index| range.start + index)
        .map_err(|_| AdError::Shape {
            expected: format!("an explicit CSR entry at ({row}, {column})"),
            actual: "no stored entry".to_owned(),
        })
}

fn to_backend(matrix: &ComplexMatrix) -> DMatrix<Complex64> {
    DMatrix::from_row_slice(matrix.rows(), matrix.columns(), matrix.as_slice())
}

fn from_backend(matrix: &DMatrix<Complex64>) -> Result<ComplexMatrix, AdError> {
    Ok(ComplexMatrix::new(
        matrix.nrows(),
        matrix.ncols(),
        (0..matrix.nrows())
            .flat_map(|row| (0..matrix.ncols()).map(move |column| matrix[(row, column)]))
            .collect(),
    )?)
}
