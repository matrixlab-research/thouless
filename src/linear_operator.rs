//! Matrix-free linear operators and a canonical CSR implementation.

use std::error::Error;
use std::fmt;

use crate::{Complex64, ComplexMatrix};

/// Errors raised while constructing or applying a linear operator.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinearOperatorError {
    /// A matrix must have at least one row and one column.
    EmptyShape {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// An operation requires a square matrix.
    NonSquare {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// CSR storage requires one row offset per row plus a terminal offset.
    InvalidRowOffsetCount {
        /// Required number of offsets.
        expected: usize,
        /// Supplied number of offsets.
        actual: usize,
    },
    /// The first CSR row offset must be zero.
    NonzeroFirstRowOffset {
        /// Supplied first offset.
        actual: usize,
    },
    /// CSR row offsets must be monotone.
    NonmonotoneRowOffsets {
        /// Row whose terminal offset is smaller than its initial offset.
        row: usize,
    },
    /// The terminal CSR offset must equal the number of stored entries.
    InvalidTerminalRowOffset {
        /// Number of stored entries.
        expected: usize,
        /// Supplied terminal offset.
        actual: usize,
    },
    /// CSR column indices and values must have the same length.
    InvalidStoredEntryCount {
        /// Number of column indices.
        indices: usize,
        /// Number of values.
        values: usize,
    },
    /// A CSR column index lies outside the matrix.
    ColumnOutOfBounds {
        /// Row containing the invalid index.
        row: usize,
        /// Supplied column index.
        column: usize,
        /// Number of matrix columns.
        columns: usize,
    },
    /// Columns within each CSR row must be strictly increasing.
    NoncanonicalRow {
        /// Row containing an unsorted or duplicate column.
        row: usize,
        /// Previous column index.
        previous: usize,
        /// Current column index.
        current: usize,
    },
    /// A matrix or vector value is NaN or infinity.
    NonFiniteValue,
    /// An input vector has an incompatible length.
    InputDimension {
        /// Required input length.
        expected: usize,
        /// Supplied input length.
        actual: usize,
    },
    /// An output vector has an incompatible length.
    OutputDimension {
        /// Required output length.
        expected: usize,
        /// Supplied output length.
        actual: usize,
    },
    /// A tolerance is negative or non-finite.
    InvalidTolerance,
    /// An incomplete factorization requires an explicit diagonal entry.
    MissingDiagonal {
        /// Row without an explicit diagonal entry.
        row: usize,
    },
    /// Explicit dense materialization would overflow the addressable size.
    DenseSizeOverflow {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
}

impl fmt::Display for LinearOperatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyShape { rows, columns } => {
                write!(formatter, "linear operator shape {rows}x{columns} is empty")
            }
            Self::NonSquare { rows, columns } => {
                write!(
                    formatter,
                    "linear operator shape {rows}x{columns} is not square"
                )
            }
            Self::InvalidRowOffsetCount { expected, actual } => write!(
                formatter,
                "CSR storage has {actual} row offsets; expected {expected}"
            ),
            Self::NonzeroFirstRowOffset { actual } => {
                write!(formatter, "first CSR row offset is {actual}; expected zero")
            }
            Self::NonmonotoneRowOffsets { row } => {
                write!(formatter, "CSR row offsets decrease at row {row}")
            }
            Self::InvalidTerminalRowOffset { expected, actual } => write!(
                formatter,
                "terminal CSR row offset is {actual}; expected {expected}"
            ),
            Self::InvalidStoredEntryCount { indices, values } => write!(
                formatter,
                "CSR storage has {indices} column indices but {values} values"
            ),
            Self::ColumnOutOfBounds {
                row,
                column,
                columns,
            } => write!(
                formatter,
                "CSR entry ({row}, {column}) is outside a matrix with {columns} columns"
            ),
            Self::NoncanonicalRow {
                row,
                previous,
                current,
            } => write!(
                formatter,
                "CSR row {row} has non-increasing columns {previous}, {current}"
            ),
            Self::NonFiniteValue => {
                write!(formatter, "linear operator contains a non-finite value")
            }
            Self::InputDimension { expected, actual } => write!(
                formatter,
                "linear-operator input has length {actual}; expected {expected}"
            ),
            Self::OutputDimension { expected, actual } => write!(
                formatter,
                "linear-operator output has length {actual}; expected {expected}"
            ),
            Self::InvalidTolerance => write!(
                formatter,
                "operator tolerance must be finite and nonnegative"
            ),
            Self::MissingDiagonal { row } => {
                write!(
                    formatter,
                    "incomplete factorization requires a diagonal entry in row {row}"
                )
            }
            Self::DenseSizeOverflow { rows, columns } => write!(
                formatter,
                "cannot materialize a {rows}x{columns} linear operator"
            ),
        }
    }
}

impl Error for LinearOperatorError {}

/// A matrix-free complex linear operator.
///
/// Implementations write into caller-owned output storage so iterative
/// algorithms can reuse allocations.
pub trait LinearOperator {
    /// Number of output components.
    fn rows(&self) -> usize;

    /// Number of input components.
    fn columns(&self) -> usize;

    /// Applies the operator to `input`, replacing all entries of `output`.
    fn apply_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), LinearOperatorError>;

    /// Applies the operator and returns a newly allocated result.
    fn apply(&self, input: &[Complex64]) -> Result<Vec<Complex64>, LinearOperatorError> {
        let mut output = vec![Complex64::new(0.0, 0.0); self.rows()];
        self.apply_into(input, &mut output)?;
        Ok(output)
    }

    /// Returns the matrix shape.
    fn shape(&self) -> (usize, usize) {
        (self.rows(), self.columns())
    }
}

/// Numerical controls for restarted GMRES.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GmresOptions {
    /// Relative residual tolerance with respect to the right-hand side norm.
    pub relative_tolerance: f64,
    /// Absolute residual tolerance.
    pub absolute_tolerance: f64,
    /// Maximum Krylov-space dimension before restarting.
    pub restart: usize,
    /// Maximum total number of operator applications.
    pub max_iterations: usize,
}

impl Default for GmresOptions {
    fn default() -> Self {
        Self {
            relative_tolerance: 1.0e-10,
            absolute_tolerance: 1.0e-12,
            restart: 64,
            max_iterations: 4096,
        }
    }
}

/// A converged GMRES solution and its true residual.
#[derive(Clone, Debug, PartialEq)]
pub struct GmresSolution {
    vector: Vec<Complex64>,
    iterations: usize,
    residual_norm: f64,
}

impl GmresSolution {
    /// Returns the solution vector.
    #[must_use]
    pub fn vector(&self) -> &[Complex64] {
        &self.vector
    }

    /// Number of Krylov iterations used across all restarts.
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Euclidean norm of the true, unpreconditioned residual.
    #[must_use]
    pub const fn residual_norm(&self) -> f64 {
        self.residual_norm
    }

    /// Consumes the report and returns the solution vector.
    #[must_use]
    pub fn into_vector(self) -> Vec<Complex64> {
        self.vector
    }
}

/// Errors raised by an iterative linear solve.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IterativeSolveError {
    /// The operator must be nonempty and square.
    InvalidOperatorShape,
    /// A right preconditioner must match the square operator.
    InvalidPreconditionerShape,
    /// Solver tolerances or iteration budgets are invalid.
    InvalidOptions,
    /// The initial vector or right-hand side has an incompatible length.
    InvalidVectorDimension,
    /// The Arnoldi Hessenberg system became singular.
    SingularKrylovSystem,
    /// The requested residual tolerance was not reached.
    NoConvergence {
        /// Number of completed Krylov iterations.
        iterations: usize,
    },
    /// Applying the linear operator failed.
    Operator(LinearOperatorError),
}

impl fmt::Display for IterativeSolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOperatorShape => {
                write!(formatter, "GMRES requires a nonempty square operator")
            }
            Self::InvalidPreconditionerShape => {
                write!(
                    formatter,
                    "GMRES right preconditioner must match the operator shape"
                )
            }
            Self::InvalidOptions => write!(formatter, "GMRES options are invalid"),
            Self::InvalidVectorDimension => {
                write!(
                    formatter,
                    "GMRES vector dimensions do not match the operator"
                )
            }
            Self::SingularKrylovSystem => {
                write!(formatter, "GMRES produced a singular Krylov system")
            }
            Self::NoConvergence { iterations } => {
                write!(
                    formatter,
                    "GMRES did not converge after {iterations} iterations"
                )
            }
            Self::Operator(error) => error.fmt(formatter),
        }
    }
}

impl Error for IterativeSolveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Operator(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LinearOperatorError> for IterativeSolveError {
    fn from(error: LinearOperatorError) -> Self {
        Self::Operator(error)
    }
}

/// Solves a square complex linear system with restarted GMRES.
///
/// The implementation uses twice-reorthogonalized modified Gram-Schmidt and
/// complex Givens rotations. It never materializes the operator and reports
/// convergence using the true residual rather than only the Hessenberg
/// estimate.
pub fn gmres<O: LinearOperator>(
    operator: &O,
    right_hand_side: &[Complex64],
    initial: Option<&[Complex64]>,
    options: GmresOptions,
) -> Result<GmresSolution, IterativeSolveError> {
    gmres_impl(operator, None, right_hand_side, initial, options)
}

/// Solves a square complex system with right-preconditioned restarted GMRES.
///
/// `right_preconditioner` applies an approximation to the inverse operator.
/// Right preconditioning keeps convergence tests in the original,
/// unpreconditioned residual norm.
pub fn gmres_with_right_preconditioner<O: LinearOperator, P: LinearOperator>(
    operator: &O,
    right_preconditioner: &P,
    right_hand_side: &[Complex64],
    initial: Option<&[Complex64]>,
    options: GmresOptions,
) -> Result<GmresSolution, IterativeSolveError> {
    gmres_impl(
        operator,
        Some(right_preconditioner),
        right_hand_side,
        initial,
        options,
    )
}

fn gmres_impl<O: LinearOperator>(
    operator: &O,
    right_preconditioner: Option<&dyn LinearOperator>,
    right_hand_side: &[Complex64],
    initial: Option<&[Complex64]>,
    options: GmresOptions,
) -> Result<GmresSolution, IterativeSolveError> {
    let dimension = operator.rows();
    if dimension == 0 || operator.columns() != dimension {
        return Err(IterativeSolveError::InvalidOperatorShape);
    }
    if right_preconditioner
        .is_some_and(|value| value.rows() != dimension || value.columns() != dimension)
    {
        return Err(IterativeSolveError::InvalidPreconditionerShape);
    }
    if right_hand_side.len() != dimension || initial.is_some_and(|value| value.len() != dimension) {
        return Err(IterativeSolveError::InvalidVectorDimension);
    }
    if !options.relative_tolerance.is_finite()
        || options.relative_tolerance <= 0.0
        || !options.absolute_tolerance.is_finite()
        || options.absolute_tolerance < 0.0
        || options.restart == 0
        || options.max_iterations == 0
    {
        return Err(IterativeSolveError::InvalidOptions);
    }

    let mut solution = initial.map_or_else(
        || vec![Complex64::new(0.0, 0.0); dimension],
        <[Complex64]>::to_vec,
    );
    let right_norm = l2_norm(right_hand_side);
    let target = options.absolute_tolerance + options.relative_tolerance * right_norm;
    let mut total_iterations = 0;

    loop {
        let residual = true_residual(operator, right_hand_side, &solution)?;
        let residual_norm = l2_norm(&residual);
        if residual_norm <= target {
            return Ok(GmresSolution {
                vector: solution,
                iterations: total_iterations,
                residual_norm,
            });
        }
        if total_iterations >= options.max_iterations {
            return Err(IterativeSolveError::NoConvergence {
                iterations: total_iterations,
            });
        }

        let cycle_size = options
            .restart
            .min(options.max_iterations - total_iterations)
            .min(dimension);
        let mut basis = Vec::with_capacity(cycle_size + 1);
        let mut preconditioned_basis = Vec::with_capacity(cycle_size);
        basis.push(
            residual
                .iter()
                .map(|value| *value / residual_norm)
                .collect::<Vec<_>>(),
        );
        let mut hessenberg = vec![vec![Complex64::new(0.0, 0.0); cycle_size]; cycle_size + 1];
        let mut cosines = vec![0.0; cycle_size];
        let mut sines = vec![Complex64::new(0.0, 0.0); cycle_size];
        let mut transformed_residual = vec![Complex64::new(0.0, 0.0); cycle_size + 1];
        transformed_residual[0] = Complex64::new(residual_norm, 0.0);
        let cycle_origin = solution.clone();
        let mut restarted = false;

        for column in 0..cycle_size {
            let preconditioned = right_preconditioner.map_or_else(
                || Ok(basis[column].clone()),
                |value| value.apply(&basis[column]),
            )?;
            let mut work = operator.apply(&preconditioned)?;
            preconditioned_basis.push(preconditioned);
            for _ in 0..2 {
                for row in 0..=column {
                    let projection = conjugate_dot(&basis[row], &work);
                    hessenberg[row][column] += projection;
                    for (value, basis_value) in work.iter_mut().zip(&basis[row]) {
                        *value -= projection * basis_value;
                    }
                }
            }
            let next_norm = l2_norm(&work);
            hessenberg[column + 1][column] = Complex64::new(next_norm, 0.0);
            let breakdown =
                next_norm <= f64::EPSILON.sqrt() * hessenberg[column][column].norm().max(1.0);
            if !breakdown {
                basis.push(work.iter().map(|value| *value / next_norm).collect());
            }

            for rotation in 0..column {
                let upper = hessenberg[rotation][column];
                let lower = hessenberg[rotation + 1][column];
                hessenberg[rotation][column] = upper * cosines[rotation] + lower * sines[rotation];
                hessenberg[rotation + 1][column] =
                    -upper * sines[rotation].conj() + lower * cosines[rotation];
            }
            let (cosine, sine, diagonal) =
                complex_givens(hessenberg[column][column], hessenberg[column + 1][column]);
            cosines[column] = cosine;
            sines[column] = sine;
            hessenberg[column][column] = diagonal;
            hessenberg[column + 1][column] = Complex64::new(0.0, 0.0);

            let upper = transformed_residual[column];
            let lower = transformed_residual[column + 1];
            transformed_residual[column] = upper * cosine + lower * sine;
            transformed_residual[column + 1] = -upper * sine.conj() + lower * cosine;
            total_iterations += 1;

            let used = column + 1;
            let estimated_residual = transformed_residual[used].norm();
            if estimated_residual <= target
                || breakdown
                || used == cycle_size
                || total_iterations == options.max_iterations
            {
                let coefficients = back_substitute(&hessenberg, &transformed_residual, used)?;
                solution.clone_from(&cycle_origin);
                for (basis_vector, coefficient) in
                    preconditioned_basis.iter().take(used).zip(coefficients)
                {
                    for (value, basis_value) in solution.iter_mut().zip(basis_vector) {
                        *value += coefficient * basis_value;
                    }
                }
                let true_value = l2_norm(&true_residual(operator, right_hand_side, &solution)?);
                if true_value <= target {
                    return Ok(GmresSolution {
                        vector: solution,
                        iterations: total_iterations,
                        residual_norm: true_value,
                    });
                }
                if breakdown || total_iterations == options.max_iterations {
                    return Err(IterativeSolveError::NoConvergence {
                        iterations: total_iterations,
                    });
                }
                restarted = true;
                break;
            }
        }
        if !restarted {
            return Err(IterativeSolveError::NoConvergence {
                iterations: total_iterations,
            });
        }
    }
}

fn true_residual<O: LinearOperator>(
    operator: &O,
    right_hand_side: &[Complex64],
    solution: &[Complex64],
) -> Result<Vec<Complex64>, IterativeSolveError> {
    let applied = operator.apply(solution)?;
    Ok(right_hand_side
        .iter()
        .zip(applied)
        .map(|(right, left)| *right - left)
        .collect())
}

fn l2_norm(vector: &[Complex64]) -> f64 {
    vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

fn conjugate_dot(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.conj() * right)
        .sum()
}

fn complex_givens(upper: Complex64, lower: Complex64) -> (f64, Complex64, Complex64) {
    let lower_norm = lower.norm();
    if lower_norm == 0.0 {
        return (1.0, Complex64::new(0.0, 0.0), upper);
    }
    let upper_norm = upper.norm();
    if upper_norm == 0.0 {
        let sine = lower.conj() / lower_norm;
        return (0.0, sine, Complex64::new(lower_norm, 0.0));
    }
    let scale = upper_norm + lower_norm;
    let norm = scale * ((upper_norm / scale).powi(2) + (lower_norm / scale).powi(2)).sqrt();
    let phase = upper / upper_norm;
    (upper_norm / norm, phase * lower.conj() / norm, phase * norm)
}

fn back_substitute(
    upper: &[Vec<Complex64>],
    right: &[Complex64],
    dimension: usize,
) -> Result<Vec<Complex64>, IterativeSolveError> {
    let mut result = vec![Complex64::new(0.0, 0.0); dimension];
    for row in (0..dimension).rev() {
        let remainder = ((row + 1)..dimension)
            .map(|column| upper[row][column] * result[column])
            .sum::<Complex64>();
        let diagonal = upper[row][row];
        if diagonal.norm() <= f64::EPSILON {
            return Err(IterativeSolveError::SingularKrylovSystem);
        }
        result[row] = (right[row] - remainder) / diagonal;
    }
    Ok(result)
}

impl LinearOperator for ComplexMatrix {
    fn rows(&self) -> usize {
        self.rows()
    }

    fn columns(&self) -> usize {
        self.columns()
    }

    fn apply_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), LinearOperatorError> {
        validate_vectors(self.rows(), self.columns(), input, output)?;
        for (row, result) in output.iter_mut().enumerate() {
            *result = (0..self.columns())
                .map(|column| self.as_slice()[row * self.columns() + column] * input[column])
                .sum();
        }
        validate_finite_output(output)
    }
}

/// An owned canonical compressed-sparse-row complex matrix.
///
/// Column indices in every row are strictly increasing. This makes structural
/// validation, Hermiticity checks, and deterministic multiplication possible
/// without hidden normalization work.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix {
    rows: usize,
    columns: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<Complex64>,
}

impl CsrMatrix {
    /// Creates a canonical CSR matrix.
    pub fn new(
        rows: usize,
        columns: usize,
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<Complex64>,
    ) -> Result<Self, LinearOperatorError> {
        if rows == 0 || columns == 0 {
            return Err(LinearOperatorError::EmptyShape { rows, columns });
        }
        let expected_offsets = rows
            .checked_add(1)
            .ok_or(LinearOperatorError::DenseSizeOverflow { rows, columns })?;
        if row_offsets.len() != expected_offsets {
            return Err(LinearOperatorError::InvalidRowOffsetCount {
                expected: expected_offsets,
                actual: row_offsets.len(),
            });
        }
        if row_offsets[0] != 0 {
            return Err(LinearOperatorError::NonzeroFirstRowOffset {
                actual: row_offsets[0],
            });
        }
        if column_indices.len() != values.len() {
            return Err(LinearOperatorError::InvalidStoredEntryCount {
                indices: column_indices.len(),
                values: values.len(),
            });
        }
        for row in 0..rows {
            if row_offsets[row] > row_offsets[row + 1] {
                return Err(LinearOperatorError::NonmonotoneRowOffsets { row });
            }
        }
        if row_offsets[rows] != values.len() {
            return Err(LinearOperatorError::InvalidTerminalRowOffset {
                expected: values.len(),
                actual: row_offsets[rows],
            });
        }
        if values
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(LinearOperatorError::NonFiniteValue);
        }
        for row in 0..rows {
            let range = row_offsets[row]..row_offsets[row + 1];
            let mut previous = None;
            for &column in &column_indices[range] {
                if column >= columns {
                    return Err(LinearOperatorError::ColumnOutOfBounds {
                        row,
                        column,
                        columns,
                    });
                }
                if let Some(previous) = previous {
                    if column <= previous {
                        return Err(LinearOperatorError::NoncanonicalRow {
                            row,
                            previous,
                            current: column,
                        });
                    }
                }
                previous = Some(column);
            }
        }
        Ok(Self {
            rows,
            columns,
            row_offsets,
            column_indices,
            values,
        })
    }

    /// Converts a dense matrix, dropping entries no larger than `zero_tolerance`.
    pub fn from_dense(
        matrix: &ComplexMatrix,
        zero_tolerance: f64,
    ) -> Result<Self, LinearOperatorError> {
        if !zero_tolerance.is_finite() || zero_tolerance < 0.0 {
            return Err(LinearOperatorError::InvalidTolerance);
        }
        let mut row_offsets = Vec::with_capacity(matrix.rows() + 1);
        let mut column_indices = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(0);
        for row in 0..matrix.rows() {
            for column in 0..matrix.columns() {
                let value = matrix.as_slice()[row * matrix.columns() + column];
                if value.norm() > zero_tolerance {
                    column_indices.push(column);
                    values.push(value);
                }
            }
            row_offsets.push(values.len());
        }
        Self::new(
            matrix.rows(),
            matrix.columns(),
            row_offsets,
            column_indices,
            values,
        )
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Number of explicitly stored entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// CSR row offsets.
    #[must_use]
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    /// CSR column indices.
    #[must_use]
    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    /// Explicitly stored values.
    #[must_use]
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    /// Tests Hermiticity without dense materialization.
    pub fn is_hermitian(&self, tolerance: f64) -> Result<bool, LinearOperatorError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(LinearOperatorError::InvalidTolerance);
        }
        if self.rows != self.columns {
            return Ok(false);
        }
        for row in 0..self.rows {
            for entry in self.row_offsets[row]..self.row_offsets[row + 1] {
                let column = self.column_indices[entry];
                let value = self.values[entry];
                if row == column {
                    if value.im.abs() > tolerance {
                        return Ok(false);
                    }
                    continue;
                }
                let reverse = self.value_at(column, row);
                if (value - reverse.conj()).norm() > tolerance {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Returns conservative Hermitian spectral bounds from Gershgorin discs.
    pub fn gershgorin_bounds(&self) -> Result<(f64, f64), LinearOperatorError> {
        if self.rows != self.columns {
            return Err(LinearOperatorError::NonSquare {
                rows: self.rows,
                columns: self.columns,
            });
        }
        let mut lower = f64::INFINITY;
        let mut upper = f64::NEG_INFINITY;
        for row in 0..self.rows {
            let mut center = 0.0;
            let mut radius = 0.0;
            for entry in self.row_offsets[row]..self.row_offsets[row + 1] {
                let column = self.column_indices[entry];
                let value = self.values[entry];
                if row == column {
                    center = value.re;
                } else {
                    radius += value.norm();
                }
            }
            lower = lower.min(center - radius);
            upper = upper.max(center + radius);
        }
        if lower.is_finite() && upper.is_finite() {
            Ok((lower, upper))
        } else {
            Err(LinearOperatorError::NonFiniteValue)
        }
    }

    /// Explicitly materializes the sparse matrix.
    pub fn to_dense(&self) -> Result<ComplexMatrix, LinearOperatorError> {
        let entries =
            self.rows
                .checked_mul(self.columns)
                .ok_or(LinearOperatorError::DenseSizeOverflow {
                    rows: self.rows,
                    columns: self.columns,
                })?;
        let mut data = vec![Complex64::new(0.0, 0.0); entries];
        for row in 0..self.rows {
            for entry in self.row_offsets[row]..self.row_offsets[row + 1] {
                data[row * self.columns + self.column_indices[entry]] = self.values[entry];
            }
        }
        ComplexMatrix::new(self.rows, self.columns, data)
            .map_err(|_| LinearOperatorError::NonFiniteValue)
    }

    fn value_at(&self, row: usize, column: usize) -> Complex64 {
        let range = self.row_offsets[row]..self.row_offsets[row + 1];
        self.column_indices[range.clone()]
            .binary_search(&column)
            .map_or(Complex64::new(0.0, 0.0), |index| {
                self.values[range.start + index]
            })
    }
}

impl LinearOperator for CsrMatrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn apply_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), LinearOperatorError> {
        validate_vectors(self.rows, self.columns, input, output)?;
        for (row, result) in output.iter_mut().enumerate() {
            *result = (self.row_offsets[row]..self.row_offsets[row + 1])
                .map(|entry| self.values[entry] * input[self.column_indices[entry]])
                .sum();
        }
        validate_finite_output(output)
    }
}

/// Zero-fill incomplete LU factorization used as a sparse inverse operator.
///
/// The factorization preserves the input CSR sparsity pattern. Small pivots
/// are shifted by a caller-controlled relative tolerance instead of creating
/// dense fill-in. Applying the preconditioner performs sparse forward and
/// backward substitution.
#[derive(Clone, Debug, PartialEq)]
pub struct Ilu0Preconditioner {
    dimension: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    factors: Vec<Complex64>,
    diagonal_indices: Vec<usize>,
}

impl Ilu0Preconditioner {
    /// Factors a square CSR matrix without adding fill entries.
    pub fn factor(matrix: &CsrMatrix, pivot_tolerance: f64) -> Result<Self, LinearOperatorError> {
        if matrix.rows() != matrix.columns() {
            return Err(LinearOperatorError::NonSquare {
                rows: matrix.rows(),
                columns: matrix.columns(),
            });
        }
        if !pivot_tolerance.is_finite() || pivot_tolerance <= 0.0 {
            return Err(LinearOperatorError::InvalidTolerance);
        }
        let dimension = matrix.rows();
        let row_offsets = matrix.row_offsets().to_vec();
        let column_indices = matrix.column_indices().to_vec();
        let mut factors = matrix.values().to_vec();
        let diagonal_indices = (0..dimension)
            .map(|row| {
                let range = row_offsets[row]..row_offsets[row + 1];
                column_indices[range.clone()]
                    .binary_search(&row)
                    .map(|index| range.start + index)
                    .map_err(|_| LinearOperatorError::MissingDiagonal { row })
            })
            .collect::<Result<Vec<_>, _>>()?;

        for row in 0..dimension {
            let row_end = row_offsets[row + 1];
            for lower_index in row_offsets[row]..diagonal_indices[row] {
                let lower_column = column_indices[lower_index];
                let pivot = factors[diagonal_indices[lower_column]];
                let multiplier = factors[lower_index] / pivot;
                factors[lower_index] = multiplier;
                let mut target = lower_index + 1;
                for upper_index in
                    (diagonal_indices[lower_column] + 1)..row_offsets[lower_column + 1]
                {
                    let upper_column = column_indices[upper_index];
                    while target < row_end && column_indices[target] < upper_column {
                        target += 1;
                    }
                    if target < row_end && column_indices[target] == upper_column {
                        let upper_value = factors[upper_index];
                        factors[target] -= multiplier * upper_value;
                    }
                }
            }

            let row_scale = factors[row_offsets[row]..row_end]
                .iter()
                .map(|value| value.norm())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let threshold = pivot_tolerance * row_scale;
            let diagonal = &mut factors[diagonal_indices[row]];
            if diagonal.norm() <= threshold {
                let direction = if diagonal.norm() == 0.0 {
                    Complex64::new(1.0, 0.0)
                } else {
                    *diagonal / diagonal.norm()
                };
                *diagonal += direction * threshold;
            }
        }
        if factors
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(LinearOperatorError::NonFiniteValue);
        }
        Ok(Self {
            dimension,
            row_offsets,
            column_indices,
            factors,
            diagonal_indices,
        })
    }

    /// Number of stored incomplete-factor entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.factors.len()
    }
}

impl LinearOperator for Ilu0Preconditioner {
    fn rows(&self) -> usize {
        self.dimension
    }

    fn columns(&self) -> usize {
        self.dimension
    }

    fn apply_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), LinearOperatorError> {
        validate_vectors(self.dimension, self.dimension, input, output)?;
        for row in 0..self.dimension {
            let correction = (self.row_offsets[row]..self.diagonal_indices[row])
                .map(|entry| self.factors[entry] * output[self.column_indices[entry]])
                .sum::<Complex64>();
            output[row] = input[row] - correction;
        }
        for row in (0..self.dimension).rev() {
            let correction = ((self.diagonal_indices[row] + 1)..self.row_offsets[row + 1])
                .map(|entry| self.factors[entry] * output[self.column_indices[entry]])
                .sum::<Complex64>();
            output[row] = (output[row] - correction) / self.factors[self.diagonal_indices[row]];
        }
        validate_finite_output(output)
    }
}

fn validate_vectors(
    rows: usize,
    columns: usize,
    input: &[Complex64],
    output: &[Complex64],
) -> Result<(), LinearOperatorError> {
    if input.len() != columns {
        return Err(LinearOperatorError::InputDimension {
            expected: columns,
            actual: input.len(),
        });
    }
    if output.len() != rows {
        return Err(LinearOperatorError::OutputDimension {
            expected: rows,
            actual: output.len(),
        });
    }
    if input
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(LinearOperatorError::NonFiniteValue);
    }
    Ok(())
}

fn validate_finite_output(output: &[Complex64]) -> Result<(), LinearOperatorError> {
    if output
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        Err(LinearOperatorError::NonFiniteValue)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_and_dense_operators_apply_identically() {
        let dense = ComplexMatrix::new(
            3,
            3,
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-0.5, 0.0),
                Complex64::new(0.3, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.3, 0.0),
                Complex64::new(0.7, 0.0),
            ],
        )
        .unwrap();
        let sparse = CsrMatrix::from_dense(&dense, 0.0).unwrap();
        let vector = [
            Complex64::new(0.4, -0.2),
            Complex64::new(-0.1, 0.8),
            Complex64::new(0.6, 0.3),
        ];
        assert_eq!(
            sparse.apply(&vector).unwrap(),
            dense.apply(&vector).unwrap()
        );
        assert!(sparse.is_hermitian(1.0e-12).unwrap());
        assert_eq!(sparse.to_dense().unwrap(), dense);
    }

    #[test]
    fn canonical_csr_structure_is_enforced() {
        assert_eq!(
            CsrMatrix::new(
                1,
                2,
                vec![0, 2],
                vec![1, 1],
                vec![Complex64::new(1.0, 0.0); 2],
            ),
            Err(LinearOperatorError::NoncanonicalRow {
                row: 0,
                previous: 1,
                current: 1,
            })
        );
    }

    #[test]
    fn gmres_recovers_a_complex_nonnormal_solution() {
        let operator = ComplexMatrix::new(
            3,
            3,
            vec![
                Complex64::new(3.0, 0.4),
                Complex64::new(-0.7, 0.2),
                Complex64::new(0.1, -0.3),
                Complex64::new(0.8, 0.0),
                Complex64::new(2.2, -0.5),
                Complex64::new(-0.4, 0.1),
                Complex64::new(0.0, 0.6),
                Complex64::new(0.5, 0.0),
                Complex64::new(1.7, 0.2),
            ],
        )
        .unwrap();
        let expected = vec![
            Complex64::new(0.4, -0.2),
            Complex64::new(-0.8, 0.5),
            Complex64::new(0.3, 0.7),
        ];
        let right = operator.apply(&expected).unwrap();
        let solution = gmres(&operator, &right, None, GmresOptions::default()).unwrap();
        assert!(solution.residual_norm() < 1.0e-11);
        for (actual, expected) in solution.vector().iter().zip(expected) {
            assert!((*actual - expected).norm() < 1.0e-10);
        }
    }

    #[test]
    fn ilu0_right_preconditioning_solves_a_sparse_complex_system() {
        let dimension = 64;
        let mut row_offsets = Vec::with_capacity(dimension + 1);
        let mut columns = Vec::with_capacity(3 * dimension);
        let mut values = Vec::with_capacity(3 * dimension);
        row_offsets.push(0);
        for row in 0..dimension {
            if row > 0 {
                columns.push(row - 1);
                values.push(Complex64::new(-0.4, 0.1));
            }
            columns.push(row);
            values.push(Complex64::new(2.5 + row as f64 / 32.0, 0.2));
            if row + 1 < dimension {
                columns.push(row + 1);
                values.push(Complex64::new(0.3, -0.05));
            }
            row_offsets.push(values.len());
        }
        let operator = CsrMatrix::new(dimension, dimension, row_offsets, columns, values).unwrap();
        let preconditioner = Ilu0Preconditioner::factor(&operator, 1.0e-12).unwrap();
        assert_eq!(preconditioner.nnz(), operator.nnz());
        let expected = (0..dimension)
            .map(|index| Complex64::new(index as f64 / 17.0, -0.3))
            .collect::<Vec<_>>();
        let right = operator.apply(&expected).unwrap();
        let solution = gmres_with_right_preconditioner(
            &operator,
            &preconditioner,
            &right,
            None,
            GmresOptions::default(),
        )
        .unwrap();
        assert!(solution.iterations() <= 2);
        assert!(solution.residual_norm() < 1.0e-10);
        assert!(solution
            .vector()
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (*actual - expected).norm() < 1.0e-10));
    }

    #[test]
    fn gmres_solves_a_large_sparse_chain_without_dense_storage() {
        let dimension = 100_000;
        let mut row_offsets = Vec::with_capacity(dimension + 1);
        let mut columns = Vec::with_capacity(3 * dimension);
        let mut values = Vec::with_capacity(3 * dimension);
        row_offsets.push(0);
        for row in 0..dimension {
            if row > 0 {
                columns.push(row - 1);
                values.push(Complex64::new(-0.25, 0.05));
            }
            columns.push(row);
            values.push(Complex64::new(2.0, 0.3));
            if row + 1 < dimension {
                columns.push(row + 1);
                values.push(Complex64::new(0.4, -0.1));
            }
            row_offsets.push(values.len());
        }
        let operator = CsrMatrix::new(dimension, dimension, row_offsets, columns, values).unwrap();
        assert!(operator.nnz() < 3 * dimension);
        let expected = vec![Complex64::new(0.7, -0.2); dimension];
        let right = operator.apply(&expected).unwrap();
        let solution = gmres(
            &operator,
            &right,
            None,
            GmresOptions {
                relative_tolerance: 1.0e-11,
                absolute_tolerance: 1.0e-12,
                restart: 24,
                max_iterations: 256,
            },
        )
        .unwrap();
        assert!(solution.iterations() < 64);
        assert!(solution
            .vector()
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (*actual - expected).norm() < 1.0e-9));
    }
}
