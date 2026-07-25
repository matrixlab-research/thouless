use std::error::Error;
use std::fmt;

/// Errors raised by dense complex-matrix operations.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatrixError {
    /// The number of supplied entries does not match the matrix shape.
    InvalidDataLength {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
        /// Number of supplied entries.
        actual: usize,
    },
    /// A row or column index is outside the matrix.
    IndexOutOfBounds {
        /// Requested row.
        row: usize,
        /// Requested column.
        column: usize,
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// An operation requires a square matrix.
    NotSquare {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// A matrix contains NaN or infinity.
    NonFiniteValue,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDataLength {
                rows,
                columns,
                actual,
            } => write!(
                formatter,
                "a {rows}x{columns} matrix requires {} entries, but {actual} were supplied",
                rows * columns
            ),
            Self::IndexOutOfBounds {
                row,
                column,
                rows,
                columns,
            } => write!(
                formatter,
                "matrix index ({row}, {column}) is outside shape ({rows}, {columns})"
            ),
            Self::NotSquare { rows, columns } => {
                write!(formatter, "matrix shape ({rows}, {columns}) is not square")
            }
            Self::NonFiniteValue => write!(formatter, "matrix contains a non-finite value"),
        }
    }
}

impl Error for MatrixError {}

/// Errors raised while constructing or evaluating a tight-binding model.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelError {
    /// Primitive lattice vectors must form a square matrix.
    InvalidPrimitiveVectors {
        /// Expected number of components in every vector.
        expected: usize,
        /// Index of the invalid vector.
        vector: usize,
        /// Supplied number of components.
        actual: usize,
    },
    /// Primitive lattice vectors are linearly dependent.
    SingularLattice,
    /// A periodic axis is outside the real-space dimension.
    InvalidPeriodicAxis {
        /// Invalid axis.
        axis: usize,
        /// Number of real-space axes.
        dimension: usize,
    },
    /// A periodic axis occurs more than once.
    DuplicatePeriodicAxis {
        /// Repeated axis.
        axis: usize,
    },
    /// A floating-point input is NaN or infinite.
    NonFiniteValue {
        /// Name of the invalid field.
        field: &'static str,
    },
    /// An orbital label is empty.
    EmptyOrbitalLabel,
    /// An orbital label is already present in the model.
    DuplicateOrbitalLabel {
        /// Repeated label.
        label: String,
    },
    /// An orbital position has the wrong real-space dimension.
    InvalidOrbitalPosition {
        /// Required number of components.
        expected: usize,
        /// Supplied number of components.
        actual: usize,
    },
    /// A localized orbital must contain at least one internal degree of freedom.
    InvalidDegreesOfFreedom,
    /// An orbital identifier does not belong to the model.
    UnknownOrbital {
        /// Invalid zero-based orbital index.
        index: usize,
    },
    /// A hopping cell offset has the wrong real-space dimension.
    InvalidCellOffset {
        /// Required number of integer components.
        expected: usize,
        /// Supplied number of integer components.
        actual: usize,
    },
    /// A local or hopping block has a shape incompatible with its orbitals.
    InvalidBlockShape {
        /// Required number of rows.
        expected_rows: usize,
        /// Required number of columns.
        expected_columns: usize,
        /// Supplied number of rows.
        actual_rows: usize,
        /// Supplied number of columns.
        actual_columns: usize,
    },
    /// An onsite block is not Hermitian.
    NonHermitianOnsite,
    /// A zero-displacement self hopping must be represented as an onsite block.
    SelfHoppingAtHome,
    /// A hopping duplicates an existing term or its implicit Hermitian partner.
    DuplicateHopping,
    /// A model cannot be built without at least one orbital.
    EmptyModel,
    /// A reciprocal-space point has the wrong periodic dimension.
    InvalidMomentum {
        /// Required number of reduced components.
        expected: usize,
        /// Supplied number of reduced components.
        actual: usize,
    },
    /// A matrix operation failed.
    Matrix(MatrixError),
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrimitiveVectors {
                expected,
                vector,
                actual,
            } => write!(
                formatter,
                "primitive vector {vector} has {actual} components; expected {expected}"
            ),
            Self::SingularLattice => write!(formatter, "primitive vectors are singular"),
            Self::InvalidPeriodicAxis { axis, dimension } => write!(
                formatter,
                "periodic axis {axis} is outside real-space dimension {dimension}"
            ),
            Self::DuplicatePeriodicAxis { axis } => {
                write!(formatter, "periodic axis {axis} occurs more than once")
            }
            Self::NonFiniteValue { field } => {
                write!(formatter, "{field} contains a non-finite value")
            }
            Self::EmptyOrbitalLabel => write!(formatter, "orbital labels cannot be empty"),
            Self::DuplicateOrbitalLabel { label } => {
                write!(formatter, "orbital label {label:?} is already present")
            }
            Self::InvalidOrbitalPosition { expected, actual } => write!(
                formatter,
                "orbital position has {actual} components; expected {expected}"
            ),
            Self::InvalidDegreesOfFreedom => {
                write!(
                    formatter,
                    "an orbital must have at least one degree of freedom"
                )
            }
            Self::UnknownOrbital { index } => {
                write!(
                    formatter,
                    "orbital index {index} does not belong to the model"
                )
            }
            Self::InvalidCellOffset { expected, actual } => write!(
                formatter,
                "cell offset has {actual} components; expected {expected}"
            ),
            Self::InvalidBlockShape {
                expected_rows,
                expected_columns,
                actual_rows,
                actual_columns,
            } => write!(
                formatter,
                "block shape ({actual_rows}, {actual_columns}) does not match \
                 required shape ({expected_rows}, {expected_columns})"
            ),
            Self::NonHermitianOnsite => write!(formatter, "onsite block is not Hermitian"),
            Self::SelfHoppingAtHome => write!(
                formatter,
                "zero-displacement self hopping must be represented as onsite"
            ),
            Self::DuplicateHopping => write!(
                formatter,
                "hopping duplicates an existing term or its Hermitian partner"
            ),
            Self::EmptyModel => write!(formatter, "a model must contain at least one orbital"),
            Self::InvalidMomentum { expected, actual } => write!(
                formatter,
                "momentum has {actual} reduced components; expected {expected}"
            ),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl Error for ModelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatrixError> for ModelError {
    fn from(error: MatrixError) -> Self {
        Self::Matrix(error)
    }
}

/// Errors raised by spectral algorithms.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpectrumError {
    /// The input matrix is not square.
    NotSquare {
        /// Number of rows.
        rows: usize,
        /// Number of columns.
        columns: usize,
    },
    /// The input matrix is not Hermitian within the requested tolerance.
    NonHermitian,
    /// The numerical eigensolver failed.
    DecompositionFailure,
    /// Matrix construction or access failed.
    Matrix(MatrixError),
}

impl fmt::Display for SpectrumError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSquare { rows, columns } => {
                write!(formatter, "matrix shape ({rows}, {columns}) is not square")
            }
            Self::NonHermitian => write!(formatter, "matrix is not Hermitian"),
            Self::DecompositionFailure => write!(formatter, "Hermitian eigensolver failed"),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl Error for SpectrumError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatrixError> for SpectrumError {
    fn from(error: MatrixError) -> Self {
        Self::Matrix(error)
    }
}

/// Errors raised by numerical differentiation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DifferentiationError {
    /// At least two samples are required.
    InsufficientSamples,
    /// The coordinate step must be finite and nonzero.
    InvalidStep,
    /// Every sampled matrix must have the same shape.
    ShapeMismatch,
    /// Matrix construction failed.
    Matrix(MatrixError),
}

impl fmt::Display for DifferentiationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientSamples => {
                write!(formatter, "at least two samples are required")
            }
            Self::InvalidStep => write!(formatter, "sample step must be finite and nonzero"),
            Self::ShapeMismatch => write!(formatter, "sampled matrices have different shapes"),
            Self::Matrix(error) => error.fmt(formatter),
        }
    }
}

impl Error for DifferentiationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Matrix(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MatrixError> for DifferentiationError {
    fn from(error: MatrixError) -> Self {
        Self::Matrix(error)
    }
}

/// Errors raised by discrete geometric and topological calculations.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TopologyError {
    /// A Wilson line requires at least two state frames.
    InsufficientFrames,
    /// Every frame must contain the same nonempty subspace and basis.
    IncompatibleFrames,
    /// Consecutive subspaces have a singular overlap.
    SingularOverlap,
    /// A Wilson-loop eigendecomposition did not converge.
    EigendecompositionFailed,
    /// A Berry-connection link must be square.
    NonSquareLink,
    /// A Berry-connection coordinate step must be finite and nonzero.
    InvalidConnectionStep,
    /// A matrix advertised as a transport link is not unitary.
    NonUnitaryLink,
    /// A unitary matrix power requires a finite real exponent.
    InvalidUnitaryExponent,
}

impl fmt::Display for TopologyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientFrames => {
                write!(formatter, "a Wilson line requires at least two frames")
            }
            Self::IncompatibleFrames => {
                write!(formatter, "state frames have incompatible shapes")
            }
            Self::SingularOverlap => {
                write!(
                    formatter,
                    "consecutive state subspaces have singular overlap"
                )
            }
            Self::EigendecompositionFailed => {
                write!(formatter, "Wilson-loop eigendecomposition failed")
            }
            Self::NonSquareLink => write!(formatter, "parallel-transport link must be square"),
            Self::InvalidConnectionStep => {
                write!(
                    formatter,
                    "Berry-connection step must be finite and nonzero"
                )
            }
            Self::NonUnitaryLink => write!(formatter, "transport link must be unitary"),
            Self::InvalidUnitaryExponent => {
                write!(formatter, "unitary matrix exponent must be finite")
            }
        }
    }
}

impl Error for TopologyError {}

/// Errors raised while sampling paths in reciprocal space.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeometryError {
    /// Reciprocal-space sampling requires at least one periodic direction.
    NoPeriodicDirections,
    /// A path requires at least two nodes.
    InsufficientPathNodes,
    /// A path node has the wrong reduced-coordinate dimension.
    InvalidPathNode {
        /// Index of the invalid node.
        node: usize,
        /// Required number of reduced components.
        expected: usize,
        /// Supplied number of reduced components.
        actual: usize,
    },
    /// The requested number of samples cannot represent every path node.
    InsufficientPathSamples {
        /// Minimum number of samples.
        minimum: usize,
        /// Requested number of samples.
        actual: usize,
    },
    /// A path node contains NaN or infinity.
    NonFinitePathNode,
    /// The selected periodic primitive vectors are numerically singular.
    SingularPeriodicGeometry,
    /// Every segment in the path has zero Cartesian length.
    ZeroLengthPath,
    /// A reciprocal mesh requires one nonzero extent per periodic direction.
    InvalidMeshShape {
        /// Number of periodic directions.
        expected: usize,
        /// Number of supplied mesh extents.
        actual: usize,
    },
    /// One reciprocal-mesh extent is zero.
    EmptyMeshAxis {
        /// Zero-sized reduced-coordinate axis.
        axis: usize,
    },
    /// One fractional mesh offset is outside `[0, 1)`.
    InvalidMeshOffset {
        /// Invalid reduced-coordinate axis.
        axis: usize,
    },
    /// The product of reciprocal-mesh extents exceeds `usize`.
    MeshSizeOverflow,
}

impl fmt::Display for GeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPeriodicDirections => {
                write!(formatter, "reciprocal sampling requires a periodic lattice")
            }
            Self::InsufficientPathNodes => write!(formatter, "a path requires at least two nodes"),
            Self::InvalidPathNode {
                node,
                expected,
                actual,
            } => write!(
                formatter,
                "path node {node} has {actual} components; expected {expected}"
            ),
            Self::InsufficientPathSamples { minimum, actual } => write!(
                formatter,
                "path requires at least {minimum} samples, but {actual} were requested"
            ),
            Self::NonFinitePathNode => write!(formatter, "path nodes must be finite"),
            Self::SingularPeriodicGeometry => {
                write!(
                    formatter,
                    "periodic primitive vectors are numerically singular"
                )
            }
            Self::ZeroLengthPath => {
                write!(
                    formatter,
                    "path must contain a nonzero Cartesian displacement"
                )
            }
            Self::InvalidMeshShape { expected, actual } => write!(
                formatter,
                "reciprocal mesh has {actual} extents; expected {expected}"
            ),
            Self::EmptyMeshAxis { axis } => {
                write!(formatter, "reciprocal mesh axis {axis} has zero samples")
            }
            Self::InvalidMeshOffset { axis } => write!(
                formatter,
                "reciprocal mesh offset on axis {axis} must be finite and in [0, 1)"
            ),
            Self::MeshSizeOverflow => {
                write!(formatter, "reciprocal mesh contains too many points")
            }
        }
    }
}

impl Error for GeometryError {}

/// Errors raised while projecting observables into state subspaces.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ObservableError {
    /// A state frame must contain at least one state and one basis component.
    EmptyStateFrame,
    /// A diagonal observable must provide one value per basis component.
    InvalidDiagonalLength {
        /// Required number of values.
        expected: usize,
        /// Supplied number of values.
        actual: usize,
    },
    /// An observable value is NaN or infinity.
    NonFiniteValue,
    /// Pauli decomposition requires a two-by-two matrix.
    InvalidPauliShape {
        /// Supplied number of rows.
        rows: usize,
        /// Supplied number of columns.
        columns: usize,
    },
}

impl fmt::Display for ObservableError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyStateFrame => write!(formatter, "state frame must be nonempty"),
            Self::InvalidDiagonalLength { expected, actual } => write!(
                formatter,
                "diagonal observable has {actual} values; expected {expected}"
            ),
            Self::NonFiniteValue => write!(formatter, "observable values must be finite"),
            Self::InvalidPauliShape { rows, columns } => write!(
                formatter,
                "Pauli decomposition requires a 2x2 matrix, got {rows}x{columns}"
            ),
        }
    }
}

impl Error for ObservableError {}
