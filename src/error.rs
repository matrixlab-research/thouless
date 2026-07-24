use std::error::Error;
use std::fmt;

/// Errors raised while constructing a tight-binding model.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelError {
    /// The embedding dimension must be greater than zero.
    InvalidRealDimension,
    /// More independent translations were supplied than embedding dimensions.
    TooManyTranslationVectors {
        /// Embedding-space dimension.
        real_dimension: usize,
        /// Number of translation vectors.
        translation_count: usize,
    },
    /// A translation vector has the wrong embedding dimension.
    InvalidTranslationVector {
        /// Index of the invalid translation vector.
        index: usize,
        /// Required number of components.
        expected: usize,
        /// Supplied number of components.
        actual: usize,
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
    /// An orbital position has the wrong embedding dimension.
    InvalidOrbitalPosition {
        /// Required number of components.
        expected: usize,
        /// Supplied number of components.
        actual: usize,
    },
    /// An orbital identifier does not belong to the model.
    UnknownOrbital {
        /// Invalid zero-based orbital index.
        index: usize,
    },
    /// A hopping cell offset has the wrong periodic dimension.
    InvalidCellOffset {
        /// Required number of integer components.
        expected: usize,
        /// Supplied number of integer components.
        actual: usize,
    },
    /// A hopping duplicates an existing term or its implicit Hermitian partner.
    DuplicateHopping,
    /// A model cannot be built without at least one orbital.
    EmptyModel,
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRealDimension => {
                write!(
                    formatter,
                    "the embedding dimension must be greater than zero"
                )
            }
            Self::TooManyTranslationVectors {
                real_dimension,
                translation_count,
            } => write!(
                formatter,
                "{translation_count} translation vectors cannot fit in \
                 {real_dimension}-dimensional space"
            ),
            Self::InvalidTranslationVector {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "translation vector {index} has {actual} components; expected {expected}"
            ),
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
            Self::DuplicateHopping => write!(
                formatter,
                "hopping duplicates an existing term or its Hermitian partner"
            ),
            Self::EmptyModel => write!(formatter, "a model must contain at least one orbital"),
        }
    }
}

impl Error for ModelError {}
