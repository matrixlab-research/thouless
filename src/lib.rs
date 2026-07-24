//! Rust-native building blocks for tight-binding, topology, and steady-state
//! quantum transport.
//!
//! The implementation is incomplete, but already includes model construction,
//! dense spectral algorithms, reciprocal geometry, discrete topology,
//! observables, and structure-preserving model transformations. Remaining
//! capabilities are tracked in the repository coverage matrices and issues.

#![forbid(unsafe_code)]

pub mod differentiation;
mod error;
pub mod geometry;
pub mod matrix;
pub mod model;
pub mod observables;
pub mod spectrum;
pub mod topology;
pub mod transform;

pub use error::{
    DifferentiationError, GeometryError, MatrixError, ModelError, ObservableError, SpectrumError,
    TopologyError,
};
pub use matrix::ComplexMatrix;
pub use num_complex::Complex64;
