//! Rust-native building blocks for tight-binding, topology, and steady-state
//! quantum transport.
//!
//! The crate is in its bootstrap stage. The currently implemented surface is
//! limited to model construction and structural invariants. Numerical
//! capabilities are tracked in the repository coverage matrices and issues.

#![forbid(unsafe_code)]

pub mod differentiation;
mod error;
pub mod geometry;
pub mod matrix;
pub mod model;
pub mod spectrum;
pub mod topology;

pub use error::{
    DifferentiationError, GeometryError, MatrixError, ModelError, SpectrumError, TopologyError,
};
pub use matrix::ComplexMatrix;
pub use num_complex::Complex64;
