//! Rust-native building blocks for tight-binding, topology, and steady-state
//! quantum transport.
//!
//! The implementation is incomplete, but already includes model construction,
//! dense spectral algorithms, reciprocal geometry, discrete topology,
//! observables, and structure-preserving model transformations. Remaining
//! capabilities are tracked in the repository coverage matrices and issues.

#![forbid(unsafe_code)]

pub mod bands;
pub mod block_system;
pub mod continuum;
pub mod decomposition;
pub mod differentiation;
pub mod digest;
mod error;
pub mod gauge;
pub mod geometry;
pub mod graph;
pub mod interpolation;
pub mod kpm;
pub mod lattice_geometry;
pub mod lattice_reduction;
pub mod lead_modes;
pub mod linear_operator;
pub mod matrix;
pub mod model;
pub mod observables;
pub mod periodic;
pub mod random_matrix;
pub mod response;
pub mod spectrum;
pub mod symmetry;
pub mod topology;
pub mod transform;
pub mod transport;
pub mod wannier;

pub use error::{
    DifferentiationError, GeometryError, MatrixError, ModelError, ObservableError, SpectrumError,
    TopologyError,
};
pub use matrix::{ComplexMatrix, RealMatrix};
pub use num_complex::Complex64;
