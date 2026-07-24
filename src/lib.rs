//! Rust-native building blocks for tight-binding, topology, and steady-state
//! quantum transport.
//!
//! The crate is in its bootstrap stage. The currently implemented surface is
//! limited to model construction and structural invariants. Numerical
//! capabilities are tracked in the repository coverage matrices and issues.

#![forbid(unsafe_code)]

mod error;
pub mod model;

pub use error::ModelError;
pub use num_complex::Complex64;
