//! # sparseir-rust: Rust implementation of SparseIR functionality
//!
//! A high-performance implementation of the SparseIR (Sparse Intermediate Representation)
//! library in Rust, providing analytical continuation and sparse representation
//! functionality for quantum many-body physics calculations.

pub mod traits;
pub mod kernel;
pub mod poly;

// Re-export commonly used types and traits
pub use traits::{Statistics, StatisticsType, Fermionic, Bosonic, StatisticsMarker};
pub use kernel::{AbstractKernel, LogisticKernel, RegularizedBoseKernel, compute_f64};
pub use poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector, PolynomialData, PolynomialInfo};

// Re-export external dependencies for convenience
pub use ndarray::{Array1, Array2};
pub use twofloat::TwoFloat;
