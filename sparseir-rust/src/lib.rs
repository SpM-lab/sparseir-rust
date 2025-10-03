//! # sparseir-rust: Rust implementation of SparseIR functionality
//!
//! A high-performance implementation of the SparseIR (Sparse Intermediate Representation)
//! library in Rust, providing analytical continuation and sparse representation
//! functionality for quantum many-body physics calculations.

pub mod traits;
pub mod kernel;
pub mod poly;
pub mod gauss;
pub mod twofloattrait;

// Re-export commonly used types and traits
pub use traits::{Statistics, StatisticsType, Fermionic, Bosonic, StatisticsMarker};
pub use kernel::{AbstractKernel, LogisticKernel, RegularizedBoseKernel, ReducedKernel, compute_f64, KernelProperties, SVEHints, LogisticSVEHints, RegularizedBoseSVEHints, ReducedSVEHints};
pub use poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
pub use gauss::{Rule, legendre, legendre_custom, legendre_twofloat};
pub use twofloattrait::{CustomNumeric, TwoFloatArrayOps};

// Re-export external dependencies for convenience
pub use ndarray::{Array1, Array2};
pub use twofloat::TwoFloat;
