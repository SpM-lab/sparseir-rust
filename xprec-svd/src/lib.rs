//! # xprec-svd: High-Precision Truncated SVD
//!
//! A high-precision truncated SVD (TSVD) library implemented in Rust,
//! based on algorithms from libsparseir and Eigen3.

pub mod precision;
pub mod qr;
pub mod svd;
pub mod tsvd;
pub mod utils;

pub use precision::Precision;
pub use qr::{QRPivoted, rrqr, truncate_qr_result};
pub use svd::{SVDResult, jacobi_svd};
pub use tsvd::{tsvd, tsvd_f64, TSVDConfig, TSVDError};
pub use utils::{norm_2, norm_frobenius, norm_inf, norm_max, permutation_matrix};

// Re-export common types
pub use ndarray::{Array1, Array2};

// Type aliases for convenience
pub type Matrix = Array2<f64>;
pub type Vector = Array1<f64>;
