//! # xprec-svd: High-Precision Truncated SVD
//!
//! A high-precision truncated SVD (TSVD) library implemented in Rust,
//! based on algorithms from libsparseir and Eigen3.
//!
//! Now using mdarray for array operations.

pub mod precision;
pub mod qr;
pub mod svd;
pub mod tsvd;
pub mod utils;

pub use precision::{Df64Precision, Precision};
pub use qr::{QRPivoted, rrqr, truncate_qr_result};
pub use svd::{SVDResult, jacobi_svd};
pub use tsvd::{TSVDConfig, TSVDError, tsvd, tsvd_df64, tsvd_df64_from_f64, tsvd_f64};
pub use utils::{norm_2, norm_frobenius, norm_inf, norm_max, permutation_matrix};

// Re-export mdarray types
pub use mdarray::{DTensor, Tensor};

// Type aliases for convenience
pub type Matrix = Tensor<f64, (usize, usize)>;
pub type Vector = Tensor<f64, (usize,)>;
