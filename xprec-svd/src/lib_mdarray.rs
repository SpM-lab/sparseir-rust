//! # xprec-svd: High-Precision Truncated SVD (mdarray backend)
//!
//! A high-precision truncated SVD (TSVD) library implemented in Rust,
//! based on algorithms from libsparseir and Eigen3.
//!
//! This is the mdarray backend version.

pub mod precision;
pub mod qr;
pub mod svd;
pub mod tsvd;
pub mod utils;

pub use precision::{Precision, TwoFloatPrecision};
pub use qr::{QRPivoted, rrqr, truncate_qr_result};
pub use svd::{SVDResult, jacobi_svd};
pub use tsvd::{tsvd, tsvd_f64, tsvd_twofloat, tsvd_twofloat_from_f64, TSVDConfig, TSVDError};
pub use utils::{norm_2, norm_frobenius, norm_inf, norm_max, permutation_matrix};

// Re-export mdarray types
pub use mdarray::{Tensor, DTensor};

// Type aliases for convenience
pub type Matrix = Tensor<f64, (usize, usize)>;
pub type Vector = Tensor<f64, (usize,)>;

