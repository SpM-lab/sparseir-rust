//! # xprec-svd: High-Precision Truncated SVD
//!
//! A high-precision truncated SVD (TSVD) library implemented in Rust,
//! based on algorithms from libsparseir and Eigen3.
//!
//! Supports both ndarray and mdarray backends via feature flags.

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

// Backend-specific exports
#[cfg(feature = "ndarray-backend")]
pub use ndarray::{Array1, Array2};

#[cfg(feature = "mdarray-backend")]
pub use mdarray::{Tensor, DTensor};

// Type aliases for convenience
#[cfg(feature = "ndarray-backend")]
pub type Matrix = Array2<f64>;
#[cfg(feature = "ndarray-backend")]
pub type Vector = Array1<f64>;

#[cfg(feature = "mdarray-backend")]
pub type Matrix = Tensor<f64, (usize, usize)>;
#[cfg(feature = "mdarray-backend")]
pub type Vector = Tensor<f64, (usize,)>;
