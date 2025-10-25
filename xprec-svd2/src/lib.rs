//! High-precision truncated SVD implementation using nalgebra
//!
//! This library provides QR + SVD based truncated SVD decomposition
//! with support for extended precision arithmetic.

pub mod svd;
pub mod tsvd;

// Re-export main types
pub use xprec::Df64;
pub use nalgebra::{DMatrix, DVector};

// Re-export main functions
pub use svd::{svd_decompose, SVDResult};
pub use tsvd::{tsvd, tsvd_df64, tsvd_df64_from_f64, tsvd_f64, TSVDConfig, TSVDError};

// Type aliases for convenience
pub type Matrix = DMatrix<f64>;
pub type Vector = DVector<f64>;
pub type Df64Matrix = DMatrix<Df64>;
pub type Df64Vector = DVector<Df64>;