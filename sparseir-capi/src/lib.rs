//! C API for SparseIR Rust implementation
//!
//! This crate provides a C-compatible interface to the SparseIR library,
//! enabling usage from languages like Julia, Python, Fortran, and C++.

mod types;
mod kernel;

pub use types::*;
pub use kernel::*;

/// Error codes for C API (compatible with libsparseir)
pub type StatusCode = libc::c_int;

// Status codes matching libsparseir
pub const SPIR_COMPUTATION_SUCCESS: StatusCode = 0;
pub const SPIR_GET_IMPL_FAILED: StatusCode = -1;
pub const SPIR_INVALID_DIMENSION: StatusCode = -2;
pub const SPIR_INPUT_DIMENSION_MISMATCH: StatusCode = -3;
pub const SPIR_OUTPUT_DIMENSION_MISMATCH: StatusCode = -4;
pub const SPIR_NOT_SUPPORTED: StatusCode = -5;
pub const SPIR_INVALID_ARGUMENT: StatusCode = -6;
pub const SPIR_INTERNAL_ERROR: StatusCode = -7;

// Aliases for convenience
pub const SPIR_SUCCESS: StatusCode = SPIR_COMPUTATION_SUCCESS;
