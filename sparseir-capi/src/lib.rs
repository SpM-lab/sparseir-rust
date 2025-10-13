//! C API for SparseIR Rust implementation
//!
//! This crate provides a C-compatible interface to the SparseIR library,
//! enabling usage from languages like Julia, Python, Fortran, and C++.

mod types;
mod kernel;

pub use types::*;
pub use kernel::*;

/// Error codes for C API
pub type StatusCode = libc::c_int;

pub const SPIR_SUCCESS: StatusCode = 0;
pub const SPIR_ERROR_NULL_POINTER: StatusCode = -1;
pub const SPIR_ERROR_INVALID_ARGUMENT: StatusCode = -2;
pub const SPIR_ERROR_PANIC: StatusCode = -99;
