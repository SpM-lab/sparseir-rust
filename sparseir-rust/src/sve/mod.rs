//! Singular Value Expansion (SVE) module
//!
//! This module provides functionality for computing the singular value expansion
//! of integral kernels, which is the core of the SparseIR algorithm.
//!
//! # Design Principles
//!
//! 1. **Separation of Concerns**: Clear separation between general SVE processing
//!    and symmetry-specific logic
//! 2. **Type-driven Precision**: Automatic selection of working precision based on
//!    required accuracy
//! 3. **Symmetry Exploitation**: Efficient computation for centrosymmetric kernels
//!    via even/odd decomposition
//! 4. **Composability**: Modular design allowing easy extension and testing

mod types;
mod result;
pub mod utils;  // Public for testing
mod strategy;
mod compute;

// Re-export public API
pub use types::{TworkType, SVDStrategy, safe_epsilon};
pub use result::SVEResult;
pub use strategy::{SVEStrategy, SamplingSVE, CentrosymmSVE};
pub use compute::{compute_sve, compute_svd, truncate};


#[cfg(test)]
#[path = "../sve_comparison_tests.rs"]
mod sve_comparison_tests;

#[cfg(test)]
#[path = "../sve_extend_tests.rs"]
mod sve_extend_tests;
