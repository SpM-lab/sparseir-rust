//! Opaque types for C API
//!
//! All Rust objects are wrapped in opaque pointers to hide implementation
//! details from C code.

use std::sync::Arc;
use sparseir_rust::kernel::{LogisticKernel, RegularizedBoseKernel, CentrosymmKernel};

/// Opaque kernel type for C API
///
/// This is a tagged union that can hold either LogisticKernel or RegularizedBoseKernel.
/// The actual type is determined by which constructor was used.
#[repr(C)]
pub struct SparseIRKernel {
    inner: KernelType,
}

/// Internal kernel type (not exposed to C)
pub(crate) enum KernelType {
    Logistic(Arc<LogisticKernel>),
    RegularizedBose(Arc<RegularizedBoseKernel>),
}

impl SparseIRKernel {
    pub(crate) fn new_logistic(lambda: f64) -> Self {
        Self {
            inner: KernelType::Logistic(Arc::new(LogisticKernel::new(lambda))),
        }
    }

    pub(crate) fn new_regularized_bose(lambda: f64) -> Self {
        Self {
            inner: KernelType::RegularizedBose(Arc::new(RegularizedBoseKernel::new(lambda))),
        }
    }

    pub(crate) fn lambda(&self) -> f64 {
        match &self.inner {
            KernelType::Logistic(k) => k.lambda(),
            KernelType::RegularizedBose(k) => k.lambda(),
        }
    }

    pub(crate) fn compute(&self, x: f64, y: f64) -> f64 {
        match &self.inner {
            KernelType::Logistic(k) => k.compute(x, y),
            KernelType::RegularizedBose(k) => k.compute(x, y),
        }
    }
}

