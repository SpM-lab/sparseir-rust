//! Opaque types for C API
//!
//! All Rust objects are wrapped in opaque pointers to hide implementation
//! details from C code.

use std::sync::Arc;
use sparseir_rust::kernel::{LogisticKernel, RegularizedBoseKernel, CentrosymmKernel};
use sparseir_rust::sve::SVEResult;

/// Opaque kernel type for C API (compatible with libsparseir)
///
/// This is a tagged union that can hold either LogisticKernel or RegularizedBoseKernel.
/// The actual type is determined by which constructor was used.
///
/// Note: Named `spir_kernel` to match libsparseir C++ API exactly.
#[repr(C)]
pub struct spir_kernel {
    inner: KernelType,
}

/// Opaque SVE result type for C API (compatible with libsparseir)
///
/// Contains singular values and singular functions from SVE computation.
///
/// Note: Named `spir_sve_result` to match libsparseir C++ API exactly.
#[repr(C)]
pub struct spir_sve_result {
    inner: Arc<SVEResult>,
}

/// Internal kernel type (not exposed to C)
pub(crate) enum KernelType {
    Logistic(Arc<LogisticKernel>),
    RegularizedBose(Arc<RegularizedBoseKernel>),
}

impl spir_kernel {
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

    /// Get the inner kernel for SVE computation
    pub(crate) fn as_logistic(&self) -> Option<&Arc<LogisticKernel>> {
        match &self.inner {
            KernelType::Logistic(k) => Some(k),
            _ => None,
        }
    }

    pub(crate) fn as_regularized_bose(&self) -> Option<&Arc<RegularizedBoseKernel>> {
        match &self.inner {
            KernelType::RegularizedBose(k) => Some(k),
            _ => None,
        }
    }
}

impl spir_sve_result {
    pub(crate) fn new(sve_result: SVEResult) -> Self {
        Self {
            inner: Arc::new(sve_result),
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.inner.s.len()
    }

    pub(crate) fn svals(&self) -> &[f64] {
        &self.inner.s
    }

    pub(crate) fn epsilon(&self) -> f64 {
        self.inner.epsilon
    }

    pub(crate) fn truncate(&self, epsilon: f64, max_size: Option<usize>) -> Self {
        let (u_part, s_part, v_part) = self.inner.part(Some(epsilon), max_size);
        let truncated = SVEResult::new(u_part, s_part, v_part, epsilon);
        Self::new(truncated)
    }
}

