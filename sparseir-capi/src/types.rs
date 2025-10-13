//! Opaque types for C API
//!
//! All Rust objects are wrapped in opaque pointers to hide implementation
//! details from C code.

use std::sync::Arc;
use sparseir_rust::kernel::{LogisticKernel, RegularizedBoseKernel, CentrosymmKernel};
use sparseir_rust::sve::SVEResult;
use sparseir_rust::basis::FiniteTempBasis;
use sparseir_rust::{Bosonic, Fermionic};

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

/// Opaque basis type for C API (compatible with libsparseir)
///
/// Represents a finite temperature basis (IR or DLR).
///
/// Note: Named `spir_basis` to match libsparseir C++ API exactly.
#[repr(C)]
pub struct spir_basis {
    inner: BasisType,
}

/// Internal basis type (not exposed to C)
pub(crate) enum BasisType {
    LogisticFermionic(Arc<FiniteTempBasis<LogisticKernel, Fermionic>>),
    LogisticBosonic(Arc<FiniteTempBasis<LogisticKernel, Bosonic>>),
    RegularizedBoseFermionic(Arc<FiniteTempBasis<RegularizedBoseKernel, Fermionic>>),
    RegularizedBoseBosonic(Arc<FiniteTempBasis<RegularizedBoseKernel, Bosonic>>),
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

    /// Get inner SVEResult for basis construction
    pub(crate) fn inner(&self) -> &Arc<SVEResult> {
        &self.inner
    }
}

impl spir_basis {
    pub(crate) fn new_logistic_fermionic(basis: FiniteTempBasis<LogisticKernel, Fermionic>) -> Self {
        Self {
            inner: BasisType::LogisticFermionic(Arc::new(basis)),
        }
    }

    pub(crate) fn new_logistic_bosonic(basis: FiniteTempBasis<LogisticKernel, Bosonic>) -> Self {
        Self {
            inner: BasisType::LogisticBosonic(Arc::new(basis)),
        }
    }

    pub(crate) fn new_regularized_bose_fermionic(basis: FiniteTempBasis<RegularizedBoseKernel, Fermionic>) -> Self {
        Self {
            inner: BasisType::RegularizedBoseFermionic(Arc::new(basis)),
        }
    }

    pub(crate) fn new_regularized_bose_bosonic(basis: FiniteTempBasis<RegularizedBoseKernel, Bosonic>) -> Self {
        Self {
            inner: BasisType::RegularizedBoseBosonic(Arc::new(basis)),
        }
    }

    pub(crate) fn size(&self) -> usize {
        match &self.inner {
            BasisType::LogisticFermionic(b) => b.size(),
            BasisType::LogisticBosonic(b) => b.size(),
            BasisType::RegularizedBoseFermionic(b) => b.size(),
            BasisType::RegularizedBoseBosonic(b) => b.size(),
        }
    }

    pub(crate) fn svals(&self) -> Vec<f64> {
        match &self.inner {
            BasisType::LogisticFermionic(b) => b.s.clone(),
            BasisType::LogisticBosonic(b) => b.s.clone(),
            BasisType::RegularizedBoseFermionic(b) => b.s.clone(),
            BasisType::RegularizedBoseBosonic(b) => b.s.clone(),
        }
    }

    pub(crate) fn statistics(&self) -> i32 {
        // 0 = Bosonic, 1 = Fermionic (matching libsparseir)
        match &self.inner {
            BasisType::LogisticFermionic(_) => 1,
            BasisType::LogisticBosonic(_) => 0,
            BasisType::RegularizedBoseFermionic(_) => 1,
            BasisType::RegularizedBoseBosonic(_) => 0,
        }
    }

    pub(crate) fn beta(&self) -> f64 {
        match &self.inner {
            BasisType::LogisticFermionic(b) => b.beta,
            BasisType::LogisticBosonic(b) => b.beta,
            BasisType::RegularizedBoseFermionic(b) => b.beta,
            BasisType::RegularizedBoseBosonic(b) => b.beta,
        }
    }

    pub(crate) fn wmax(&self) -> f64 {
        match &self.inner {
            BasisType::LogisticFermionic(b) => b.wmax(),
            BasisType::LogisticBosonic(b) => b.wmax(),
            BasisType::RegularizedBoseFermionic(b) => b.wmax(),
            BasisType::RegularizedBoseBosonic(b) => b.wmax(),
        }
    }

    pub(crate) fn default_tau_sampling_points(&self) -> Vec<f64> {
        match &self.inner {
            BasisType::LogisticFermionic(b) => b.default_tau_sampling_points(),
            BasisType::LogisticBosonic(b) => b.default_tau_sampling_points(),
            BasisType::RegularizedBoseFermionic(b) => b.default_tau_sampling_points(),
            BasisType::RegularizedBoseBosonic(b) => b.default_tau_sampling_points(),
        }
    }

    pub(crate) fn default_matsubara_sampling_points(&self, positive_only: bool) -> Vec<i64> {
        match &self.inner {
            BasisType::LogisticFermionic(b) => b.default_matsubara_sampling_points_i64(positive_only),
            BasisType::LogisticBosonic(b) => b.default_matsubara_sampling_points_i64(positive_only),
            BasisType::RegularizedBoseFermionic(b) => b.default_matsubara_sampling_points_i64(positive_only),
            BasisType::RegularizedBoseBosonic(b) => b.default_matsubara_sampling_points_i64(positive_only),
        }
    }
}

