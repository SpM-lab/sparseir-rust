//! Opaque types for C API
//!
//! All Rust objects are wrapped in opaque pointers to hide implementation
//! details from C code.

use std::sync::Arc;
use sparseir_rust::kernel::{LogisticKernel, RegularizedBoseKernel, CentrosymmKernel};
use sparseir_rust::sve::SVEResult;
use sparseir_rust::basis::FiniteTempBasis;
use sparseir_rust::{Bosonic, Fermionic};
use sparseir_rust::poly::PiecewiseLegendrePolyVector;
use sparseir_rust::polyfourier::PiecewiseLegendreFTVector;
use sparseir_rust::freq::MatsubaraFreq;

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
    pub(crate) inner: BasisType,
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

    pub(crate) fn default_omega_sampling_points(&self) -> Vec<f64> {
        match &self.inner {
            BasisType::LogisticFermionic(b) => b.default_omega_sampling_points(),
            BasisType::LogisticBosonic(b) => b.default_omega_sampling_points(),
            BasisType::RegularizedBoseFermionic(b) => b.default_omega_sampling_points(),
            BasisType::RegularizedBoseBosonic(b) => b.default_omega_sampling_points(),
        }
    }
}

/// Internal enum to hold different function types
pub(crate) enum FuncsType {
    /// u, v: imaginary-time and real-frequency basis functions
    PolyVector(Arc<PiecewiseLegendrePolyVector>),
    
    /// uhat: Matsubara-frequency basis functions (Fermionic)
    FTVectorFermionic(Arc<PiecewiseLegendreFTVector<Fermionic>>),
    
    /// uhat: Matsubara-frequency basis functions (Bosonic)
    FTVectorBosonic(Arc<PiecewiseLegendreFTVector<Bosonic>>),
}

/// Opaque funcs type for C API (compatible with libsparseir)
///
/// Wraps piecewise Legendre polynomial representations:
/// - PiecewiseLegendrePolyVector for u and v
/// - PiecewiseLegendreFTVector for uhat
///
/// Note: Named `spir_funcs` to match libsparseir C++ API exactly.
#[repr(C)]
pub struct spir_funcs {
    inner: FuncsType,
    beta: f64,
}

impl spir_funcs {
    /// Create a new funcs object from a PiecewiseLegendrePolyVector
    pub(crate) fn from_poly_vector(poly: Arc<PiecewiseLegendrePolyVector>, beta: f64) -> Self {
        Self {
            inner: FuncsType::PolyVector(poly),
            beta,
        }
    }

    /// Create a new funcs object from a PiecewiseLegendreFTVector (Fermionic)
    pub(crate) fn from_ft_vector_fermionic(ft: Arc<PiecewiseLegendreFTVector<Fermionic>>, beta: f64) -> Self {
        Self {
            inner: FuncsType::FTVectorFermionic(ft),
            beta,
        }
    }

    /// Create a new funcs object from a PiecewiseLegendreFTVector (Bosonic)
    pub(crate) fn from_ft_vector_bosonic(ft: Arc<PiecewiseLegendreFTVector<Bosonic>>, beta: f64) -> Self {
        Self {
            inner: FuncsType::FTVectorBosonic(ft),
            beta,
        }
    }

    /// Get the number of basis functions
    pub(crate) fn size(&self) -> usize {
        match &self.inner {
            FuncsType::PolyVector(poly) => poly.polyvec.len(),
            FuncsType::FTVectorFermionic(ft) => ft.polyvec.len(),
            FuncsType::FTVectorBosonic(ft) => ft.polyvec.len(),
        }
    }

    /// Get knots for continuous functions (PolyVector only)
    pub(crate) fn knots(&self) -> Option<Vec<f64>> {
        match &self.inner {
            FuncsType::PolyVector(poly) => {
                // Get unique knots from all polynomials
                let mut all_knots = Vec::new();
                for p in &poly.polyvec {
                    for &knot in &p.knots {
                        if !all_knots.iter().any(|&k: &f64| (k - knot).abs() < 1e-14) {
                            all_knots.push(knot);
                        }
                    }
                }
                all_knots.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Some(all_knots)
            },
            _ => None, // FT vectors don't have knots in the traditional sense
        }
    }

    /// Evaluate at a single tau point (for continuous functions only)
    ///
    /// # Arguments
    /// * `x` - Point in [-1, 1] where tau = beta/2 * (x + 1) - beta/2
    ///
    /// # Returns
    /// Vector of function values, or None if not continuous
    pub(crate) fn eval_continuous(&self, x: f64) -> Option<Vec<f64>> {
        match &self.inner {
            FuncsType::PolyVector(poly) => {
                let mut result = Vec::with_capacity(poly.polyvec.len());
                for p in &poly.polyvec {
                    result.push(p.evaluate(x));
                }
                Some(result)
            },
            _ => None,
        }
    }

    /// Evaluate at a single Matsubara frequency (for FT functions only)
    ///
    /// # Arguments
    /// * `n` - Matsubara frequency index
    ///
    /// # Returns
    /// Vector of complex function values, or None if not FT type
    pub(crate) fn eval_matsubara(&self, n: i64) -> Option<Vec<num_complex::Complex64>> {
        use num_complex::Complex64;
        
        match &self.inner {
            FuncsType::FTVectorFermionic(ft) => {
                let freq = MatsubaraFreq::<Fermionic>::new(n).ok()?;
                let mut result = Vec::with_capacity(ft.polyvec.len());
                for p in &ft.polyvec {
                    result.push(p.evaluate(&freq));
                }
                Some(result)
            },
            FuncsType::FTVectorBosonic(ft) => {
                let freq = MatsubaraFreq::<Bosonic>::new(n).ok()?;
                let mut result = Vec::with_capacity(ft.polyvec.len());
                for p in &ft.polyvec {
                    result.push(p.evaluate(&freq));
                }
                Some(result)
            },
            _ => None,
        }
    }

    /// Batch evaluate at multiple tau points
    pub(crate) fn batch_eval_continuous(&self, xs: &[f64]) -> Option<Vec<Vec<f64>>> {
        match &self.inner {
            FuncsType::PolyVector(poly) => {
                let n_funcs = poly.polyvec.len();
                let n_points = xs.len();
                let mut result = vec![vec![0.0; n_points]; n_funcs];
                
                for (i, p) in poly.polyvec.iter().enumerate() {
                    for (j, &x) in xs.iter().enumerate() {
                        result[i][j] = p.evaluate(x);
                    }
                }
                Some(result)
            },
            _ => None,
        }
    }

    /// Batch evaluate at multiple Matsubara frequencies
    pub(crate) fn batch_eval_matsubara(&self, ns: &[i64]) -> Option<Vec<Vec<num_complex::Complex64>>> {
        use num_complex::Complex64;
        
        match &self.inner {
            FuncsType::FTVectorFermionic(ft) => {
                let n_funcs = ft.polyvec.len();
                let n_points = ns.len();
                let mut result = vec![vec![Complex64::new(0.0, 0.0); n_points]; n_funcs];
                
                for (i, p) in ft.polyvec.iter().enumerate() {
                    for (j, &n) in ns.iter().enumerate() {
                        if let Ok(freq) = MatsubaraFreq::<Fermionic>::new(n) {
                            result[i][j] = p.evaluate(&freq);
                        }
                    }
                }
                Some(result)
            },
            FuncsType::FTVectorBosonic(ft) => {
                let n_funcs = ft.polyvec.len();
                let n_points = ns.len();
                let mut result = vec![vec![Complex64::new(0.0, 0.0); n_points]; n_funcs];
                
                for (i, p) in ft.polyvec.iter().enumerate() {
                    for (j, &n) in ns.iter().enumerate() {
                        if let Ok(freq) = MatsubaraFreq::<Bosonic>::new(n) {
                            result[i][j] = p.evaluate(&freq);
                        }
                    }
                }
                Some(result)
            },
            _ => None,
        }
    }
}

