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
#[derive(Clone)]
#[repr(C)]
pub struct spir_kernel {
    inner: KernelType,
}

/// Opaque SVE result type for C API (compatible with libsparseir)
///
/// Contains singular values and singular functions from SVE computation.
///
/// Note: Named `spir_sve_result` to match libsparseir C++ API exactly.
#[derive(Clone)]
#[repr(C)]
pub struct spir_sve_result {
    inner: Arc<SVEResult>,
}

/// Opaque basis type for C API (compatible with libsparseir)
///
/// Represents a finite temperature basis (IR or DLR).
///
/// Note: Named `spir_basis` to match libsparseir C++ API exactly.
#[derive(Clone)]
#[repr(C)]
pub struct spir_basis {
    pub(crate) inner: BasisType,
}

/// Internal basis type (not exposed to C)
#[derive(Clone)]
pub(crate) enum BasisType {
    LogisticFermionic(Arc<FiniteTempBasis<LogisticKernel, Fermionic>>),
    LogisticBosonic(Arc<FiniteTempBasis<LogisticKernel, Bosonic>>),
    RegularizedBoseFermionic(Arc<FiniteTempBasis<RegularizedBoseKernel, Fermionic>>),
    RegularizedBoseBosonic(Arc<FiniteTempBasis<RegularizedBoseKernel, Bosonic>>),
}

/// Internal kernel type (not exposed to C)
#[derive(Clone)]
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
#[derive(Clone)]
pub(crate) enum FuncsType {
    /// Continuous functions (u or v): PiecewiseLegendrePolyVector
    /// statistics: -1 for v (omega, no periodicity)
    ///              0 for u (tau, Bosonic)
    ///              1 for u (tau, Fermionic)
    PolyVector {
        poly: Arc<PiecewiseLegendrePolyVector>,
        statistics: i32,
    },
    
    /// Fourier-transformed functions (uhat): PiecewiseLegendreFTVector
    /// statistics: 0 for Bosonic, 1 for Fermionic
    /// Only one of ft_fermionic/ft_bosonic is Some, the other is None
    FTVector {
        ft_fermionic: Option<Arc<PiecewiseLegendreFTVector<Fermionic>>>,
        ft_bosonic: Option<Arc<PiecewiseLegendreFTVector<Bosonic>>>,
        statistics: i32,
    },
}

/// Opaque funcs type for C API (compatible with libsparseir)
///
/// Wraps piecewise Legendre polynomial representations:
/// - PiecewiseLegendrePolyVector for u and v
/// - PiecewiseLegendreFTVector for uhat
///
/// Note: Named `spir_funcs` to match libsparseir C++ API exactly.
#[derive(Clone)]
#[repr(C)]
pub struct spir_funcs {
    pub(crate) inner: FuncsType,
    pub(crate) beta: f64,
}

impl spir_funcs {
    /// Create u funcs (tau-domain, Fermionic)
    pub(crate) fn from_u_fermionic(poly: Arc<PiecewiseLegendrePolyVector>, beta: f64) -> Self {
        Self {
            inner: FuncsType::PolyVector { poly, statistics: 1 },
            beta,
        }
    }

    /// Create u funcs (tau-domain, Bosonic)
    pub(crate) fn from_u_bosonic(poly: Arc<PiecewiseLegendrePolyVector>, beta: f64) -> Self {
        Self {
            inner: FuncsType::PolyVector { poly, statistics: 0 },
            beta,
        }
    }

    /// Create v funcs (omega-domain, no statistics)
    pub(crate) fn from_v(poly: Arc<PiecewiseLegendrePolyVector>, beta: f64) -> Self {
        Self {
            inner: FuncsType::PolyVector { poly, statistics: -1 },
            beta,
        }
    }

    /// Create uhat funcs (Matsubara-domain, Fermionic)
    pub(crate) fn from_uhat_fermionic(ft: Arc<PiecewiseLegendreFTVector<Fermionic>>, beta: f64) -> Self {
        Self {
            inner: FuncsType::FTVector {
                ft_fermionic: Some(ft),
                ft_bosonic: None,
                statistics: 1,
            },
            beta,
        }
    }

    /// Create uhat funcs (Matsubara-domain, Bosonic)
    pub(crate) fn from_uhat_bosonic(ft: Arc<PiecewiseLegendreFTVector<Bosonic>>, beta: f64) -> Self {
        Self {
            inner: FuncsType::FTVector {
                ft_fermionic: None,
                ft_bosonic: Some(ft),
                statistics: 0,
            },
            beta,
        }
    }

    /// Get the number of basis functions
    pub(crate) fn size(&self) -> usize {
        match &self.inner {
            FuncsType::PolyVector { poly, .. } => poly.polyvec.len(),
            FuncsType::FTVector { ft_fermionic, ft_bosonic, .. } => {
                if let Some(ft) = ft_fermionic {
                    ft.polyvec.len()
                } else if let Some(ft) = ft_bosonic {
                    ft.polyvec.len()
                } else {
                    0
                }
            },
        }
    }

    /// Get knots for continuous functions (PolyVector only)
    pub(crate) fn knots(&self) -> Option<Vec<f64>> {
        match &self.inner {
            FuncsType::PolyVector { poly, .. } => {
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

    /// Evaluate at a single tau/omega point (for continuous functions only)
    ///
    /// # Arguments
    /// * `x` - For u: tau ∈ [-beta, beta], For v: omega ∈ [-omega_max, omega_max]
    ///
    /// # Returns
    /// Vector of function values, or None if not continuous
    pub(crate) fn eval_continuous(&self, x: f64) -> Option<Vec<f64>> {
        match &self.inner {
            FuncsType::PolyVector { poly, statistics } => {
                // For u (tau functions): handle periodicity
                // For v (omega functions): no periodicity (statistics = -1)
                let (x_reg, sign) = if *statistics >= 0 {
                    // u functions: regularize tau to [0, beta]
                    let fermionic_sign = if *statistics == 1 { -1.0 } else { 1.0 };
                    regularize_tau(x, self.beta, fermionic_sign)
                } else {
                    // v functions: no regularization needed
                    (x, 1.0)
                };
                
                let mut result = Vec::with_capacity(poly.polyvec.len());
                for p in &poly.polyvec {
                    result.push(sign * p.evaluate(x_reg));
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
        match &self.inner {
            FuncsType::FTVector { ft_fermionic, ft_bosonic, statistics } => {
                if *statistics == 1 {
                    // Fermionic
                    let ft = ft_fermionic.as_ref()?;
                    let freq = MatsubaraFreq::<Fermionic>::new(n).ok()?;
                    let mut result = Vec::with_capacity(ft.polyvec.len());
                    for p in &ft.polyvec {
                        result.push(p.evaluate(&freq));
                    }
                    Some(result)
                } else {
                    // Bosonic
                    let ft = ft_bosonic.as_ref()?;
                    let freq = MatsubaraFreq::<Bosonic>::new(n).ok()?;
                    let mut result = Vec::with_capacity(ft.polyvec.len());
                    for p in &ft.polyvec {
                        result.push(p.evaluate(&freq));
                    }
                    Some(result)
                }
            },
            _ => None,
        }
    }

    /// Batch evaluate at multiple tau/omega points
    pub(crate) fn batch_eval_continuous(&self, xs: &[f64]) -> Option<Vec<Vec<f64>>> {
        match &self.inner {
            FuncsType::PolyVector { poly, statistics } => {
                let n_funcs = poly.polyvec.len();
                let n_points = xs.len();
                let mut result = vec![vec![0.0; n_points]; n_funcs];
                
                let fermionic_sign = if *statistics == 1 { -1.0 } else { 1.0 };
                
                for (i, p) in poly.polyvec.iter().enumerate() {
                    for (j, &x) in xs.iter().enumerate() {
                        let (x_reg, sign) = if *statistics >= 0 {
                            // u functions: regularize tau
                            regularize_tau(x, self.beta, fermionic_sign)
                        } else {
                            // v functions: no regularization
                            (x, 1.0)
                        };
                        result[i][j] = sign * p.evaluate(x_reg);
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
            FuncsType::FTVector { ft_fermionic, ft_bosonic, statistics } => {
                if *statistics == 1 {
                    // Fermionic
                    let ft = ft_fermionic.as_ref()?;
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
                } else {
                    // Bosonic
                    let ft = ft_bosonic.as_ref()?;
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
                }
            },
            _ => None,
        }
    }

    /// Create a new funcs object containing a subset of functions
    ///
    /// # Arguments
    /// * `indices` - Slice of indices specifying which functions to include
    ///
    /// # Returns
    /// A new `spir_funcs` object with only the selected functions, or None if indices are invalid
    pub(crate) fn get_slice(&self, indices: &[usize]) -> Option<Self> {
        use sparseir_rust::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
        
        match &self.inner {
            FuncsType::PolyVector { poly, statistics } => {
                // Check that all indices are valid
                if indices.iter().any(|&i| i >= poly.polyvec.len()) {
                    return None;
                }
                
                    // Extract the selected polynomials
                    let new_polyvec: Vec<PiecewiseLegendrePoly> = indices
                        .iter()
                        .map(|&i| poly.polyvec[i].clone())
                        .collect();
                    
                    let new_poly = Arc::new(PiecewiseLegendrePolyVector::new(new_polyvec));
                
                Some(Self {
                    inner: FuncsType::PolyVector {
                        poly: new_poly,
                        statistics: *statistics,
                    },
                    beta: self.beta,
                })
            },
            FuncsType::FTVector { ft_fermionic, ft_bosonic, statistics } => {
                if *statistics == 1 {
                    // Fermionic
                    use sparseir_rust::traits::Fermionic;
                    let ft = ft_fermionic.as_ref()?;
                    
                    // Check that all indices are valid
                    if indices.iter().any(|&i| i >= ft.polyvec.len()) {
                        return None;
                    }
                    
                    // Extract the selected FT polynomials
                    let new_polyvec: Vec<sparseir_rust::polyfourier::PiecewiseLegendreFT<Fermionic>> = indices
                        .iter()
                        .map(|&i| ft.polyvec[i].clone())
                        .collect();
                    
                    let new_ft = Arc::new(PiecewiseLegendreFTVector::from_vector(new_polyvec));
                    
                    Some(Self {
                        inner: FuncsType::FTVector {
                            ft_fermionic: Some(new_ft),
                            ft_bosonic: None,
                            statistics: *statistics,
                        },
                        beta: self.beta,
                    })
                } else {
                    // Bosonic
                    use sparseir_rust::traits::Bosonic;
                    let ft = ft_bosonic.as_ref()?;
                    
                    // Check that all indices are valid
                    if indices.iter().any(|&i| i >= ft.polyvec.len()) {
                        return None;
                    }
                    
                    // Extract the selected FT polynomials
                    let new_polyvec: Vec<sparseir_rust::polyfourier::PiecewiseLegendreFT<Bosonic>> = indices
                        .iter()
                        .map(|&i| ft.polyvec[i].clone())
                        .collect();
                    
                    let new_ft = Arc::new(PiecewiseLegendreFTVector::from_vector(new_polyvec));
                    
                    Some(Self {
                        inner: FuncsType::FTVector {
                            ft_fermionic: None,
                            ft_bosonic: Some(new_ft),
                            statistics: *statistics,
                        },
                        beta: self.beta,
                    })
                }
            },
        }
    }
}

/// Regularize tau coordinate to [0, beta]
///
/// Handles periodicity for tau functions:
/// - tau ∈ (0, beta] → (tau, sign=1.0)
/// - tau ∈ [-beta, 0) → (tau+beta, sign=fermionic_sign)
/// - tau = 0 → (0, sign=1.0)
///
/// # Arguments
/// * `tau` - Imaginary time coordinate
/// * `beta` - Inverse temperature
/// * `fermionic_sign` - Sign for negative tau: -1.0 for Fermionic, +1.0 for Bosonic
///
/// # Returns
/// (tau_regularized, sign)
fn regularize_tau(tau: f64, beta: f64, fermionic_sign: f64) -> (f64, f64) {
    if tau < -beta || tau > beta {
        panic!("tau {} is outside range [-beta={}, beta={}]", tau, beta, beta);
    }

    if tau > 0.0 && tau <= beta {
        (tau, 1.0)
    } else if tau >= -beta && tau < 0.0 {
        (tau + beta, fermionic_sign)
    } else {
        // tau == 0.0 (or -0.0)
        // Check sign bit for -0.0 vs +0.0
        if tau.is_sign_negative() {
            (beta, fermionic_sign)
        } else {
            (0.0, 1.0)
        }
    }
}

// ============================================================================
// Sampling Type (unified for tau/matsubara/omega sampling)
// ============================================================================

/// Opaque sampling type for C API (compatible with libsparseir)
///
/// Represents sparse sampling in imaginary time (τ), Matsubara frequency (iωn),
/// or real frequency (ω) domains.
///
/// Created by:
/// - `spir_tau_sampling_new()` - τ sampling
/// - `spir_matsu_sampling_new()` - iωn sampling  
///
/// Note: Named `spir_sampling` to match libsparseir C++ API exactly.
#[derive(Clone)]
#[repr(C)]
pub struct spir_sampling {
    pub(crate) inner: SamplingType,
}

/// Internal sampling type (holds different sampling implementations)
#[derive(Clone)]
pub(crate) enum SamplingType {
    /// Tau sampling (real-valued, τ domain)
    TauFermionic(Arc<sparseir_rust::sampling::TauSampling<Fermionic>>),
    TauBosonic(Arc<sparseir_rust::sampling::TauSampling<Bosonic>>),
    
    /// Matsubara sampling (complex-valued, iωn domain)
    MatsubaraFermionic(Arc<sparseir_rust::matsubara_sampling::MatsubaraSampling<Fermionic>>),
    MatsubaraBosonic(Arc<sparseir_rust::matsubara_sampling::MatsubaraSampling<Bosonic>>),
}

