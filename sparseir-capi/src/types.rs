//! Opaque types for C API
//!
//! All Rust objects are wrapped in opaque pointers to hide implementation
//! details from C code.

use std::sync::Arc;
use sparseir_rust::kernel::{LogisticKernel, RegularizedBoseKernel, CentrosymmKernel};
use sparseir_rust::sve::SVEResult;
use sparseir_rust::basis::FiniteTempBasis;
use sparseir_rust::{Bosonic, Fermionic};
use sparseir_rust::traits::Statistics;
use sparseir_rust::poly::PiecewiseLegendrePolyVector;
use sparseir_rust::polyfourier::PiecewiseLegendreFTVector;
use sparseir_rust::freq::MatsubaraFreq;

/// Convert Statistics enum to C-API integer
#[inline]
pub(crate) fn statistics_to_c(stats: Statistics) -> i32 {
    match stats {
        Statistics::Fermionic => 1,
        Statistics::Bosonic => 0,
    }
}

/// Convert C-API integer to Statistics enum
#[inline]
pub(crate) fn statistics_from_c(value: i32) -> Statistics {
    match value {
        1 => Statistics::Fermionic,
        _ => Statistics::Bosonic, // Default to Bosonic for invalid values
    }
}

/// Function domain type for continuous functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FunctionDomain {
    /// Tau domain with periodicity (statistics-dependent)
    Tau(Statistics),
    /// Omega (frequency) domain without periodicity
    Omega,
}

impl FunctionDomain {
    /// Check if this is a tau function with the given statistics
    pub(crate) fn is_tau_with_statistics(&self, stats: Statistics) -> bool {
        matches!(self, FunctionDomain::Tau(s) if *s == stats)
    }
    
    /// Check if this is an omega function
    pub(crate) fn is_omega(&self) -> bool {
        matches!(self, FunctionDomain::Omega)
    }
}

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
    // DLR (Discrete Lehmann Representation) variants
    // Note: DLR always uses LogisticKernel internally, regardless of input kernel type
    DLRFermionic(Arc<sparseir_rust::dlr::DiscreteLehmannRepresentation<Fermionic>>),
    DLRBosonic(Arc<sparseir_rust::dlr::DiscreteLehmannRepresentation<Bosonic>>),
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
            BasisType::DLRFermionic(dlr) => dlr.poles.len(),
            BasisType::DLRBosonic(dlr) => dlr.poles.len(),
        }
    }

    pub(crate) fn svals(&self) -> Vec<f64> {
        match &self.inner {
            BasisType::LogisticFermionic(b) => b.s.clone(),
            BasisType::LogisticBosonic(b) => b.s.clone(),
            BasisType::RegularizedBoseFermionic(b) => b.s.clone(),
            BasisType::RegularizedBoseBosonic(b) => b.s.clone(),
            // DLR: no singular values, return empty
            BasisType::DLRFermionic(_) | BasisType::DLRBosonic(_) => vec![],
        }
    }

    pub(crate) fn statistics(&self) -> i32 {
        // 0 = Bosonic, 1 = Fermionic (matching libsparseir)
        match &self.inner {
            BasisType::LogisticFermionic(_) => 1,
            BasisType::LogisticBosonic(_) => 0,
            BasisType::RegularizedBoseFermionic(_) => 1,
            BasisType::RegularizedBoseBosonic(_) => 0,
            BasisType::DLRFermionic(_) => 1,
            BasisType::DLRBosonic(_) => 0,
        }
    }

    pub(crate) fn beta(&self) -> f64 {
        match &self.inner {
            BasisType::LogisticFermionic(b) => b.beta,
            BasisType::LogisticBosonic(b) => b.beta,
            BasisType::RegularizedBoseFermionic(b) => b.beta,
            BasisType::RegularizedBoseBosonic(b) => b.beta,
            BasisType::DLRFermionic(dlr) => dlr.beta,
            BasisType::DLRBosonic(dlr) => dlr.beta,
        }
    }

    pub(crate) fn wmax(&self) -> f64 {
        match &self.inner {
            BasisType::LogisticFermionic(b) => b.wmax(),
            BasisType::LogisticBosonic(b) => b.wmax(),
            BasisType::RegularizedBoseFermionic(b) => b.wmax(),
            BasisType::RegularizedBoseBosonic(b) => b.wmax(),
            BasisType::DLRFermionic(dlr) => dlr.wmax,
            BasisType::DLRBosonic(dlr) => dlr.wmax,
        }
    }

    pub(crate) fn default_tau_sampling_points(&self) -> Vec<f64> {
        match &self.inner {
            BasisType::LogisticFermionic(b) => b.default_tau_sampling_points(),
            BasisType::LogisticBosonic(b) => b.default_tau_sampling_points(),
            BasisType::RegularizedBoseFermionic(b) => b.default_tau_sampling_points(),
            BasisType::RegularizedBoseBosonic(b) => b.default_tau_sampling_points(),
            // DLR: no default tau sampling points
            BasisType::DLRFermionic(_) | BasisType::DLRBosonic(_) => vec![],
        }
    }

    pub(crate) fn default_matsubara_sampling_points(&self, positive_only: bool) -> Vec<i64> {
        match &self.inner {
            BasisType::LogisticFermionic(b) => b.default_matsubara_sampling_points_i64(positive_only),
            BasisType::LogisticBosonic(b) => b.default_matsubara_sampling_points_i64(positive_only),
            BasisType::RegularizedBoseFermionic(b) => b.default_matsubara_sampling_points_i64(positive_only),
            BasisType::RegularizedBoseBosonic(b) => b.default_matsubara_sampling_points_i64(positive_only),
            // DLR: no default Matsubara sampling points
            BasisType::DLRFermionic(_) | BasisType::DLRBosonic(_) => vec![],
        }
    }

    pub(crate) fn default_omega_sampling_points(&self) -> Vec<f64> {
        match &self.inner {
            BasisType::LogisticFermionic(b) => b.default_omega_sampling_points(),
            BasisType::LogisticBosonic(b) => b.default_omega_sampling_points(),
            BasisType::RegularizedBoseFermionic(b) => b.default_omega_sampling_points(),
            BasisType::RegularizedBoseBosonic(b) => b.default_omega_sampling_points(),
            // DLR: return poles as omega sampling points
            BasisType::DLRFermionic(dlr) => dlr.poles.clone(),
            BasisType::DLRBosonic(dlr) => dlr.poles.clone(),
        }
    }
}

// ============================================================================
// Wrapper types for different function representations
// ============================================================================

/// Wrapper for PiecewiseLegendrePolyVector with domain information
#[derive(Clone)]
pub(crate) struct PolyVectorFuncs {
    pub poly: Arc<PiecewiseLegendrePolyVector>,
    pub domain: FunctionDomain,
}

impl PolyVectorFuncs {
    /// Evaluate all functions at a single point
    pub fn evaluate_at(&self, x: f64, beta: f64) -> Vec<f64> {
        // Regularize x based on domain
        let (x_reg, sign) = match self.domain {
            FunctionDomain::Tau(stats) => {
                // u functions: regularize tau to [0, beta]
                let fermionic_sign = if stats == Statistics::Fermionic { -1.0 } else { 1.0 };
                regularize_tau(x, beta, fermionic_sign)
            },
            FunctionDomain::Omega => {
                // v functions: no regularization needed
                (x, 1.0)
            },
        };
        
        // Evaluate all polynomials at the regularized point
        self.poly.polyvec.iter()
            .map(|p| sign * p.evaluate(x_reg))
            .collect()
    }
    
    /// Batch evaluate all functions at multiple points
    /// Returns Vec<Vec<f64>> where result[i][j] is function i evaluated at point j
    pub fn batch_evaluate_at(&self, xs: &[f64], beta: f64) -> Vec<Vec<f64>> {
        let n_funcs = self.poly.polyvec.len();
        let n_points = xs.len();
        let mut result = vec![vec![0.0; n_points]; n_funcs];
        
        // Regularize all points based on domain
        let regularized: Vec<(f64, f64)> = xs.iter().map(|&x| {
            match self.domain {
                FunctionDomain::Tau(stats) => {
                    let fermionic_sign = if stats == Statistics::Fermionic { -1.0 } else { 1.0 };
                    regularize_tau(x, beta, fermionic_sign)
                },
                FunctionDomain::Omega => (x, 1.0),
            }
        }).collect();
        
        // Extract regularized x values and signs
        let xs_reg: Vec<f64> = regularized.iter().map(|(x, _)| *x).collect();
        let signs: Vec<f64> = regularized.iter().map(|(_, s)| *s).collect();
        
        // Evaluate each polynomial at all regularized points using evaluate_many
        for (i, p) in self.poly.polyvec.iter().enumerate() {
            let values = p.evaluate_many(&xs_reg);
            for (j, &val) in values.iter().enumerate() {
                result[i][j] = signs[j] * val;
            }
        }
        
        result
    }
}

/// Wrapper for Fourier-transformed functions (PiecewiseLegendreFTVector)
#[derive(Clone)]
pub(crate) struct FTVectorFuncs {
    pub ft_fermionic: Option<Arc<PiecewiseLegendreFTVector<Fermionic>>>,
    pub ft_bosonic: Option<Arc<PiecewiseLegendreFTVector<Bosonic>>>,
    pub statistics: Statistics,
}

/// Wrapper for DLR functions in tau domain
#[derive(Clone)]
pub(crate) struct DLRTauFuncs {
    pub poles: Vec<f64>,
    pub beta: f64,
    pub wmax: f64,
    pub inv_weights: Vec<f64>,
    pub statistics: Statistics,
}

impl DLRTauFuncs {
    /// Evaluate all DLR tau functions at a single point
    pub fn evaluate_at(&self, tau: f64) -> Vec<f64> {
        use sparseir_rust::kernel::LogisticKernel;
        
        // Regularize tau to [0, beta]
        let fermionic_sign = if self.statistics == Statistics::Fermionic { -1.0 } else { 1.0 };
        let (tau_reg, sign) = regularize_tau(tau, self.beta, fermionic_sign);
        
        // Compute kernel parameters
        // DLR always uses LogisticKernel
        let lambda = self.beta * self.wmax;
        let kernel = LogisticKernel::new(lambda);
        let x_kern = 2.0 * tau_reg / self.beta - 1.0;  // x_kern ∈ [-1, 1]
        
        // Evaluate DLR functions: u_l(tau) = sign * inv_weight[l] * (-K(x, y_l))
        self.poles.iter().zip(self.inv_weights.iter())
            .map(|(&pole, &inv_weight)| {
                let y = pole / self.wmax;
                let k_val = kernel.compute(x_kern, y);
                sign * (-k_val) * inv_weight
            })
            .collect()
    }
    
    /// Batch evaluate all DLR tau functions at multiple points
    /// Returns Vec<Vec<f64>> where result[i][j] is function i evaluated at point j
    pub fn batch_evaluate_at(&self, taus: &[f64]) -> Vec<Vec<f64>> {
        use sparseir_rust::kernel::LogisticKernel;
        
        let n_funcs = self.poles.len();
        let n_points = taus.len();
        let mut result = vec![vec![0.0; n_points]; n_funcs];
        
        let fermionic_sign = if self.statistics == Statistics::Fermionic { -1.0 } else { 1.0 };
        
        // DLR always uses LogisticKernel
        let lambda = self.beta * self.wmax;
        let kernel = LogisticKernel::new(lambda);
        
        // Evaluate at each point
        for (j, &tau) in taus.iter().enumerate() {
            let (tau_reg, sign) = regularize_tau(tau, self.beta, fermionic_sign);
            let x_kern = 2.0 * tau_reg / self.beta - 1.0;
            
            for (i, (&pole, &inv_weight)) in self.poles.iter().zip(self.inv_weights.iter()).enumerate() {
                let y = pole / self.wmax;
                let k_val = kernel.compute(x_kern, y);
                result[i][j] = sign * (-k_val) * inv_weight;
            }
        }
        
        result
    }
}

/// Wrapper for DLR functions in Matsubara domain
#[derive(Clone)]
pub(crate) struct DLRMatsubaraFuncs {
    pub poles: Vec<f64>,
    pub beta: f64,
    pub inv_weights: Vec<f64>,
    pub statistics: Statistics,
}

// ============================================================================
// Internal enum to hold different function types
// ============================================================================

/// Internal enum to hold different function types
#[derive(Clone)]
pub(crate) enum FuncsType {
    /// Continuous functions (u or v): PiecewiseLegendrePolyVector
    PolyVector(PolyVectorFuncs),
    
    /// Fourier-transformed functions (uhat): PiecewiseLegendreFTVector
    FTVector(FTVectorFuncs),
    
    /// DLR functions in tau domain (discrete poles)
    DLRTau(DLRTauFuncs),
    
    /// DLR functions in Matsubara domain (discrete poles)
    DLRMatsubara(DLRMatsubaraFuncs),
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
            inner: FuncsType::PolyVector(PolyVectorFuncs {
                poly,
                domain: FunctionDomain::Tau(Statistics::Fermionic),
            }),
            beta,
        }
    }

    /// Create u funcs (tau-domain, Bosonic)
    pub(crate) fn from_u_bosonic(poly: Arc<PiecewiseLegendrePolyVector>, beta: f64) -> Self {
        Self {
            inner: FuncsType::PolyVector(PolyVectorFuncs {
                poly,
                domain: FunctionDomain::Tau(Statistics::Bosonic),
            }),
            beta,
        }
    }

    /// Create v funcs (omega-domain, no statistics)
    pub(crate) fn from_v(poly: Arc<PiecewiseLegendrePolyVector>, beta: f64) -> Self {
        Self {
            inner: FuncsType::PolyVector(PolyVectorFuncs {
                poly,
                domain: FunctionDomain::Omega,
            }),
            beta,
        }
    }

    /// Create uhat funcs (Matsubara-domain, Fermionic)
    pub(crate) fn from_uhat_fermionic(ft: Arc<PiecewiseLegendreFTVector<Fermionic>>, beta: f64) -> Self {
        Self {
            inner: FuncsType::FTVector(FTVectorFuncs {
                ft_fermionic: Some(ft),
                ft_bosonic: None,
                statistics: Statistics::Fermionic,
            }),
            beta,
        }
    }

    /// Create uhat funcs (Matsubara-domain, Bosonic)
    pub(crate) fn from_uhat_bosonic(ft: Arc<PiecewiseLegendreFTVector<Bosonic>>, beta: f64) -> Self {
        Self {
            inner: FuncsType::FTVector(FTVectorFuncs {
                ft_fermionic: None,
                ft_bosonic: Some(ft),
                statistics: Statistics::Bosonic,
            }),
            beta,
        }
    }

    /// Create DLR tau funcs (tau-domain, Fermionic)
    /// Note: DLR always uses LogisticKernel regardless of the IR basis kernel type
    pub(crate) fn from_dlr_tau_fermionic(poles: Vec<f64>, beta: f64, wmax: f64, inv_weights: Vec<f64>) -> Self {
        Self {
            inner: FuncsType::DLRTau(DLRTauFuncs {
                poles,
                beta,
                wmax,
                inv_weights,
                statistics: Statistics::Fermionic,
            }),
            beta,
        }
    }

    /// Create DLR tau funcs (tau-domain, Bosonic)
    /// Note: DLR always uses LogisticKernel regardless of the IR basis kernel type
    pub(crate) fn from_dlr_tau_bosonic(poles: Vec<f64>, beta: f64, wmax: f64, inv_weights: Vec<f64>) -> Self {
        Self {
            inner: FuncsType::DLRTau(DLRTauFuncs {
                poles,
                beta,
                wmax,
                inv_weights,
                statistics: Statistics::Bosonic,
            }),
            beta,
        }
    }

    /// Create DLR Matsubara funcs (Matsubara-domain, Fermionic)
    pub(crate) fn from_dlr_matsubara_fermionic(poles: Vec<f64>, beta: f64, inv_weights: Vec<f64>) -> Self {
        Self {
            inner: FuncsType::DLRMatsubara(DLRMatsubaraFuncs {
                poles,
                beta,
                inv_weights,
                statistics: Statistics::Fermionic,
            }),
            beta,
        }
    }

    /// Create DLR Matsubara funcs (Matsubara-domain, Bosonic)
    pub(crate) fn from_dlr_matsubara_bosonic(poles: Vec<f64>, beta: f64, inv_weights: Vec<f64>) -> Self {
        Self {
            inner: FuncsType::DLRMatsubara(DLRMatsubaraFuncs {
                poles,
                beta,
                inv_weights,
                statistics: Statistics::Bosonic,
            }),
            beta,
        }
    }

    /// Get the number of basis functions
    pub(crate) fn size(&self) -> usize {
        match &self.inner {
            FuncsType::PolyVector(pv) => pv.poly.polyvec.len(),
            FuncsType::FTVector(ftv) => {
                if let Some(ft) = &ftv.ft_fermionic {
                    ft.polyvec.len()
                } else if let Some(ft) = &ftv.ft_bosonic {
                    ft.polyvec.len()
                } else {
                    0
                }
            },
            FuncsType::DLRTau(dlr) => dlr.poles.len(),
            FuncsType::DLRMatsubara(dlr) => dlr.poles.len(),
        }
    }

    /// Get knots for continuous functions (PolyVector only)
    pub(crate) fn knots(&self) -> Option<Vec<f64>> {
        match &self.inner {
            FuncsType::PolyVector(pv) => {
                // Get unique knots from all polynomials
                let mut all_knots = Vec::new();
                for p in &pv.poly.polyvec {
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
            FuncsType::PolyVector(pv) => {
                Some(pv.evaluate_at(x, self.beta))
            },
            FuncsType::DLRTau(dlr) => {
                Some(dlr.evaluate_at(x))
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
            FuncsType::FTVector(ftv) => {
                if ftv.statistics == Statistics::Fermionic {
                    // Fermionic
                    let ft = ftv.ft_fermionic.as_ref()?;
                    let freq = MatsubaraFreq::<Fermionic>::new(n).ok()?;
                    let mut result = Vec::with_capacity(ft.polyvec.len());
                    for p in &ft.polyvec {
                        result.push(p.evaluate(&freq));
                    }
                    Some(result)
                } else {
                    // Bosonic
                    let ft = ftv.ft_bosonic.as_ref()?;
                    let freq = MatsubaraFreq::<Bosonic>::new(n).ok()?;
                    let mut result = Vec::with_capacity(ft.polyvec.len());
                    for p in &ft.polyvec {
                        result.push(p.evaluate(&freq));
                    }
                    Some(result)
                }
            },
            FuncsType::DLRMatsubara(dlr) => {
                // Evaluate DLR Matsubara functions: uhat_l(iν_n) = inv_weight[l] / (iν_n - pole_l)
                use num_complex::Complex;
                
                let mut result = Vec::with_capacity(dlr.poles.len());
                if dlr.statistics == Statistics::Fermionic {
                    // Fermionic
                    let freq = MatsubaraFreq::<Fermionic>::new(n).ok()?;
                    let iv = freq.value_imaginary(dlr.beta);
                    for (i, &pole) in dlr.poles.iter().enumerate() {
                        let inv_weight = dlr.inv_weights[i];
                        result.push(Complex::new(inv_weight, 0.0) / (iv - Complex::new(pole, 0.0)));
                    }
                } else {
                    // Bosonic
                    let freq = MatsubaraFreq::<Bosonic>::new(n).ok()?;
                    let iv = freq.value_imaginary(dlr.beta);
                    for (i, &pole) in dlr.poles.iter().enumerate() {
                        let inv_weight = dlr.inv_weights[i];
                        result.push(Complex::new(inv_weight, 0.0) / (iv - Complex::new(pole, 0.0)));
                    }
                }
                Some(result)
            },
            _ => None,
        }
    }

    /// Batch evaluate at multiple tau/omega points
    pub(crate) fn batch_eval_continuous(&self, xs: &[f64]) -> Option<Vec<Vec<f64>>> {
        match &self.inner {
            FuncsType::PolyVector(pv) => {
                Some(pv.batch_evaluate_at(xs, self.beta))
            },
            FuncsType::DLRTau(dlr) => {
                Some(dlr.batch_evaluate_at(xs))
            },
            _ => None,
        }
    }

    /// Batch evaluate at multiple Matsubara frequencies (for FT functions only)
    ///
    /// # Arguments
    /// * `ns` - Matsubara frequency indices
    ///
    /// # Returns
    /// Matrix of complex function values (size = `[n_funcs, n_freqs]`), or None if not FT type
    pub(crate) fn batch_eval_matsubara(&self, ns: &[i64]) -> Option<Vec<Vec<num_complex::Complex64>>> {
        match &self.inner {
            FuncsType::FTVector(ftv) => {
                if ftv.statistics == Statistics::Fermionic {
                    // Fermionic
                    let ft = ftv.ft_fermionic.as_ref()?;
                    let n_funcs = ft.polyvec.len();
                    let n_points = ns.len();
                    let mut result = vec![vec![num_complex::Complex64::new(0.0, 0.0); n_points]; n_funcs];
                    
                    for (j, &n) in ns.iter().enumerate() {
                        let freq = MatsubaraFreq::<Fermionic>::new(n).ok()?;
                        for (i, p) in ft.polyvec.iter().enumerate() {
                            result[i][j] = p.evaluate(&freq);
                        }
                    }
                    Some(result)
                } else {
                    // Bosonic
                    let ft = ftv.ft_bosonic.as_ref()?;
                    let n_funcs = ft.polyvec.len();
                    let n_points = ns.len();
                    let mut result = vec![vec![num_complex::Complex64::new(0.0, 0.0); n_points]; n_funcs];
                    
                    for (j, &n) in ns.iter().enumerate() {
                        let freq = MatsubaraFreq::<Bosonic>::new(n).ok()?;
                        for (i, p) in ft.polyvec.iter().enumerate() {
                            result[i][j] = p.evaluate(&freq);
                        }
                    }
                    Some(result)
                }
            },
            FuncsType::DLRMatsubara(dlr) => {
                // Batch evaluate DLR Matsubara functions: uhat_l(iν_n) = inv_weight[l] / (iν_n - pole_l)
                use num_complex::Complex;
                
                let n_funcs = dlr.poles.len();
                let n_points = ns.len();
                let mut result = vec![vec![Complex::new(0.0, 0.0); n_points]; n_funcs];
                
                for (j, &n) in ns.iter().enumerate() {
                    if dlr.statistics == Statistics::Fermionic {
                        // Fermionic
                        let freq = MatsubaraFreq::<Fermionic>::new(n).ok()?;
                        let iv = freq.value_imaginary(dlr.beta);
                        for (i, &pole) in dlr.poles.iter().enumerate() {
                            let inv_weight = dlr.inv_weights[i];
                            result[i][j] = Complex::new(inv_weight, 0.0) / (iv - Complex::new(pole, 0.0));
                        }
                    } else {
                        // Bosonic
                        let freq = MatsubaraFreq::<Bosonic>::new(n).ok()?;
                        let iv = freq.value_imaginary(dlr.beta);
                        for (i, &pole) in dlr.poles.iter().enumerate() {
                            let inv_weight = dlr.inv_weights[i];
                            result[i][j] = Complex::new(inv_weight, 0.0) / (iv - Complex::new(pole, 0.0));
                        }
                    }
                }
                
                Some(result)
            },
            FuncsType::DLRTau(_) => {
                // DLRTau is for tau, not Matsubara frequencies
                None
            },
            _ => None,
        }
    }

    /// Extract a slice of functions by indices (creates a new subset)
    ///
    /// # Arguments
    /// * `indices` - Indices of functions to extract
    ///
    /// # Returns
    /// New funcs object with the selected subset, or None if operation not supported
    pub(crate) fn get_slice(&self, indices: &[usize]) -> Option<Self> {
        match &self.inner {
            FuncsType::PolyVector(pv) => {
                let mut new_polys = Vec::with_capacity(indices.len());
                for &idx in indices {
                    if idx >= pv.poly.polyvec.len() {
                        return None;
                    }
                    new_polys.push(pv.poly.polyvec[idx].clone());
                }
                let new_poly_vec = PiecewiseLegendrePolyVector::new(new_polys);
                Some(Self {
                    inner: FuncsType::PolyVector(PolyVectorFuncs {
                        poly: Arc::new(new_poly_vec),
                        domain: pv.domain,
                    }),
                    beta: self.beta,
                })
            },
            FuncsType::FTVector(_) => {
                // FTVector slicing not yet supported (requires public constructor)
                None
            },
            FuncsType::DLRTau(dlr) => {
                // Select subset of poles
                let mut new_poles = Vec::with_capacity(indices.len());
                let mut new_inv_weights = Vec::with_capacity(indices.len());
                for &idx in indices {
                    if idx >= dlr.poles.len() {
                        return None;
                    }
                    new_poles.push(dlr.poles[idx]);
                    new_inv_weights.push(dlr.inv_weights[idx]);
                }
                Some(Self {
                    inner: FuncsType::DLRTau(DLRTauFuncs {
                        poles: new_poles,
                        beta: dlr.beta,
                        wmax: dlr.wmax,
                        inv_weights: new_inv_weights,
                        statistics: dlr.statistics,
                    }),
                    beta: dlr.beta,
                })
            },
            FuncsType::DLRMatsubara(dlr) => {
                // Select subset of poles
                let mut new_poles = Vec::with_capacity(indices.len());
                let mut new_inv_weights = Vec::with_capacity(indices.len());
                for &idx in indices {
                    if idx >= dlr.poles.len() {
                        return None;
                    }
                    new_poles.push(dlr.poles[idx]);
                    new_inv_weights.push(dlr.inv_weights[idx]);
                }
                Some(Self {
                    inner: FuncsType::DLRMatsubara(DLRMatsubaraFuncs {
                        poles: new_poles,
                        beta: dlr.beta,
                        inv_weights: new_inv_weights,
                        statistics: dlr.statistics,
                    }),
                    beta: dlr.beta,
                })
            },
        }
    }
}

// Helper function for tau regularization (same as C++)
fn regularize_tau(tau: f64, beta: f64, fermionic_sign: f64) -> (f64, f64) {
    // C++: libsparseir/include/sparseir/funcs.hpp:52-73
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_funcs_creation() {
        // Basic test that funcs types can be created
        // More comprehensive tests should be in integration tests
    }
}
/// Sampling type for C API (unified type for all domains)
///
/// This wraps different sampling implementations:
/// - TauSampling (for tau-domain)
/// - MatsubaraSampling (for Matsubara frequencies, full range or positive-only)
#[derive(Clone)]
#[repr(C)]
pub struct spir_sampling {
    pub(crate) inner: SamplingType,
}

/// Internal enum to distinguish between different sampling types
#[derive(Clone)]
pub(crate) enum SamplingType {
    TauFermionic(Arc<sparseir_rust::sampling::TauSampling<Fermionic>>),
    TauBosonic(Arc<sparseir_rust::sampling::TauSampling<Bosonic>>),
    MatsubaraFermionic(Arc<sparseir_rust::matsubara_sampling::MatsubaraSampling<Fermionic>>),
    MatsubaraBosonic(Arc<sparseir_rust::matsubara_sampling::MatsubaraSampling<Bosonic>>),
    MatsubaraPositiveOnlyFermionic(Arc<sparseir_rust::matsubara_sampling::MatsubaraSamplingPositiveOnly<Fermionic>>),
    MatsubaraPositiveOnlyBosonic(Arc<sparseir_rust::matsubara_sampling::MatsubaraSamplingPositiveOnly<Bosonic>>),
}

#[cfg(test)]
mod sampling_tests {
    use super::*;
    use sparseir_rust::basis::FiniteTempBasis;
    use sparseir_rust::kernel::LogisticKernel;
    use sparseir_rust::traits::{Fermionic, Bosonic};
    
    #[test]
    fn test_sampling_creation() {
        // Basic test that sampling types can be created
        // More comprehensive tests should be in integration tests
    }
}
/// Spir error codes (compatible with libsparseir)
pub type StatusCode = i32;

/// Success code (0)
pub const SPIR_SUCCESS: StatusCode = 0;
/// Computation completed successfully (0, alias for SUCCESS)
pub const SPIR_COMPUTATION_SUCCESS: StatusCode = 0;
/// Invalid argument error (-6)
pub const SPIR_INVALID_ARGUMENT: StatusCode = -6;
/// Internal error (-7)
pub const SPIR_INTERNAL_ERROR: StatusCode = -7;
/// Not supported error (-5)
pub const SPIR_NOT_SUPPORTED: StatusCode = -5;

