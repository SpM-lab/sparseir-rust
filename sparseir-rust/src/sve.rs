//! Singular Value Expansion (SVE) module
//! 
//! This module provides functionality for computing the singular value expansion
//! of integral kernels, which is the core of the sparseir algorithm.

use ndarray::{Array2, Array1};
use crate::{
    poly::{PiecewiseLegendrePolyVector, PiecewiseLegendrePoly},
    kernel::{AbstractKernel, KernelProperties, SVEHints},
    numeric::CustomNumeric,
    gauss::{Rule, legendre},
    DiscretizedKernel, matrix_from_gauss,
};
use num_traits::ToPrimitive;
use xprec_svd::{tsvd_f64, tsvd_twofloat_from_f64};

/// Working precision type for SVE computations
/// 
/// Values match the C-API constants defined in sparseir.h
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TworkType {
    /// Use double precision (64-bit)
    Float64 = 0,      // SPIR_TWORK_FLOAT64
    /// Use extended precision (128-bit double-double)
    Float64X2 = 1,    // SPIR_TWORK_FLOAT64X2
    /// Automatically choose precision based on epsilon
    Auto = -1,        // SPIR_TWORK_AUTO
}

/// SVD computation strategy
/// 
/// Values match the C-API constants defined in sparseir.h
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SVDStrategy {
    /// Fast computation
    Fast = 0,         // SPIR_SVDSTRAT_FAST
    /// Accurate computation
    Accurate = 1,     // SPIR_SVDSTRAT_ACCURATE
    /// Automatically choose strategy
    Auto = -1,        // SPIR_SVDSTRAT_AUTO
}

/// Result of Singular Value Expansion computation
#[derive(Debug, Clone)]
pub struct SVEResult {
    /// Left singular functions (u)
    pub u: PiecewiseLegendrePolyVector,
    /// Singular values in non-increasing order
    pub s: Array1<f64>,
    /// Right singular functions (v)
    pub v: PiecewiseLegendrePolyVector,
    /// Accuracy parameter used for computation
    pub epsilon: f64,
}

impl SVEResult {
    /// Create a new SVEResult
    pub fn new(
        u: PiecewiseLegendrePolyVector,
        s: Array1<f64>,
        v: PiecewiseLegendrePolyVector,
        epsilon: f64,
    ) -> Self {
        Self { u, s, v, epsilon }
    }

    /// Extract a subset of the SVE result based on epsilon and max_size
    pub fn part(&self, eps: Option<f64>, max_size: Option<usize>) -> (PiecewiseLegendrePolyVector, Array1<f64>, PiecewiseLegendrePolyVector) {
        let eps = eps.unwrap_or(self.epsilon);
        let threshold = eps * self.s[0];

        let mut cut = 0;
        for &val in self.s.iter() {
            if val >= threshold {
                cut += 1;
            } else {
                break;
            }
        }

        if let Some(max) = max_size {
            cut = cut.min(max);
        }

        // Extract subsets
        let u_part = PiecewiseLegendrePolyVector::new(
            self.u.get_polys()[..cut].to_vec()
        );
        let s_part = self.s.slice(ndarray::s![..cut]).to_owned();
        let v_part = PiecewiseLegendrePolyVector::new(
            self.v.get_polys()[..cut].to_vec()
        );

        (u_part, s_part, v_part)
    }
}

/// Compute SVD of a matrix
/// 
/// Returns (U, singular_values, V^T) where A = U * S * V^T
pub fn compute_svd<T: CustomNumeric + 'static>(matrix: &Array2<T>) -> (Array2<T>, Array1<T>, Array2<T>) {
    // Use different SVD implementations based on type
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // For f64, use xprec-svd for better precision
        compute_svd_f64_xprec(matrix)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<twofloat::TwoFloat>() {
        // For TwoFloat, use xprec-svd with TwoFloat precision
        compute_svd_twofloat_xprec(matrix)
    } else {
        // For other types, fall back to f64 computation
        compute_svd_f64_xprec(matrix)
    }
}

/// Compute SVD for f64 using xprec-svd
fn compute_svd_f64_xprec<T: CustomNumeric>(matrix: &Array2<T>) -> (Array2<T>, Array1<T>, Array2<T>) {
    // Convert to f64 matrix for xprec-svd
    let matrix_f64: Array2<f64> = matrix.map(|&x| x.to_f64());
    
    // Compute truncated SVD using xprec-svd with high tolerance
    let rtol = 1e-15; // High precision tolerance
    let result = tsvd_f64(&matrix_f64, rtol).expect("SVD computation failed");
    
    // Convert results back to T type
    let u = result.u.map(|&x| <T as CustomNumeric>::from_f64(x));
    let s = result.s.map(|&x| <T as CustomNumeric>::from_f64(x));
    let vt = result.v.map(|&x| <T as CustomNumeric>::from_f64(x));
    
    (u, s, vt)
}

/// Compute SVD for TwoFloat using xprec-svd
fn compute_svd_twofloat_xprec<T: CustomNumeric>(matrix: &Array2<T>) -> (Array2<T>, Array1<T>, Array2<T>) {
    // Convert to f64 for xprec-svd input
    let matrix_f64: Array2<f64> = matrix.map(|&x| x.to_f64());
    
    // Compute truncated SVD using xprec-svd with TwoFloat precision
    let rtol = 1e-15; // High precision tolerance
    let result = tsvd_twofloat_from_f64(&matrix_f64, rtol).expect("TwoFloat SVD computation failed");
    
    // Convert TwoFloatPrecision results back to T
    let u = result.u.map(|&x| <T as CustomNumeric>::from_f64(x.to_f64()));
    let s = result.s.map(|&x| <T as CustomNumeric>::from_f64(x.to_f64()));
    let vt = result.v.map(|&x| <T as CustomNumeric>::from_f64(x.to_f64()));
    
    (u, s, vt)
}

/// Truncate SVD results based on cutoff and maximum size
pub fn truncate<T: CustomNumeric>(
    u_list: Vec<Array2<T>>,
    s_list: Vec<Array1<T>>,
    v_list: Vec<Array2<T>>,
    cutoff: T,
    lmax: Option<usize>,
) -> (Vec<Array2<T>>, Vec<Array1<T>>, Vec<Array2<T>>) {
    let mut truncated_u = Vec::new();
    let mut truncated_s = Vec::new();
    let mut truncated_v = Vec::new();
    
    for ((u, s), v) in u_list.into_iter().zip(s_list.into_iter()).zip(v_list.into_iter()) {
        // Find cutoff point
        let mut cut = 0;
        for &val in s.iter() {
            if val >= cutoff {
                cut += 1;
            } else {
                break;
            }
        }
        
        // Apply maximum size limit
        if let Some(max) = lmax {
            cut = cut.min(max);
        }
        
        // Truncate
        truncated_u.push(u.slice(ndarray::s![.., ..cut]).to_owned());
        truncated_s.push(s.slice(ndarray::s![..cut]).to_owned());
        truncated_v.push(v.slice(ndarray::s![..cut, ..]).to_owned());
    }
    
    (truncated_u, truncated_s, truncated_v)
}

/// Compute safe epsilon value and determine working precision
fn safe_epsilon(epsilon: f64, twork: TworkType, _svd_strategy: SVDStrategy) -> (f64, TworkType, SVDStrategy) {
    let safe_epsilon = epsilon.max(1e-15); // Ensure minimum precision
    
    let twork_actual = match twork {
        TworkType::Auto => {
            if safe_epsilon < 1e-12 {
                TworkType::Float64X2
            } else {
                TworkType::Float64
            }
        }
        other => other,
    };
    
    let svd_strategy_actual = SVDStrategy::Accurate; // For now, always use accurate
    
    (safe_epsilon, twork_actual, svd_strategy_actual)
}

/// Determine the appropriate SVE strategy based on kernel properties
fn determine_sve<T, K>(
    kernel: K,
    safe_epsilon: f64,
    n_gauss: Option<usize>,
) -> Box<dyn SVEStrategy<T>>
where
    T: CustomNumeric + Send + Sync + num_traits::Zero + ToPrimitive + 'static,
    K: AbstractKernel + KernelProperties + Clone + 'static,
    <K as KernelProperties>::SVEHintsType<T>: SVEHints<T> + Clone + 'static,
{
    // Get SVE hints from the kernel
    let hints = kernel.sve_hints(safe_epsilon);
    
    // Check if the kernel is centrosymmetric
    if kernel.is_centrosymmetric() {
        // For centrosymmetric kernels, use CentrosymmSVE
        Box::new(CentrosymmSVE::new(kernel, hints, safe_epsilon, n_gauss))
    } else {
        // For non-centrosymmetric kernels, use SamplingSVE
        Box::new(SamplingSVE::new(kernel, hints, safe_epsilon, n_gauss))
    }
}

/// Trait for SVE computation strategies
pub trait SVEStrategy<T: CustomNumeric> {
    /// Compute the discretized matrices
    fn matrices(&self) -> Vec<Array2<T>>;
    
    /// Post-process SVD results to create SVEResult
    fn postprocess(
        &self,
        u_list: Vec<Array2<T>>,
        s_list: Vec<Array1<T>>,
        v_list: Vec<Array2<T>>,
    ) -> SVEResult;
}

/// Pre-process and post-process SVE computation
fn pre_postprocess<T, K>(
    kernel: K,
    safe_epsilon: f64,
    n_gauss: Option<usize>,
    cutoff: Option<f64>,
    lmax: Option<usize>,
) -> (SVEResult, Box<dyn SVEStrategy<T>>)
where
    T: CustomNumeric + Send + Sync + num_traits::Zero + ToPrimitive + 'static,
    K: AbstractKernel + KernelProperties + Clone + 'static,
    <K as KernelProperties>::SVEHintsType<T>: SVEHints<T> + Clone + 'static,
{
    let sve = determine_sve::<T, K>(kernel, safe_epsilon, n_gauss);
    
    // Compute discretized matrices
    let matrices = sve.matrices();
    
    // Compute SVD for each matrix
    let mut u_list = Vec::new();
    let mut s_list = Vec::new();
    let mut v_list = Vec::new();
    
    for matrix in matrices {
        let (u, s, v) = compute_svd(&matrix);
        u_list.push(u);
        s_list.push(s);
        v_list.push(v);
    }
    
    // Determine cutoff
    let cutoff_actual = cutoff.unwrap_or(2.0 * f64::EPSILON);
    let cutoff_t = <T as CustomNumeric>::from_f64(cutoff_actual);
    
    // Truncate results
    let (u_truncated, s_truncated, v_truncated) = truncate(
        u_list,
        s_list,
        v_list,
        cutoff_t,
        lmax,
    );
    
    // Post-process to create SVEResult
    let result = sve.postprocess(u_truncated, s_truncated, v_truncated);
    
    (result, sve)
}

/// Main SVE computation function
/// 
/// This function orchestrates the SVE computation by choosing the appropriate
/// strategy based on kernel properties and computing parameters.
pub fn compute_sve<T, K>(
    kernel: K,
    epsilon: f64,
    cutoff: Option<f64>,
    lmax: Option<usize>,
    n_gauss: Option<usize>,
    twork: TworkType,
) -> SVEResult
where
    T: CustomNumeric + Send + Sync + num_traits::Zero + ToPrimitive + 'static,
    K: AbstractKernel + KernelProperties + Clone + 'static,
    <K as KernelProperties>::SVEHintsType<T>: SVEHints<T> + Clone + 'static,
{
    // Compute safe epsilon and determine working precision
    let (safe_epsilon, twork_actual, _svd_strategy_actual) = safe_epsilon(epsilon, twork, SVDStrategy::Auto);
    
    // For now, only support f64 precision
    // TODO: Add proper TwoFloat support when num_traits are available
    if twork_actual == TworkType::Float64 || twork_actual == TworkType::Float64X2 {
        pre_postprocess::<f64, K>(kernel, safe_epsilon, n_gauss, cutoff, lmax).0
    } else {
        panic!("Invalid Twork type: {:?}", twork_actual);
    }
}

/// Sampling-based SVE computation
/// 
/// This is the main SVE computation strategy that discretizes the kernel
/// on Gauss quadrature points and computes the SVD of the resulting matrix.
pub struct SamplingSVE<T: CustomNumeric + Send + Sync, K: AbstractKernel + KernelProperties + Clone, H: SVEHints<T> + Clone> {
    /// The kernel to expand
    kernel: K,
    /// Accuracy parameter
    epsilon: f64,
    /// Number of Gauss points per segment
    n_gauss: usize,
    /// SVE hints for discretization
    #[allow(dead_code)]
    hints: H,
    /// Gauss quadrature rule
    #[allow(dead_code)]
    rule: Rule<T>,
    /// Segments for x coordinate
    segs_x: Vec<T>,
    /// Segments for y coordinate
    segs_y: Vec<T>,
    /// Gauss points for x coordinate
    gauss_x: DiscretizedKernel<T>,
    /// Gauss points for y coordinate
    gauss_y: DiscretizedKernel<T>,
}

impl<T: CustomNumeric + Send + Sync + num_traits::Zero + ToPrimitive, K: AbstractKernel + KernelProperties + Clone, H: SVEHints<T> + Clone> SamplingSVE<T, K, H> {
    /// Create a new SamplingSVE
    pub fn new(kernel: K, hints: H, epsilon: f64, n_gauss: Option<usize>) -> Self {
        let n_gauss = n_gauss.unwrap_or(hints.ngauss());
        
        // Create Gauss quadrature rule
        let rule_f64 = legendre(n_gauss);
        let rule = Rule::<T>::from_vectors(
            rule_f64.x.to_vec().iter().map(|&x| <T as CustomNumeric>::from_f64(x)).collect(),
            rule_f64.w.to_vec().iter().map(|&w| <T as CustomNumeric>::from_f64(w)).collect(),
            <T as CustomNumeric>::from_f64(rule_f64.a),
            <T as CustomNumeric>::from_f64(rule_f64.b),
        );
        
        let segs_x = hints.segments_x();
        let segs_y = hints.segments_y();
        
        let gauss_x_rule = rule.piecewise(&segs_x);
        let gauss_y_rule = rule.piecewise(&segs_y);
        
        // Create dummy matrices for DiscretizedKernel
        let dummy_matrix_x = Array2::zeros((gauss_x_rule.x.len(), 1));
        let dummy_matrix_y = Array2::zeros((gauss_y_rule.x.len(), 1));
        
        let gauss_x = DiscretizedKernel::new(
            dummy_matrix_x,
            gauss_x_rule.clone(),
            gauss_y_rule.clone(),
        );
        let gauss_y = DiscretizedKernel::new(
            dummy_matrix_y,
            gauss_x_rule,
            gauss_y_rule,
        );
        
        Self {
            kernel,
            epsilon,
            n_gauss,
            hints,
            rule,
            segs_x,
            segs_y,
            gauss_x,
            gauss_y,
        }
    }
    
    /// Convert SVD matrix to polynomial representation
    fn svd_to_polynomials(
        &self,
        svd_matrix: &Array2<T>,
        segments: &[T],
        _gauss_rule: &crate::gauss::Rule<f64>,
    ) -> Vec<PiecewiseLegendrePoly> {
        let mut polynomials = Vec::new();
        
        for col in 0..svd_matrix.ncols() {
            let coeffs = svd_matrix.column(col).to_owned();
            
            // Convert coefficients to f64
            let coeffs_f64 = coeffs.map(|&x| x.to_f64());
            
            // Create polynomial
            let coeffs_2d = coeffs_f64.clone().into_shape((coeffs_f64.len(), 1)).unwrap().to_owned();
            let poly = PiecewiseLegendrePoly::new(
                coeffs_2d,
                segments.iter().map(|&x| x.to_f64()).collect(),
                0,
                None,
                col as i32,
            );
            
            polynomials.push(poly);
        }
        
        polynomials
    }
}

impl<T: CustomNumeric + Send + Sync + num_traits::Zero + ToPrimitive, K: AbstractKernel + KernelProperties + Clone, H: SVEHints<T> + Clone> SVEStrategy<T> for SamplingSVE<T, K, H> {
    fn matrices(&self) -> Vec<Array2<T>> {
        // Compute kernel matrix using Gauss quadrature
        let discretized = matrix_from_gauss(
            &self.kernel,
            &self.gauss_x.gauss_x,
            &self.gauss_x.gauss_y,
        );
        
        // Apply weights (sqrt of Gauss weights)
        let weighted_matrix = discretized.apply_weights_for_sve();
        
        vec![weighted_matrix]
    }
    
    fn postprocess(
        &self,
        u_list: Vec<Array2<T>>,
        s_list: Vec<Array1<T>>,
        v_list: Vec<Array2<T>>,
    ) -> SVEResult {
        let u = &u_list[0];
        let s = &s_list[0];
        let v = &v_list[0];
        
        // Convert singular values to f64
        let s_f64 = s.map(|&x| x.to_f64());
        
        // Create polynomial vectors from SVD results
        // Convert T rules to f64 rules for polynomial creation
        let gauss_x_f64 = Rule::<f64>::from_vectors(
            self.gauss_x.gauss_x.x.iter().map(|&x| x.to_f64()).collect(),
            self.gauss_x.gauss_x.w.iter().map(|&w| w.to_f64()).collect(),
            self.gauss_x.gauss_x.a.to_f64(),
            self.gauss_x.gauss_x.b.to_f64(),
        );
        let gauss_y_f64 = Rule::<f64>::from_vectors(
            self.gauss_y.gauss_y.x.iter().map(|&x| x.to_f64()).collect(),
            self.gauss_y.gauss_y.w.iter().map(|&w| w.to_f64()).collect(),
            self.gauss_y.gauss_y.a.to_f64(),
            self.gauss_y.gauss_y.b.to_f64(),
        );
        
        let u_polys = self.svd_to_polynomials(u, &self.segs_x, &gauss_x_f64);
        let v_polys = self.svd_to_polynomials(v, &self.segs_y, &gauss_y_f64);
        
        let u_vec = PiecewiseLegendrePolyVector::new(u_polys);
        let v_vec = PiecewiseLegendrePolyVector::new(v_polys);
        
        SVEResult::new(u_vec, s_f64, v_vec, self.epsilon)
    }
}


/// Centrosymmetric SVE computation
/// 
/// Optimized SVE computation for centrosymmetric kernels that can be
/// block-diagonalized to reduce computational cost.
pub struct CentrosymmSVE<T: CustomNumeric + Send + Sync, K: AbstractKernel + KernelProperties + Clone, H: SVEHints<T> + Clone> {
    /// The centrosymmetric kernel
    #[allow(dead_code)]
    kernel: K,
    /// Accuracy parameter
    #[allow(dead_code)]
    epsilon: f64,
    /// Number of Gauss points per segment
    #[allow(dead_code)]
    n_gauss: usize,
    /// Even symmetry SVE
    even_sve: SamplingSVE<T, K, H>,
    /// Odd symmetry SVE
    #[allow(dead_code)]
    odd_sve: SamplingSVE<T, K, H>,
}

impl<T: CustomNumeric + Send + Sync + num_traits::Zero + ToPrimitive, K: AbstractKernel + KernelProperties + Clone, H: SVEHints<T> + Clone> CentrosymmSVE<T, K, H> {
    /// Create a new CentrosymmSVE
    pub fn new(kernel: K, hints: H, epsilon: f64, n_gauss: Option<usize>) -> Self {
        // For now, create two identical SamplingSVE instances
        // TODO: Implement proper centrosymmetric decomposition
        let even_sve = SamplingSVE::new(kernel.clone(), hints.clone(), epsilon, n_gauss);
        let odd_sve = SamplingSVE::new(kernel.clone(), hints.clone(), epsilon, n_gauss);
        
        Self {
            kernel,
            epsilon,
            n_gauss: even_sve.n_gauss,
            even_sve,
            odd_sve,
        }
    }
}

impl<T: CustomNumeric + Send + Sync + num_traits::Zero + ToPrimitive, K: AbstractKernel + KernelProperties + Clone, H: SVEHints<T> + Clone> SVEStrategy<T> for CentrosymmSVE<T, K, H> {
    fn matrices(&self) -> Vec<Array2<T>> {
        // For now, just return even matrices
        // TODO: Implement proper centrosymmetric decomposition
        self.even_sve.matrices()
    }
    
    fn postprocess(
        &self,
        u_list: Vec<Array2<T>>,
        s_list: Vec<Array1<T>>,
        v_list: Vec<Array2<T>>,
    ) -> SVEResult {
        // For now, just use even postprocessing
        // TODO: Implement proper centrosymmetric postprocessing
        self.even_sve.postprocess(u_list, s_list, v_list)
    }
}