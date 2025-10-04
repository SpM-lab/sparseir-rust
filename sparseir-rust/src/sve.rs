//! Singular Value Expansion (SVE) module
//! 
//! This module provides functionality for computing the singular value expansion
//! of integral kernels, which is the core of the sparseir algorithm.

use ndarray::{Array2, Array1, Array3};
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
/// Returns (U, singular_values, V) where A = U * S * V^T
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
    
    // Compute truncated SVD using xprec-svd with relaxed tolerance
    let rtol = 1e-12; // Relaxed tolerance to avoid over-truncation
    let result = tsvd_f64(&matrix_f64, rtol).expect("SVD computation failed");
    
    // Convert results back to T type
    let u = result.u.map(|&x| <T as CustomNumeric>::from_f64(x));
    let s = result.s.map(|&x| <T as CustomNumeric>::from_f64(x));
    let v = result.v.map(|&x| <T as CustomNumeric>::from_f64(x));  // v is already the right singular vectors
    
    (u, s, v)
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
    let v = result.v.map(|&x| <T as CustomNumeric>::from_f64(x.to_f64()));  // v is already the right singular vectors
    
    (u, s, v)
}

/// Truncate SVD results based on cutoff and maximum size
pub fn truncate<T: CustomNumeric + num_traits::Zero>(
    u_list: Vec<Array2<T>>,
    s_list: Vec<Array1<T>>,
    v_list: Vec<Array2<T>>,
    rtol: T,
    max_num_svals: Option<usize>,
) -> (Vec<Array2<T>>, Vec<Array1<T>>, Vec<Array2<T>>) {
    // Validate parameters (following C++ implementation)
    if let Some(max) = max_num_svals {
        if (max as isize) < 0 {
            panic!("max_num_svals must be non-negative");
        }
    }
    if rtol < T::zero() || rtol > <T as CustomNumeric>::from_f64(1.0) {
        panic!("rtol must be in [0, 1]");
    }
    
    // Collect all singular values from all SVD results (following C++ implementation)
    let mut all_singular_values = Vec::new();
    for s in &s_list {
        for &val in s.iter() {
            all_singular_values.push(val);
        }
    }
    
    // Find global maximum singular value
    let max_singular_value = all_singular_values.iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or(T::zero());
    
    // Calculate global cutoff value
    let cutoff = if let Some(max_count) = max_num_svals {
        if max_count < all_singular_values.len() {
            // Sort in descending order and take the max_count-th largest
            let mut sorted = all_singular_values.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let nth_largest = sorted[max_count - 1];
            // Use the maximum of rtol * max_singular_value and nth_largest
            if rtol * max_singular_value > nth_largest {
                rtol * max_singular_value
            } else {
                nth_largest
            }
        } else {
            rtol * max_singular_value
        }
    } else {
        rtol * max_singular_value
    };
    
    // Count singular values above cutoff for each SVD result (using same global cutoff)
    let mut scount = Vec::new();
    for s in &s_list {
        let mut count = 0;
        for &val in s.iter() {
            if val > cutoff {
                count += 1;
            }
        }
        scount.push(count);
    }
    
    // Truncate each SVD result individually
    let mut truncated_u = Vec::new();
    let mut truncated_s = Vec::new();
    let mut truncated_v = Vec::new();
    
    for (i, ((u, s), v)) in u_list.into_iter().zip(s_list.into_iter()).zip(v_list.into_iter()).enumerate() {
        let count = scount[i];
        
        truncated_u.push(u.slice(ndarray::s![.., ..count]).to_owned());
        truncated_s.push(s.slice(ndarray::s![..count]).to_owned());
        truncated_v.push(v.slice(ndarray::s![.., ..count]).to_owned());
    }
    
    (truncated_u, truncated_s, truncated_v)
}

/// Compute safe epsilon value and determine working precision
fn safe_epsilon(epsilon: f64, twork: TworkType, svd_strategy: SVDStrategy) -> (f64, TworkType, SVDStrategy) {
    // Check for negative epsilon (following C++ implementation)
    if epsilon < 0.0 {
        panic!("eps_required must be non-negative");
    }
    
    // First, choose the working dtype based on the eps required
    let twork_actual = match twork {
        TworkType::Auto => {
            if epsilon.is_nan() || epsilon < 1e-8 {
                TworkType::Float64X2  // MAX_DTYPE equivalent
            } else {
                TworkType::Float64
            }
        }
        other => other,
    };
    
    // Next, work out the actual epsilon
    let safe_eps = match twork_actual {
        TworkType::Float64 => {
            // This is technically a bit too low (the true value is about 1.5e-8),
            // but it's not too far off and easier to remember for the user.
            1e-8
        }
        TworkType::Float64X2 => {
            // For TwoFloat, we don't have access to xprec::DDouble::epsilon()
            // Use a reasonable approximation for double-double precision
            1e-15
        }
        _ => 1e-8,
    };
    
    // Work out the SVD strategy to be used
    let svd_strategy_actual = match svd_strategy {
        SVDStrategy::Auto => {
            if !epsilon.is_nan() && epsilon < safe_eps {
                // TODO: Add warning output like C++
                SVDStrategy::Accurate
            } else {
                SVDStrategy::Fast
            }
        }
        other => other,
    };
    
    (safe_eps, twork_actual, svd_strategy_actual)
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
    max_num_svals: Option<usize>,
) -> (SVEResult, Box<dyn SVEStrategy<T>>)
where
    T: CustomNumeric + Send + Sync + num_traits::Zero + ToPrimitive + 'static,
    K: AbstractKernel + KernelProperties + Clone + 'static,
    <K as KernelProperties>::SVEHintsType<T>: SVEHints<T> + Clone + 'static,
{
    let sve = determine_sve::<T, K>(kernel, safe_epsilon, n_gauss);
    
    // Compute discretized matrices
    let matrices = sve.matrices();
    
    // Compute SVD for each matrix using xprec-svd
    let mut u_list = Vec::new();
    let mut s_list = Vec::new();
    let mut v_list = Vec::new();
    
    for matrix in matrices {
        let (u, s, v) = compute_svd(&matrix);
        u_list.push(u);
        s_list.push(s);
        v_list.push(v);
    }
    
    // Determine rtol (relative tolerance)
    let rtol_actual = cutoff.unwrap_or(2.0 * f64::EPSILON);
    let rtol_t = <T as CustomNumeric>::from_f64(rtol_actual);
    
    // Truncate results
    let (u_truncated, s_truncated, v_truncated) = truncate(
        u_list,
        s_list,
        v_list,
        rtol_t,
        max_num_svals,
    );
    
    // Post-process to create SVEResult
    let result = sve.postprocess(u_truncated, s_truncated, v_truncated);
    
    (result, sve)
}

/// Main SVE computation function
/// 
/// This function orchestrates the SVE computation by choosing the appropriate
/// strategy based on kernel properties and computing parameters.
pub fn compute_sve<K>(
    kernel: K,
    epsilon: f64,
    cutoff: Option<f64>,
    max_num_svals: Option<usize>,
    n_gauss: Option<usize>,
    twork: TworkType,
) -> SVEResult
where
    //T: CustomNumeric + Send + Sync + num_traits::Zero + ToPrimitive + 'static,
    K: AbstractKernel + KernelProperties + Clone + 'static,
    //<K as KernelProperties>::SVEHintsType<T>: SVEHints<T> + Clone + 'static,
{
    // Compute safe epsilon and determine working precision
    let (safe_epsilon, twork_actual, _svd_strategy_actual) = safe_epsilon(epsilon, twork, SVDStrategy::Auto);
    
    // For now, only support f64 precision
    // TODO: Add proper TwoFloat support when num_traits are available
    if twork_actual == TworkType::Float64 || twork_actual == TworkType::Float64X2 {
        pre_postprocess::<f64, K>(kernel, safe_epsilon, n_gauss, cutoff, max_num_svals).0
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
    gauss_x: Rule<T>,
    gauss_y: Rule<T>,
    
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
        
        let gauss_x = rule.piecewise(&segs_x);
        let gauss_y = rule.piecewise(&segs_y);
        
        Self {
            kernel,
            epsilon,
            n_gauss,
            hints,
            rule,
            segs_x,
            segs_y,
            gauss_x,
            gauss_y
        }
    }
    
    /// Convert SVD matrix to polynomial representation
    fn svd_to_polynomials(
        &self,
        u_or_v: &Array2<T>,
        segments: &[T],
        gauss_rule: &crate::gauss::Rule<f64>,
    ) -> Vec<PiecewiseLegendrePoly> {
        let mut polynomials = Vec::new();
        
        // Calculate dimensions for 3D tensor reshaping
        let n_gauss = self.n_gauss;
        let n_segments = segments.len() - 1;
        let n_singular_values = u_or_v.ncols();
        
        // Create 3D tensor: (n_gauss, n_segments, n_singular_values)
        let mut tensor_3d = Array3::<f64>::zeros((n_gauss, n_segments, n_singular_values));
        
        // Copy elements from 2D matrix to 3D tensor
        // Following C++ implementation: u_x(i, j, k) = u_x_(j * n_gauss + i, k)
        // COMMENT: Is there permutedims or transpose functions in ndarray? It would be much easier.
        for i in 0..n_gauss { //COMMENT: rename to idx_gauss_sampling_point
            for j in 0..n_segments { //COMMENT: rename to idx_segment
                for k in 0..n_singular_values { //COMMENT: rename to idx_sval
                    let row_index = j * n_gauss + i;  // C++ indexing: j * n_gauss + i
                    tensor_3d[[i, j, k]] = u_or_v[[row_index, k]].to_f64();
                }
            }
        }
        
        // Create Legendre collocation matrix
        // COMMENT: cmat size is (number of polys, number of gauss sampling points)
        let cmat = self.legendre_collocation(gauss_rule, n_gauss);
        println!("DEBUG: cmat shape = [{}, {}], n_gauss = {}", cmat.nrows(), cmat.ncols(), n_gauss);
        
        // Transform to Legendre basis: u_data(i, j, k) = sum over l: cmat(i, l) * u_x(l, j, k)
        let mut u_data = Array3::<f64>::zeros((cmat.nrows(), n_segments, n_singular_values));
        // COMMENT: is there any contraction code in ndarray? The following loops can be simplified.
        for j in 0..n_segments { //COMMENT: rename variable
            for k in 0..n_singular_values { //COMMENT: rename variable
                for i in 0..cmat.nrows() { //COMMENT: rename variable
                    let mut sum = 0.0;
                    for l in 0..n_gauss {
                        sum += cmat[[i, l]] * tensor_3d[[l, j, k]];
                    }
                    u_data[[i, j, k]] = sum;
                }
            }
        }
        
        // Apply segment length normalization: sqrt(0.5 * delta_segment)
        let mut dsegs = Vec::new();
        for i in 0..segments.len() - 1 {
            dsegs.push(segments[i + 1].to_f64() - segments[i].to_f64());
        }
        
        for j in 0..n_segments {
            let normalization = (0.5 * dsegs[j]).sqrt();
            for i in 0..u_data.shape()[0] {
                for k in 0..n_singular_values {
                    u_data[[i, j, k]] *= normalization;
                }
            }
        }
        
        // Create polynomials from transformed data
        let knots: Vec<f64> = segments.iter().map(|&x| x.to_f64()).collect();
        
        for k in 0..n_singular_values {
            // Extract coefficients for this singular value
            let mut coeffs_2d = Array2::<f64>::zeros((u_data.shape()[0], n_segments));
            for i in 0..u_data.shape()[0] {
                for j in 0..n_segments {
                    coeffs_2d[[i, j]] = u_data[[i, j, k]];
                }
            }
            
            // Calculate delta_x like C++: diff(knots)
            let delta_x: Vec<f64> = knots.windows(2).map(|w| w[1] - w[0]).collect();
            
            let poly = PiecewiseLegendrePoly::new(
                coeffs_2d,
                knots.clone(),
                k as i32,
                Some(delta_x),
                0,  // no symmetry
            );
            
            polynomials.push(poly);
        }
        
        polynomials
    }
    
    /// Create Legendre collocation matrix
    /// Equivalent to C++ legendre_collocation function
    /// COMMENT: check the return size (number of polys, number of points) 
    fn legendre_collocation(&self, rule: &crate::gauss::Rule<f64>, n: usize) -> Array2<f64> {
        // Create Legendre Vandermonde matrix
        let lv = self.legvander(&rule.x.to_vec(), n - 1);
        
        // Apply weights: lv(i, j) *= rule.w[i]
        let mut weighted_lv = lv.clone();
        for i in 0..rule.w.len() {
            for j in 0..weighted_lv.ncols() {
                weighted_lv[[i, j]] *= rule.w[i];
            }
        }
        
        // Transpose the result
        weighted_lv.t().to_owned()
    }
    
    /// Create Legendre Vandermonde matrix
    /// Equivalent to C++ legvander function
    fn legvander(&self, x: &[f64], deg: usize) -> Array2<f64> {
        let n = x.len();
        let mut v = Array2::<f64>::zeros((n, deg + 1));
        
        // First column is all ones
        for i in 0..n {
            v[[i, 0]] = 1.0;
        }
        
        // Second column is x
        if deg > 0 {
            for i in 0..n {
                v[[i, 1]] = x[i];
            }
        }
        
        // Recurrence relation: P_n(x) = ((2*n-1)*x*P_{n-1}(x) - (n-1)*P_{n-2}(x)) / n
        for j in 2..=deg {
            for i in 0..n {
                let n_f64 = j as f64;
                v[[i, j]] = ((2.0 * n_f64 - 1.0) * x[i] * v[[i, j-1]] - (n_f64 - 1.0) * v[[i, j-2]]) / n_f64;
            }
        }
        
        v
    }
}

impl<T: CustomNumeric + Send + Sync + num_traits::Zero + ToPrimitive, K: AbstractKernel + KernelProperties + Clone, H: SVEHints<T> + Clone> SVEStrategy<T> for SamplingSVE<T, K, H> {
    fn matrices(&self) -> Vec<Array2<T>> {
        // Compute kernel matrix using Gauss quadrature
        let discretized = matrix_from_gauss(
            &self.kernel,
            &self.gauss_x,
            &self.gauss_y,
        );
        
        // Apply weights (sqrt of Gauss weights) for SVE
        // This is equivalent to C++ implementation:
        // for (int i = 0; i < gauss_x.w.size(); ++i) {
        //     A.row(i) *= sqrt_impl(gauss_x.w[i]);
        // }
        // for (int j = 0; j < gauss_y.w.size(); ++j) {
        //     A.col(j) *= sqrt_impl(gauss_y.w[j]);
        // }
        let mut weighted_matrix = discretized.matrix.clone();
        
        // Apply row weights (sqrt of x Gauss weights)
        for i in 0..self.gauss_x.w.len() {
            let weight = self.gauss_x.w[i];
            let sqrt_weight = <T as CustomNumeric>::sqrt(weight);
            for j in 0..weighted_matrix.ncols() {
                weighted_matrix[[i, j]] = weighted_matrix[[i, j]] * sqrt_weight;
            }
        }
        
        // Apply column weights (sqrt of y Gauss weights)
        for j in 0..self.gauss_y.w.len() {
            let weight = self.gauss_y.w[j];
            let sqrt_weight = <T as CustomNumeric>::sqrt(weight);
            for i in 0..weighted_matrix.nrows() {
                weighted_matrix[[i, j]] = weighted_matrix[[i, j]] * sqrt_weight;
            }
        }
        
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
        
        // Remove weights from singular vectors (inverse of the weighting applied in matrices())
        // C++ equivalent: u_x_(i, j) = u(i, j) / sqrt(gauss_x_w[i]);
        let mut u_unweighted = u.clone();
        for i in 0..u_unweighted.nrows() {
            let weight = self.gauss_x.w[i];
            let sqrt_weight = <T as CustomNumeric>::sqrt(weight);
            for j in 0..u_unweighted.ncols() {
                u_unweighted[[i, j]] = u_unweighted[[i, j]] / sqrt_weight;
            }
        }
        
        let mut v_unweighted = v.clone();
        for j in 0..v_unweighted.ncols() {
            let weight = self.gauss_y.w[j];
            let sqrt_weight = <T as CustomNumeric>::sqrt(weight);
            for i in 0..v_unweighted.nrows() {
                v_unweighted[[i, j]] = v_unweighted[[i, j]] / sqrt_weight;
            }
        }
        
        // Create polynomial vectors from SVD results
        // Convert T rules to f64 rules for polynomial creation
        let gauss_x_f64 = Rule::<f64>::from_vectors(
            self.gauss_x.x.iter().map(|&x| x.to_f64()).collect(),
            self.gauss_x.w.iter().map(|&w| w.to_f64()).collect(),
            self.gauss_x.a.to_f64(),
            self.gauss_x.b.to_f64(),
        );
        let gauss_y_f64 = Rule::<f64>::from_vectors(
            self.gauss_y.x.iter().map(|&x| x.to_f64()).collect(),
            self.gauss_y.w.iter().map(|&w| w.to_f64()).collect(),
            self.gauss_y.a.to_f64(),
            self.gauss_y.b.to_f64(),
        );

        let u_polys = self.svd_to_polynomials(&u_unweighted, &self.segs_x, &gauss_x_f64);
        let v_polys = self.svd_to_polynomials(&v_unweighted, &self.segs_y, &gauss_y_f64);
        
        let u_vec = PiecewiseLegendrePolyVector::new(u_polys);
        let v_vec = PiecewiseLegendrePolyVector::new(v_polys);
        
        SVEResult::new(u_vec, s_f64, v_vec, self.epsilon)
    }
}


/// Centrosymmetric SVE computation
/// 
/// Optimized SVE computation for centrosymmetric kernels that can be
/// block-diagonalized to reduce computational cost.
/// TODO: restrict to a centrosymmetric kernel
pub struct CentrosymmSVE<T: CustomNumeric + Send + Sync + 'static, K: AbstractKernel + KernelProperties + Clone> {
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
    even_sve: SamplingSVE<T, crate::kernel::SymmetrizedKernel<K>, crate::kernel::ReducedSVEHintsWrapper<T>>,
    /// Odd symmetry SVE
    #[allow(dead_code)]
    odd_sve: SamplingSVE<T, crate::kernel::SymmetrizedKernel<K>, crate::kernel::ReducedSVEHintsWrapper<T>>,
}

impl<T: CustomNumeric + Send + Sync + num_traits::Zero + ToPrimitive + 'static, K: AbstractKernel + KernelProperties + Clone> CentrosymmSVE<T, K> 
where 
    K::SVEHintsType<T>: SVEHints<T> + Clone,
    crate::kernel::ReducedSVEHintsWrapper<T>: SVEHints<T> + Clone,
{
    /// Create a new CentrosymmSVE
    pub fn new(kernel: K, _hints: K::SVEHintsType<T>, epsilon: f64, n_gauss: Option<usize>) -> Self {
        use crate::kernel::SymmetrizedKernel;

        if (!kernel.is_centrosymmetric()) {
            panic!("Supported only centrosymmetric kernel!");
        }
        
        // Create even and odd symmetrized kernels
        let even_kernel = SymmetrizedKernel::new(kernel.clone(), 1);  // +1 for even
        let odd_kernel = SymmetrizedKernel::new(kernel.clone(), -1);  // -1 for odd
        
        // Create SVE hints for symmetrized kernels
        let even_hints = even_kernel.sve_hints(epsilon);
        let odd_hints = odd_kernel.sve_hints(epsilon);
        
        // Create SamplingSVE instances for even and odd components
        let even_sve = SamplingSVE::new(even_kernel, even_hints, epsilon, n_gauss);
        let odd_sve = SamplingSVE::new(odd_kernel, odd_hints, epsilon, n_gauss);
        
        Self {
            kernel,
            epsilon,
            n_gauss: even_sve.n_gauss,
            even_sve,
            odd_sve,
        }
    }
}

impl<T: CustomNumeric + Send + Sync + num_traits::Zero + ToPrimitive + 'static, K: AbstractKernel + KernelProperties + Clone> SVEStrategy<T> for CentrosymmSVE<T, K> 
where 
    K::SVEHintsType<T>: SVEHints<T> + Clone,
    crate::kernel::ReducedSVEHintsWrapper<T>: SVEHints<T> + Clone,
{
    fn matrices(&self) -> Vec<Array2<T>> {
        // Return both even and odd matrices
        let even_matrices = self.even_sve.matrices();
        let odd_matrices = self.odd_sve.matrices();
        
        vec![even_matrices[0].clone(), odd_matrices[0].clone()]
    }
    
    fn postprocess(
        &self,
        u_list: Vec<Array2<T>>,
        s_list: Vec<Array1<T>>,
        v_list: Vec<Array2<T>>,
    ) -> SVEResult {
        // Separate even and odd results
        let result_even = self.even_sve.postprocess(
            vec![u_list[0].clone()], 
            vec![s_list[0].clone()], 
            vec![v_list[0].clone()]
        );
        let result_odd = self.odd_sve.postprocess(
            vec![u_list[1].clone()], 
            vec![s_list[1].clone()], 
            vec![v_list[1].clone()]
        );
        
        // Merge singular values and create signs
        let mut s_merged = result_even.s.to_vec();
        s_merged.extend(result_odd.s.to_vec());
        
        // Create signs: +1 for even, -1 for odd
        let mut signs = vec![1.0; result_even.s.len()];
        signs.extend(vec![-1.0; result_odd.s.len()]);
        
        // Sort by singular values (descending order)
        let mut indices: Vec<usize> = (0..s_merged.len()).collect();
        indices.sort_by(|&a, &b| s_merged[b].partial_cmp(&s_merged[a]).unwrap());
        
        // Get full segments for complete domain from the original LogisticKernel
        // This matches C++ implementation: sve_hints<T>(kernel, epsilon)
        use crate::kernel::LogisticKernel;
        let original_kernel = LogisticKernel::new(self.kernel.lambda());
        let full_hints = original_kernel.sve_hints::<f64>(self.epsilon);
        let segs_x_full = full_hints.segments_x();
        let segs_y_full = full_hints.segments_y();
        
        
        // Create poly_flip_x: alternating signs for Legendre polynomials
        let n_poly_coeffs = result_even.u.polyvec[0].data.shape()[0];
        let mut poly_flip_x = Vec::new();
        for i in 0..n_poly_coeffs {
            poly_flip_x.push(if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        
        // Create complete singular functions using helper function
        let mut u_complete = Vec::new();
        let mut v_complete = Vec::new();
        let mut s_sorted = Vec::new();
        
        for &idx in &indices {
            let sign = signs[idx];
            println!("s_merged {} {}", idx, s_merged[idx]);
            
            // Get the original polynomial (from reduced domain)
            let u_poly = if idx < result_even.u.polyvec.len() {
                &result_even.u.polyvec[idx]
            } else {
                &result_odd.u.polyvec[idx - result_even.u.polyvec.len()]
            };
            
            let v_poly = if idx < result_even.v.polyvec.len() {
                &result_even.v.polyvec[idx]
            } else {
                &result_odd.v.polyvec[idx - result_even.v.polyvec.len()]
            };
            
            // Extend polynomial to full domain
            let u_complete_poly = extend_polynomial_to_full_domain(
                u_poly, &segs_x_full, sign, &poly_flip_x
            );
            let v_complete_poly = extend_polynomial_to_full_domain(
                v_poly, &segs_y_full, sign, &poly_flip_x
            );
            
            u_complete.push(u_complete_poly);
            v_complete.push(v_complete_poly);
            s_sorted.push(s_merged[idx]);
        }
        
        let u_final = crate::poly::PiecewiseLegendrePolyVector::new(u_complete);
        let v_final = crate::poly::PiecewiseLegendrePolyVector::new(v_complete);
        let s_final = ndarray::Array1::from_vec(s_sorted);
        
        SVEResult::new(u_final, s_final, v_final, self.epsilon)
    }
}

/// Extend a polynomial from reduced domain [0, 1] to full domain [-1, 1]
/// following the C++ implementation logic
fn extend_polynomial_to_full_domain(
    poly: &crate::poly::PiecewiseLegendrePoly,
    full_segments: &[f64],
    sign: f64,
    poly_flip_x: &[f64],
) -> crate::poly::PiecewiseLegendrePoly {
    // Normalize by 1/sqrt(2) and convert to f64
    let pos_data = poly.data.mapv(|x| x.to_f64() / 2.0_f64.sqrt());
    
    // Create negative part by reversing and applying signs
    let mut neg_data = pos_data.clone();
    neg_data = neg_data.slice(ndarray::s![..;-1, ..]).to_owned();
    
    // Apply poly_flip_x and sign to negative part
    for (i, &flip_sign) in poly_flip_x.iter().enumerate() {
        let coeff_sign = flip_sign * sign;
        neg_data.row_mut(i).mapv_inplace(|x| x * coeff_sign);
    }
    
    // Combine positive and negative parts
    let combined_data = ndarray::concatenate![ndarray::Axis(1), neg_data, pos_data];
    
    // Create complete polynomial with full segments
    crate::poly::PiecewiseLegendrePoly::new(
        combined_data,
        full_segments.to_vec(),
        poly.polyorder as i32,
        None, // delta_x
        0     // symm (no symmetry)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::PiecewiseLegendrePoly;
    use ndarray::Array2;

    #[test]
    fn test_safe_epsilon_boundary_switch() {
        // Test boundary switching at 1e-8
        // Case 1: epsilon = 1e-8 (should use FLOAT64)
        let (safe_eps1, twork1, svd_strat1) = safe_epsilon(1e-8, TworkType::Auto, SVDStrategy::Auto);
        assert_eq!(twork1, TworkType::Float64);
        assert_eq!(safe_eps1, 1e-8);
        assert_eq!(svd_strat1, SVDStrategy::Fast); // 1e-8 >= 1e-8, so Fast

        // Case 2: epsilon = 0.9e-8 (should use FLOAT64X2)
        let (safe_eps2, twork2, svd_strat2) = safe_epsilon(0.9e-8, TworkType::Auto, SVDStrategy::Auto);
        assert_eq!(twork2, TworkType::Float64X2);
        assert_eq!(safe_eps2, 1e-15); // Float64X2 safe epsilon
                  assert_eq!(svd_strat2, SVDStrategy::Fast); // 0.9e-8 < 1e-15 is false, so Fast

        println!("1e-8: twork={:?}, safe_eps={}, svd_strat={:?}", twork1, safe_eps1, svd_strat1);
        println!("0.9e-8: twork={:?}, safe_eps={}, svd_strat={:?}", twork2, safe_eps2, svd_strat2);
    }

    #[test]
    fn test_safe_epsilon_nan_handling() {
        // Test NaN handling
        let (safe_eps, twork, svd_strat) = safe_epsilon(f64::NAN, TworkType::Auto, SVDStrategy::Auto);
        assert_eq!(twork, TworkType::Float64X2); // NaN should trigger FLOAT64X2
        assert_eq!(safe_eps, 1e-15);
        assert_eq!(svd_strat, SVDStrategy::Fast); // NaN comparison should be false
    }

    #[test]
    fn test_safe_epsilon_negative_panic() {
        // Test negative epsilon panic
        let result = std::panic::catch_unwind(|| {
            safe_epsilon(-1.0, TworkType::Auto, SVDStrategy::Auto);
        });
        assert!(result.is_err(), "Should panic for negative epsilon");
    }

    #[test]
    fn test_safe_epsilon_explicit_types() {
        // Test explicit TworkType usage
        let (safe_eps1, twork1, _) = safe_epsilon(1e-6, TworkType::Float64, SVDStrategy::Auto);
        assert_eq!(twork1, TworkType::Float64);
        assert_eq!(safe_eps1, 1e-8);

        let (safe_eps2, twork2, _) = safe_epsilon(1e-6, TworkType::Float64X2, SVDStrategy::Auto);
        assert_eq!(twork2, TworkType::Float64X2);
        assert_eq!(safe_eps2, 1e-15);
    }

    #[test]
    fn test_extend_polynomial_to_full_domain() {
        // Create a simple polynomial in reduced domain [0, 1]
        let reduced_segments = vec![0.0, 1.0];  // 1 segment, 2 knots
        let full_segments = vec![-1.0, 0.0, 1.0];  // 2 segments, 3 knots
        
        // Create test polynomial data: 2x1 matrix (2 polynomial orders, 1 segment)
        let data = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let poly = PiecewiseLegendrePoly::new(data, reduced_segments, 1, None, 0);
        
        // Test even symmetry (sign = +1)
        let poly_flip_x = vec![1.0, -1.0];  // alternating signs for Legendre polynomials
        let extended_poly = extend_polynomial_to_full_domain(
            &poly, &full_segments, 1.0, &poly_flip_x
        );
        
        // Verify the extended polynomial has correct dimensions
        // Should have 2 polynomial orders and 2 segments (full domain)
        assert_eq!(extended_poly.data.shape(), [2, 2]);
        
        // Verify the segments are correct
        assert_eq!(extended_poly.knots, full_segments);
        
        // Test odd symmetry (sign = -1)
        let extended_poly_odd = extend_polynomial_to_full_domain(
            &poly, &full_segments, -1.0, &poly_flip_x
        );
        
        // Verify the extended polynomial has correct dimensions
        assert_eq!(extended_poly_odd.data.shape(), [2, 2]);
        assert_eq!(extended_poly_odd.knots, full_segments);
        
        // Test that even and odd extensions produce different results
        // (due to different signs applied to the negative part)
        assert_ne!(extended_poly.data, extended_poly_odd.data);
    }

    #[test]
    fn test_extend_polynomial_basic_functionality() {
        // Test basic functionality of the extension
        let reduced_segments = vec![0.0, 1.0];  // 1 segment
        let full_segments = vec![-1.0, 0.0, 1.0];  // 2 segments
        
        // Create simple polynomial data: constant function
        let data = Array2::from_shape_vec((2, 1), vec![1.0, 0.0]).unwrap();  // constant + linear term
        let poly = PiecewiseLegendrePoly::new(data, reduced_segments, 1, None, 0);
        
        let poly_flip_x = vec![1.0, -1.0];
        
        // Test even extension
        let extended_even = extend_polynomial_to_full_domain(
            &poly, &full_segments, 1.0, &poly_flip_x
        );
        
        // Test odd extension
        let extended_odd = extend_polynomial_to_full_domain(
            &poly, &full_segments, -1.0, &poly_flip_x
        );
        
        // Verify both extensions have correct dimensions
        assert_eq!(extended_even.data.shape(), [2, 2]);
        assert_eq!(extended_odd.data.shape(), [2, 2]);
        
        // Verify both use the correct full segments
        assert_eq!(extended_even.knots, full_segments);
        assert_eq!(extended_odd.knots, full_segments);
        
        // Test that extensions produce different results
        assert_ne!(extended_even.data, extended_odd.data);
        
        // Test evaluation at origin (should work for both)
        let even_at_0 = extended_even.evaluate(0.0);
        let odd_at_0 = extended_odd.evaluate(0.0);
        
        // Both should be finite values
        assert!(even_at_0.is_finite());
        assert!(odd_at_0.is_finite());
    }
}