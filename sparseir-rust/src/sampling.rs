//! Sparse sampling in imaginary time
//!
//! This module provides `TauSampling` for transforming between IR basis coefficients
//! and values at sparse sampling points in imaginary time.

use crate::basis::FiniteTempBasis;
use crate::gemm::matmul_par;
use crate::kernel::{KernelProperties, CentrosymmKernel};
use crate::traits::StatisticsType;
use mdarray::DTensor;

/// Sparse sampling in imaginary time
///
/// Allows transformation between the IR basis and a set of sampling points
/// in imaginary time (τ).
pub struct TauSampling<S>
where
    S: StatisticsType,
{
    /// Sampling points in imaginary time τ ∈ [0, β]
    sampling_points: Vec<f64>,
    
    /// Sampling matrix: A[i, l] = u_l(τ_i)
    /// Shape: (n_sampling_points, basis_size)
    matrix: DTensor<f64, 2>,
    
    /// SVD of the sampling matrix (for fitting)
    matrix_svd: Option<SamplingMatrixSVD>,
    
    /// Marker for statistics type
    _phantom: std::marker::PhantomData<S>,
}

/// SVD decomposition of the sampling matrix
///
/// Stores U, S, V from A = U * S * Vᵀ for efficient pseudoinverse computation
struct SamplingMatrixSVD {
    u: DTensor<f64, 2>,
    s: Vec<f64>,
    vt: DTensor<f64, 2>,
}

impl<S> TauSampling<S>
where
    S: StatisticsType,
{
    /// Create a new TauSampling with default sampling points
    ///
    /// The default sampling points are chosen as the extrema of the highest-order
    /// basis function, which gives near-optimal conditioning.
    ///
    /// # Arguments
    /// * `basis` - The finite temperature basis
    ///
    /// # Returns
    /// A new TauSampling object with SVD computed
    pub fn new<K>(basis: &FiniteTempBasis<K, S>) -> Self
    where
        K: KernelProperties + CentrosymmKernel + Clone + 'static,
    {
        let sampling_points = basis.default_tau_sampling_points();
        Self::with_sampling_points(basis, sampling_points, true)
    }
    
    /// Create a new TauSampling with custom sampling points
    ///
    /// # Arguments
    /// * `basis` - The finite temperature basis
    /// * `sampling_points` - Custom sampling points in τ ∈ [0, β]
    /// * `compute_svd` - Whether to compute the SVD for fitting
    ///
    /// # Returns
    /// A new TauSampling object
    ///
    /// # Panics
    /// Panics if `sampling_points` is empty or if any point is outside [0, β]
    pub fn with_sampling_points<K>(
        basis: &FiniteTempBasis<K, S>,
        sampling_points: Vec<f64>,
        compute_svd: bool,
    ) -> Self
    where
        K: KernelProperties + CentrosymmKernel + Clone + 'static,
    {
        assert!(!sampling_points.is_empty(), "No sampling points given");
        
        let beta = basis.beta;
        for &tau in &sampling_points {
            assert!(
                tau >= 0.0 && tau <= beta,
                "Sampling point τ={} is outside [0, β={}]",
                tau,
                beta
            );
        }
        
        // Compute sampling matrix: A[i, l] = u_l(τ_i)
        let matrix = eval_matrix_tau(basis, &sampling_points);
        
        // Compute SVD if requested
        let matrix_svd = if compute_svd {
            Some(compute_matrix_svd(&matrix))
        } else {
            None
        };
        
        // Check conditioning
        if let Some(ref svd) = matrix_svd {
            let condition_number = svd.s[0] / svd.s[svd.s.len() - 1];
            if condition_number > 1e8 {
                eprintln!(
                    "Warning: Sampling matrix is poorly conditioned (cond = {:.2e})",
                    condition_number
                );
            }
        }
        
        Self {
            sampling_points,
            matrix,
            matrix_svd,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Get the sampling points
    pub fn sampling_points(&self) -> &[f64] {
        &self.sampling_points
    }
    
    /// Get the number of sampling points
    pub fn n_sampling_points(&self) -> usize {
        self.sampling_points.len()
    }
    
    /// Get the basis size
    pub fn basis_size(&self) -> usize {
        self.matrix.shape().1
    }
    
    /// Get the sampling matrix
    pub fn matrix(&self) -> &DTensor<f64, 2> {
        &self.matrix
    }
    
    /// Evaluate basis coefficients at sampling points
    ///
    /// Computes g(τ_i) = Σ_l a_l * u_l(τ_i) for all sampling points
    ///
    /// # Arguments
    /// * `coeffs` - Basis coefficients (length = basis_size)
    ///
    /// # Returns
    /// Values at sampling points (length = n_sampling_points)
    ///
    /// # Panics
    /// Panics if `coeffs.len() != basis_size`
    pub fn evaluate(&self, coeffs: &[f64]) -> Vec<f64> {
        let basis_size = self.basis_size();
        assert_eq!(
            coeffs.len(),
            basis_size,
            "Number of coefficients ({}) must match basis size ({})",
            coeffs.len(),
            basis_size
        );
        
        // Convert coeffs to column vector (basis_size, 1)
        let coeffs_col = DTensor::<f64, 2>::from_fn([basis_size, 1], |idx| coeffs[idx[0]]);
        
        // result = matrix * coeffs
        let result = matmul_par(&self.matrix, &coeffs_col);
        
        // Extract result as Vec
        let n = self.n_sampling_points();
        (0..n).map(|i| result[[i, 0]]).collect()
    }
    
    /// Fit basis coefficients from values at sampling points
    ///
    /// Solves the least-squares problem: min ||A * coeffs - values||
    /// using the SVD pseudoinverse: coeffs = Vᵀ * S⁻¹ * Uᵀ * values
    ///
    /// # Arguments
    /// * `values` - Values at sampling points (length = n_sampling_points)
    ///
    /// # Returns
    /// Fitted basis coefficients (length = basis_size)
    ///
    /// # Panics
    /// Panics if `values.len() != n_sampling_points` or if SVD was not computed
    pub fn fit(&self, values: &[f64]) -> Vec<f64> {
        let n_points = self.n_sampling_points();
        assert_eq!(
            values.len(),
            n_points,
            "Number of values ({}) must match number of sampling points ({})",
            values.len(),
            n_points
        );
        
        let svd = self.matrix_svd.as_ref()
            .expect("SVD not computed. Create TauSampling with compute_svd=true");
        
        // Convert values to column vector
        let values_col = DTensor::<f64, 2>::from_fn([n_points, 1], |idx| values[idx[0]]);
        
        // Compute U^T * values
        let ut = svd.u.transpose().to_tensor();
        let ut_values = matmul_par(&ut, &values_col);
        
        // Divide by singular values: S^{-1} * (U^T * values)
        let basis_size = self.basis_size();
        let s_inv_ut_values = DTensor::<f64, 2>::from_fn([basis_size, 1], |idx| {
            let i = idx[0];
            if i < svd.s.len() {
                ut_values[[i, 0]] / svd.s[i]
            } else {
                0.0
            }
        });
        
        // coeffs = V^T^T * (S^{-1} * U^T * values) = V * (S^{-1} * U^T * values)
        let v = svd.vt.transpose().to_tensor();
        let coeffs_col = matmul_par(&v, &s_inv_ut_values);
        
        // Extract result as Vec
        (0..basis_size).map(|i| coeffs_col[[i, 0]]).collect()
    }
}

/// Evaluate the sampling matrix: A[i, l] = u_l(τ_i)
///
/// # Arguments
/// * `basis` - The finite temperature basis
/// * `sampling_points` - Sampling points in τ ∈ [0, β]
///
/// # Returns
/// Sampling matrix of shape (n_sampling_points, basis_size)
fn eval_matrix_tau<K, S>(basis: &FiniteTempBasis<K, S>, sampling_points: &[f64]) -> DTensor<f64, 2>
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType,
{
    let n_points = sampling_points.len();
    let basis_size = basis.size();
    
    // A[i, l] = u_l(τ_i)
    DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
        let (i, l) = (idx[0], idx[1]);
        let tau = sampling_points[i];
        basis.u[l].evaluate(tau)
    })
}

/// Compute SVD of the sampling matrix using Faer
fn compute_matrix_svd(matrix: &DTensor<f64, 2>) -> SamplingMatrixSVD {
    use mdarray_linalg::{SVD, SVDDecomp};
    use mdarray_linalg_faer::Faer;
    
    // Clone matrix for SVD (Faer's SVD may modify input)
    let mut a = matrix.clone();
    
    // Compute thin SVD
    let SVDDecomp { u, s, vt } = Faer.svd(&mut a)
        .expect("SVD computation failed");
    
    // Convert s to Vec<f64>
    let s_vec: Vec<f64> = (0..s.len()).map(|i| s[i]).collect();
    
    SamplingMatrixSVD {
        u,
        s: s_vec,
        vt,
    }
}

