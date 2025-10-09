//! Sparse sampling in imaginary time
//!
//! This module provides `TauSampling` for transforming between IR basis coefficients
//! and values at sparse sampling points in imaginary time.

use crate::basis::FiniteTempBasis;
use crate::gemm::matmul_par;
use crate::kernel::{KernelProperties, CentrosymmKernel};
use crate::traits::StatisticsType;
use mdarray::{DTensor, Tensor, DynRank, Shape};
use num_complex::Complex;
use std::cell::RefCell;

/// Move axis from position `src` to position `dst`
///
/// This is equivalent to numpy.moveaxis or libsparseir's movedim.
/// It creates a permutation array that moves the specified axis.
///
/// # Arguments
/// * `arr` - Input tensor
/// * `src` - Source axis position
/// * `dst` - Destination axis position
///
/// # Returns
/// Tensor with axes permuted
///
/// # Example
/// ```ignore
/// // For a 4D tensor with shape (2, 3, 4, 5)
/// // movedim(arr, 0, 2) moves axis 0 to position 2
/// // Result shape: (3, 4, 2, 5) with axes permuted as [1, 2, 0, 3]
/// ```
fn movedim<T: Clone>(arr: &Tensor<T, DynRank>, src: usize, dst: usize) -> Tensor<T, DynRank> {
    if src == dst {
        return arr.clone();
    }
    
    let rank = arr.rank();
    assert!(src < rank, "src axis {} out of bounds for rank {}", src, rank);
    assert!(dst < rank, "dst axis {} out of bounds for rank {}", dst, rank);
    
    // Generate permutation: move src to dst position
    let mut perm = Vec::with_capacity(rank);
    let mut pos = 0;
    for i in 0..rank {
        if i == dst {
            perm.push(src);
        } else {
            // Skip src position
            if pos == src {
                pos += 1;
            }
            perm.push(pos);
            pos += 1;
        }
    }
    
    arr.permute(&perm[..]).to_tensor()
}

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
    
    /// SVD of the sampling matrix (lazily computed on first fit)
    matrix_svd: RefCell<Option<SamplingMatrixSVD>>,
    
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
    /// SVD is computed lazily on first call to `fit` or `fit_nd`.
    ///
    /// # Arguments
    /// * `basis` - The finite temperature basis
    ///
    /// # Returns
    /// A new TauSampling object
    pub fn new<K>(basis: &FiniteTempBasis<K, S>) -> Self
    where
        K: KernelProperties + CentrosymmKernel + Clone + 'static,
    {
        let sampling_points = basis.default_tau_sampling_points();
        Self::with_sampling_points(basis, sampling_points)
    }
    
    /// Create a new TauSampling with custom sampling points
    ///
    /// SVD is computed lazily on first call to `fit` or `fit_nd`.
    ///
    /// # Arguments
    /// * `basis` - The finite temperature basis
    /// * `sampling_points` - Custom sampling points in τ ∈ [0, β]
    ///
    /// # Returns
    /// A new TauSampling object
    ///
    /// # Panics
    /// Panics if `sampling_points` is empty or if any point is outside [0, β]
    pub fn with_sampling_points<K>(
        basis: &FiniteTempBasis<K, S>,
        sampling_points: Vec<f64>,
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
        
        Self {
            sampling_points,
            matrix,
            matrix_svd: RefCell::new(None),
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
    
    /// Internal generic fit implementation
    ///
    /// Solves: min ||A * coeffs - values|| using SVD pseudoinverse
    /// where A is real but values and coeffs can be complex
    fn fit_impl<T>(&self, values: &[T]) -> Vec<T>
    where
        T: num_complex::ComplexFloat + faer_traits::ComplexField + num_traits::One + 'static + From<f64>,
    {
        let n_points = self.n_sampling_points();
        assert_eq!(
            values.len(),
            n_points,
            "Number of values ({}) must match number of sampling points ({})",
            values.len(),
            n_points
        );
        
        // Compute SVD lazily on first call (always real, since matrix is real)
        if self.matrix_svd.borrow().is_none() {
            let svd = compute_matrix_svd(&self.matrix);
            
            // Check conditioning
            let condition_number = svd.s[0] / svd.s[svd.s.len() - 1];
            if condition_number > 1e8 {
                eprintln!(
                    "Warning: Sampling matrix is poorly conditioned (cond = {:.2e})",
                    condition_number
                );
            }
            
            *self.matrix_svd.borrow_mut() = Some(svd);
        }
        
        let svd = self.matrix_svd.borrow();
        let svd = svd.as_ref().unwrap();
        
        // Convert real SVD matrices to type T
        // U, V are real orthogonal matrices, convert to complex if needed
        let basis_size = self.basis_size();
        
        // Convert values to column vector (type T)
        let values_col = DTensor::<T, 2>::from_fn([n_points, 1], |idx| values[idx[0]]);
        
        // U^T (real) as type T
        let ut = DTensor::<T, 2>::from_fn(*svd.u.shape(), |idx| {
            svd.u[[idx[1], idx[0]]].into()  // transpose and convert
        });
        
        // Compute U^T * values (real matrix × T vector = T vector)
        let ut_values = matmul_par(&ut, &values_col);
        
        // Divide by singular values: S^{-1} * (U^T * values)
        let s_inv_ut_values = DTensor::<T, 2>::from_fn([basis_size, 1], |idx| {
            let i = idx[0];
            if i < svd.s.len() {
                ut_values[[i, 0]] / svd.s[i].into()
            } else {
                T::zero()
            }
        });
        
        // V (real) as type T
        let v = DTensor::<T, 2>::from_fn(*svd.vt.shape(), |idx| {
            svd.vt[[idx[1], idx[0]]].into()  // transpose and convert
        });
        
        // coeffs = V * (S^{-1} * U^T * values)
        let coeffs_col = matmul_par(&v, &s_inv_ut_values);
        
        // Extract result as Vec
        (0..basis_size).map(|i| coeffs_col[[i, 0]]).collect()
    }
    
    /// Fit basis coefficients from real values at sampling points
    ///
    /// Solves the least-squares problem: min ||A * coeffs - values||
    /// using the SVD pseudoinverse: coeffs = Vᵀ * S⁻¹ * Uᵀ * values
    ///
    /// SVD is computed lazily on first call to `fit` or `fit_nd`.
    ///
    /// # Arguments
    /// * `values` - Real values at sampling points (length = n_sampling_points)
    ///
    /// # Returns
    /// Fitted basis coefficients (length = basis_size)
    ///
    /// # Panics
    /// Panics if `values.len() != n_sampling_points`
    pub fn fit(&self, values: &[f64]) -> Vec<f64> {
        self.fit_impl(values)
    }
    
    /// Fit basis coefficients from complex values at sampling points
    ///
    /// Solves the least-squares problem: min ||A * coeffs - values||
    /// using the SVD pseudoinverse, where A is the real sampling matrix.
    ///
    /// # Arguments
    /// * `values` - Complex values at sampling points (length = n_sampling_points)
    ///
    /// # Returns
    /// Fitted complex basis coefficients (length = basis_size)
    ///
    /// # Panics
    /// Panics if `values.len() != n_sampling_points`
    pub fn fit_complex(&self, values: &[Complex<f64>]) -> Vec<Complex<f64>> {
        self.fit_impl(values)
    }
    
    /// Evaluate basis coefficients at sampling points (N-dimensional)
    ///
    /// Evaluates along the specified dimension, keeping other dimensions intact.
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional array with `coeffs.shape().dim(dim) == basis_size`
    /// * `dim` - Dimension along which to evaluate (0-indexed)
    ///
    /// # Returns
    /// N-dimensional array with `result.shape().dim(dim) == n_sampling_points`
    ///
    /// # Panics
    /// Panics if `coeffs.shape().dim(dim) != basis_size` or if `dim >= rank`
    ///
    /// # Example
    /// ```ignore
    /// use mdarray::tensor;
    /// // coeffs: (basis_size, n_k, n_omega)
    /// // With dim=0, result: (n_sampling_points, n_k, n_omega)
    /// let values = sampling.evaluate_nd(&coeffs, 0);
    /// ```
    pub fn evaluate_nd(
        &self,
        coeffs: &Tensor<f64, DynRank>,
        dim: usize,
    ) -> Tensor<f64, DynRank> {
        let rank = coeffs.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        
        let basis_size = self.basis_size();
        let target_dim_size = coeffs.shape().dim(dim);
        
        // Check that the target dimension matches basis_size
        assert_eq!(
            target_dim_size,
            basis_size,
            "coeffs.shape().dim({}) = {} must equal basis_size = {}",
            dim,
            target_dim_size,
            basis_size
        );
        
        // 1. Move target dimension to position 0
        let coeffs_dim0 = movedim(coeffs, dim, 0);
        
        // 2. Reshape to 2D: (basis_size, extra_size)
        let extra_size: usize = coeffs_dim0.len() / basis_size;
        
        // Convert DynRank to fixed Rank<2> for matmul_par
        let coeffs_2d_dyn = coeffs_dim0.reshape(&[basis_size, extra_size][..]).to_tensor();
        let coeffs_2d = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| {
            coeffs_2d_dyn[&[idx[0], idx[1]][..]]
        });
        
        // 3. Matrix multiply: result = A * coeffs
        //    A: (n_points, basis_size), coeffs: (basis_size, extra_size)
        //    result: (n_points, extra_size)
        let n_points = self.n_sampling_points();
        let result_2d = matmul_par(&self.matrix, &coeffs_2d);
        
        // 4. Reshape back to N-D with n_points at position 0
        let mut result_shape = vec![n_points];
        coeffs_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                result_shape.push(dims[i]);
            }
        });
        
        // Convert DTensor<f64, 2> to DynRank using into_dyn()
        let result_2d_dyn = result_2d.into_dyn();
        let result_dim0 = result_2d_dyn.reshape(&result_shape[..]).to_tensor();
        
        // 5. Move dimension back to original position
        movedim(&result_dim0, 0, dim)
    }
    
    /// Fit basis coefficients from values at sampling points (N-dimensional)
    ///
    /// Fits along the specified dimension, keeping other dimensions intact.
    ///
    /// # Arguments
    /// * `values` - N-dimensional array with `values.shape().dim(dim) == n_sampling_points`
    /// * `dim` - Dimension along which to fit (0-indexed)
    ///
    /// # Returns
    /// N-dimensional array with `result.shape().dim(dim) == basis_size`
    ///
    /// # Panics
    /// Panics if `values.shape().dim(dim) != n_sampling_points`, if `dim >= rank`, or if SVD not computed
    ///
    /// # Example
    /// ```ignore
    /// use mdarray::tensor;
    /// // values: (n_sampling_points, n_k, n_omega)
    /// // With dim=0, result: (basis_size, n_k, n_omega)
    /// let coeffs = sampling.fit_nd(&values, 0);
    /// ```
    pub fn fit_nd(
        &self,
        values: &Tensor<f64, DynRank>,
        dim: usize,
    ) -> Tensor<f64, DynRank> {
        let rank = values.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        
        // Compute SVD lazily on first call
        if self.matrix_svd.borrow().is_none() {
            let svd = compute_matrix_svd(&self.matrix);
            
            // Check conditioning
            let condition_number = svd.s[0] / svd.s[svd.s.len() - 1];
            if condition_number > 1e8 {
                eprintln!(
                    "Warning: Sampling matrix is poorly conditioned (cond = {:.2e})",
                    condition_number
                );
            }
            
            *self.matrix_svd.borrow_mut() = Some(svd);
        }
        
        let svd = self.matrix_svd.borrow();
        let svd = svd.as_ref().unwrap();
        
        let n_points = self.n_sampling_points();
        let target_dim_size = values.shape().dim(dim);
        
        // Check that the target dimension matches n_sampling_points
        assert_eq!(
            target_dim_size,
            n_points,
            "values.shape().dim({}) = {} must equal n_sampling_points = {}",
            dim,
            target_dim_size,
            n_points
        );
        
        // Apply SVD-based fit along dimension
        fit_along_dim(svd, values, dim)
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

/// Fit operation along a specific dimension using SVD
fn fit_along_dim(
    svd: &SamplingMatrixSVD,
    values: &Tensor<f64, DynRank>,
    dim: usize,
) -> Tensor<f64, DynRank> {
    if dim == 0 {
        // Get dimensions
        let rank = values.rank();
        let dim0 = values.shape().dim(0);
        let rest: usize = (1..rank).map(|i| values.shape().dim(i)).product();
        
        // Store values_shape for later use
        let values_shape: Vec<usize> = values.shape().with_dims(|dims| dims.to_vec());
        
        // Reshape values: (d0, d1, d2, ...) → (d0, d1*d2*...)
        let values_2d = DTensor::<f64, 2>::from_fn([dim0, rest], |idx| {
            let i = idx[0];
            let flat_j = idx[1];
            
            // Convert flat_j back to multi-dimensional index
            let mut multi_idx = vec![0; rank];
            multi_idx[0] = i;
            let mut remainder = flat_j;
            for d in (1..rank).rev() {
                multi_idx[d] = remainder % values_shape[d];
                remainder /= values_shape[d];
            }
            values[&multi_idx[..]]
        });
        
        // Compute U^T * values_2d
        let ut = svd.u.transpose().to_tensor();
        let ut_values = matmul_par(&ut, &values_2d);
        
        // Divide by singular values
        let basis_size = svd.vt.shape().1;
        let s_inv_ut_values = DTensor::<f64, 2>::from_fn([basis_size, rest], |idx| {
            let i = idx[0];
            let j = idx[1];
            if i < svd.s.len() {
                ut_values[[i, j]] / svd.s[i]
            } else {
                0.0
            }
        });
        
        // coeffs_2d = V * (S^{-1} * U^T * values)
        let v = svd.vt.transpose().to_tensor();
        let coeffs_2d = matmul_par(&v, &s_inv_ut_values);
        
        // Reshape back: (basis_size, rest) → (basis_size, d1, d2, ...)
        let mut result_shape = values_shape.clone();
        result_shape[0] = basis_size;
        
        // Create result tensor and fill element by element
        let mut result: Tensor<f64, DynRank> = Tensor::zeros(result_shape.as_slice());
        
        // Generate all multi-dimensional indices and fill
        let mut indices = vec![0; rank];
        loop {
            let i = indices[0];
            let mut flat_j = 0;
            let mut stride = 1;
            for d in (1..rank).rev() {
                flat_j += indices[d] * stride;
                stride *= values_shape[d];
            }
            result[indices.as_slice()] = coeffs_2d[[i, flat_j]];
            
            // Increment indices
            let mut carry = true;
            for d in (0..rank).rev() {
                if carry {
                    indices[d] += 1;
                    if indices[d] < result_shape[d] {
                        carry = false;
                    } else {
                        indices[d] = 0;
                    }
                }
            }
            if carry {
                break; // All indices exhausted
            }
        }
        
        result
    } else {
        // TODO: Support other dimensions by permuting axes
        panic!("Only dim=0 is currently supported. Got dim={}", dim);
    }
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
    
    // Convert s to Vec<f64> (mdarray-linalg stores singular values in first row: s[[0, i]])
    let min_dim = s.shape().0.min(s.shape().1);
    let s_vec: Vec<f64> = (0..min_dim).map(|i| s[[0, i]]).collect();
    
    SamplingMatrixSVD {
        u,
        s: s_vec,
        vt,
    }
}

