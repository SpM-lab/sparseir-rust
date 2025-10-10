//! Sparse sampling in Matsubara frequencies
//!
//! This module provides Matsubara frequency sampling for transforming between
//! IR basis coefficients and values at sparse Matsubara frequencies.

use crate::basis::FiniteTempBasis;
use crate::fitter::{ComplexMatrixFitter, ComplexToRealFitter};
use crate::freq::MatsubaraFreq;
use crate::kernel::{KernelProperties, CentrosymmKernel};
use crate::traits::StatisticsType;
use crate::gemm::matmul_par;
use mdarray::{DTensor, Tensor, DynRank, Shape};
use num_complex::Complex;
use std::marker::PhantomData;

/// Move axis from position src to position dst
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
            if pos == src {
                pos += 1;
            }
            if pos < rank {
                perm.push(pos);
                pos += 1;
            }
        }
    }
    
    arr.permute(&perm[..]).to_tensor()
}

/// Matsubara sampling for full frequency range (positive and negative)
///
/// General complex problem without symmetry → complex coefficients
pub struct MatsubaraSampling<S: StatisticsType> {
    sampling_points: Vec<MatsubaraFreq<S>>,
    fitter: ComplexMatrixFitter,
    _phantom: PhantomData<S>,
}

impl<S: StatisticsType> MatsubaraSampling<S> {
    // TODO: Implement default sampling point selection
    // pub fn new<K>(basis: &FiniteTempBasis<K, S>) -> Self { ... }
    
    /// Create Matsubara sampling with custom sampling points
    pub fn with_sampling_points<K>(
        basis: &FiniteTempBasis<K, S>,
        mut sampling_points: Vec<MatsubaraFreq<S>>,
    ) -> Self
    where
        K: KernelProperties + CentrosymmKernel + Clone + 'static,
    {
        // Sort sampling points
        sampling_points.sort();
        
        // Evaluate matrix at sampling points
        let matrix = eval_matrix_matsubara(basis, &sampling_points);
        
        // Create fitter (complex → complex, no symmetry)
        let fitter = ComplexMatrixFitter::new(matrix);
        
        Self {
            sampling_points,
            fitter,
            _phantom: PhantomData,
        }
    }
    
    /// Get sampling points
    pub fn sampling_points(&self) -> &[MatsubaraFreq<S>] {
        &self.sampling_points
    }
    
    /// Number of sampling points
    pub fn n_sampling_points(&self) -> usize {
        self.sampling_points.len()
    }
    
    /// Basis size
    pub fn basis_size(&self) -> usize {
        self.fitter.basis_size()
    }
    
    /// Evaluate complex basis coefficients at sampling points
    ///
    /// # Arguments
    /// * `coeffs` - Complex basis coefficients (length = basis_size)
    ///
    /// # Returns
    /// Complex values at Matsubara frequencies (length = n_sampling_points)
    pub fn evaluate(&self, coeffs: &[Complex<f64>]) -> Vec<Complex<f64>> {
        self.fitter.evaluate(coeffs)
    }
    
    /// Fit complex basis coefficients from values at sampling points
    ///
    /// # Arguments
    /// * `values` - Complex values at Matsubara frequencies (length = n_sampling_points)
    ///
    /// # Returns
    /// Fitted complex basis coefficients (length = basis_size)
    pub fn fit(&self, values: &[Complex<f64>]) -> Vec<Complex<f64>> {
        self.fitter.fit(values)
    }
    
    /// Evaluate N-dimensional array of complex basis coefficients at sampling points
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional tensor of complex basis coefficients
    /// * `dim` - Dimension along which to evaluate (must have size = basis_size)
    ///
    /// # Returns
    /// N-dimensional tensor of complex values at Matsubara frequencies
    pub fn evaluate_nd(
        &self,
        coeffs: &Tensor<Complex<f64>, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank> {
        let rank = coeffs.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        
        let basis_size = self.basis_size();
        let target_dim_size = coeffs.shape().dim(dim);
        
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
        
        let coeffs_2d_dyn = coeffs_dim0.reshape(&[basis_size, extra_size][..]).to_tensor();
        let coeffs_2d = DTensor::<Complex<f64>, 2>::from_fn([basis_size, extra_size], |idx| {
            coeffs_2d_dyn[&[idx[0], idx[1]][..]]
        });
        
        // 3. Matrix multiply: result = A * coeffs (both complex)
        let n_points = self.n_sampling_points();
        let result_2d = matmul_par(&self.fitter.matrix, &coeffs_2d);
        
        // 4. Reshape back to N-D with n_points at position 0
        let mut result_shape = vec![n_points];
        coeffs_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                result_shape.push(dims[i]);
            }
        });
        
        let result_dim0 = result_2d.into_dyn().reshape(&result_shape[..]).to_tensor();
        
        // 5. Move dimension 0 back to original position dim
        movedim(&result_dim0, 0, dim)
    }
    
    /// Fit N-dimensional array of complex values to complex basis coefficients
    ///
    /// # Arguments
    /// * `values` - N-dimensional tensor of complex values at Matsubara frequencies
    /// * `dim` - Dimension along which to fit (must have size = n_sampling_points)
    ///
    /// # Returns
    /// N-dimensional tensor of complex basis coefficients
    pub fn fit_nd(
        &self,
        values: &Tensor<Complex<f64>, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank> {
        let rank = values.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        
        let n_points = self.n_sampling_points();
        let target_dim_size = values.shape().dim(dim);
        
        assert_eq!(
            target_dim_size,
            n_points,
            "values.shape().dim({}) = {} must equal n_sampling_points = {}",
            dim,
            target_dim_size,
            n_points
        );
        
        // 1. Move target dimension to position 0
        let values_dim0 = movedim(values, dim, 0);
        
        // 2. Reshape to 2D: (n_points, extra_size)
        let extra_size: usize = values_dim0.len() / n_points;
        
        let values_2d_dyn = values_dim0.reshape(&[n_points, extra_size][..]).to_tensor();
        
        // 3. Fit each column using the fitter
        let basis_size = self.basis_size();
        let mut coeffs_2d = DTensor::<Complex<f64>, 2>::from_fn([basis_size, extra_size], |_| {
            Complex::new(0.0, 0.0)
        });
        
        for j in 0..extra_size {
            let column: Vec<Complex<f64>> = (0..n_points)
                .map(|i| values_2d_dyn[&[i, j][..]])
                .collect();
            let fitted = self.fitter.fit(&column);
            for i in 0..basis_size {
                coeffs_2d[[i, j]] = fitted[i];
            }
        }
        
        // 4. Reshape back to N-D with basis_size at position 0
        let mut coeffs_shape = vec![basis_size];
        values_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                coeffs_shape.push(dims[i]);
            }
        });
        
        let coeffs_dim0 = coeffs_2d.into_dyn().reshape(&coeffs_shape[..]).to_tensor();
        
        // 5. Move dimension 0 back to original position dim
        movedim(&coeffs_dim0, 0, dim)
    }
}

/// Matsubara sampling for positive frequencies only
///
/// Exploits symmetry to reconstruct real coefficients from positive frequencies only.
/// Supports: {0, 1, 2, 3, ...} (no negative frequencies)
pub struct MatsubaraSamplingPositiveOnly<S: StatisticsType> {
    sampling_points: Vec<MatsubaraFreq<S>>,
    fitter: ComplexToRealFitter,
    _phantom: PhantomData<S>,
}

impl<S: StatisticsType> MatsubaraSamplingPositiveOnly<S> {
    // TODO: Implement default sampling point selection
    // pub fn new<K>(basis: &FiniteTempBasis<K, S>) -> Self { ... }
    
    /// Create Matsubara sampling with custom positive-only sampling points
    pub fn with_sampling_points<K>(
        basis: &FiniteTempBasis<K, S>,
        mut sampling_points: Vec<MatsubaraFreq<S>>,
    ) -> Self
    where
        K: KernelProperties + CentrosymmKernel + Clone + 'static,
    {
        // Sort and validate (all n >= 0)
        sampling_points.sort();
        
        // TODO: Validate that all points are non-negative
        
        // Evaluate matrix at sampling points
        let matrix = eval_matrix_matsubara(basis, &sampling_points);
        
        // Create fitter (complex → real, exploits symmetry)
        let fitter = ComplexToRealFitter::new(&matrix);
        
        Self {
            sampling_points,
            fitter,
            _phantom: PhantomData,
        }
    }
    
    /// Get sampling points
    pub fn sampling_points(&self) -> &[MatsubaraFreq<S>] {
        &self.sampling_points
    }
    
    /// Number of sampling points
    pub fn n_sampling_points(&self) -> usize {
        self.sampling_points.len()
    }
    
    /// Basis size
    pub fn basis_size(&self) -> usize {
        self.fitter.basis_size()
    }
    
    /// Evaluate basis coefficients at sampling points
    pub fn evaluate(&self, coeffs: &[f64]) -> Vec<Complex<f64>> {
        self.fitter.evaluate(coeffs)
    }
    
    /// Fit basis coefficients from values at sampling points
    pub fn fit(&self, values: &[Complex<f64>]) -> Vec<f64> {
        self.fitter.fit(values)
    }
    
    /// Evaluate N-dimensional array of real basis coefficients at sampling points
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional tensor of real basis coefficients
    /// * `dim` - Dimension along which to evaluate (must have size = basis_size)
    ///
    /// # Returns
    /// N-dimensional tensor of complex values at Matsubara frequencies
    pub fn evaluate_nd(
        &self,
        coeffs: &Tensor<f64, DynRank>,
        dim: usize,
    ) -> Tensor<Complex<f64>, DynRank> {
        let rank = coeffs.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        
        let basis_size = self.basis_size();
        let target_dim_size = coeffs.shape().dim(dim);
        
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
        
        let coeffs_2d_dyn = coeffs_dim0.reshape(&[basis_size, extra_size][..]).to_tensor();
        
        // 3. Evaluate each column using the fitter
        let n_points = self.n_sampling_points();
        let mut result_2d = DTensor::<Complex<f64>, 2>::from_fn([n_points, extra_size], |_| {
            Complex::new(0.0, 0.0)
        });
        
        for j in 0..extra_size {
            let column: Vec<f64> = (0..basis_size)
                .map(|i| coeffs_2d_dyn[&[i, j][..]])
                .collect();
            let evaluated = self.fitter.evaluate(&column);
            for i in 0..n_points {
                result_2d[[i, j]] = evaluated[i];
            }
        }
        
        // 4. Reshape back to N-D with n_points at position 0
        let mut result_shape = vec![n_points];
        coeffs_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                result_shape.push(dims[i]);
            }
        });
        
        let result_dim0 = result_2d.into_dyn().reshape(&result_shape[..]).to_tensor();
        
        // 5. Move dimension 0 back to original position dim
        movedim(&result_dim0, 0, dim)
    }
    
    /// Fit N-dimensional array of complex values to real basis coefficients
    ///
    /// # Arguments
    /// * `values` - N-dimensional tensor of complex values at Matsubara frequencies
    /// * `dim` - Dimension along which to fit (must have size = n_sampling_points)
    ///
    /// # Returns
    /// N-dimensional tensor of real basis coefficients
    pub fn fit_nd(
        &self,
        values: &Tensor<Complex<f64>, DynRank>,
        dim: usize,
    ) -> Tensor<f64, DynRank> {
        let rank = values.rank();
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        
        let n_points = self.n_sampling_points();
        let target_dim_size = values.shape().dim(dim);
        
        assert_eq!(
            target_dim_size,
            n_points,
            "values.shape().dim({}) = {} must equal n_sampling_points = {}",
            dim,
            target_dim_size,
            n_points
        );
        
        // 1. Move target dimension to position 0
        let values_dim0 = movedim(values, dim, 0);
        
        // 2. Reshape to 2D: (n_points, extra_size)
        let extra_size: usize = values_dim0.len() / n_points;
        
        let values_2d_dyn = values_dim0.reshape(&[n_points, extra_size][..]).to_tensor();
        
        // 3. Fit each column using the fitter
        let basis_size = self.basis_size();
        let mut coeffs_2d = DTensor::<f64, 2>::from_fn([basis_size, extra_size], |_| 0.0);
        
        for j in 0..extra_size {
            let column: Vec<Complex<f64>> = (0..n_points)
                .map(|i| values_2d_dyn[&[i, j][..]])
                .collect();
            let fitted = self.fitter.fit(&column);
            for i in 0..basis_size {
                coeffs_2d[[i, j]] = fitted[i];
            }
        }
        
        // 4. Reshape back to N-D with basis_size at position 0
        let mut coeffs_shape = vec![basis_size];
        values_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                coeffs_shape.push(dims[i]);
            }
        });
        
        let coeffs_dim0 = coeffs_2d.into_dyn().reshape(&coeffs_shape[..]).to_tensor();
        
        // 5. Move dimension 0 back to original position dim
        movedim(&coeffs_dim0, 0, dim)
    }
}

/// Evaluate the sampling matrix at Matsubara frequencies: A[i, l] = û_l(iωn_i)
fn eval_matrix_matsubara<K, S>(
    basis: &FiniteTempBasis<K, S>,
    sampling_points: &[MatsubaraFreq<S>],
) -> DTensor<Complex<f64>, 2>
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType,
{
    let n_points = sampling_points.len();
    let basis_size = basis.size();
    
    // A[i, l] = û_l(iωn_i)
    DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
        let (i, l) = (idx[0], idx[1]);
        let freq = &sampling_points[i];
        basis.uhat[l].evaluate(freq)
    })
}

