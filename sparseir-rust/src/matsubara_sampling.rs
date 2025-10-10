//! Sparse sampling in Matsubara frequencies
//!
//! This module provides Matsubara frequency sampling for transforming between
//! IR basis coefficients and values at sparse Matsubara frequencies.

use crate::basis::FiniteTempBasis;
use crate::fitter::ComplexToRealFitter;
use crate::freq::MatsubaraFreq;
use crate::kernel::{KernelProperties, CentrosymmKernel};
use crate::traits::StatisticsType;
use mdarray::DTensor;
use num_complex::Complex;
use std::marker::PhantomData;

/// Matsubara sampling for full frequency range (positive and negative)
///
/// Supports symmetric sampling: {..., -3, -2, -1, 0, 1, 2, 3, ...}
pub struct MatsubaraSampling<S: StatisticsType> {
    sampling_points: Vec<MatsubaraFreq<S>>,
    fitter: ComplexToRealFitter,
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
        
        // Create fitter
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
    ///
    /// # Arguments
    /// * `coeffs` - Real basis coefficients (length = basis_size)
    ///
    /// # Returns
    /// Complex values at Matsubara frequencies (length = n_sampling_points)
    pub fn evaluate(&self, coeffs: &[f64]) -> Vec<Complex<f64>> {
        self.fitter.evaluate(coeffs)
    }
    
    /// Fit basis coefficients from values at sampling points
    ///
    /// # Arguments
    /// * `values` - Complex values at Matsubara frequencies (length = n_sampling_points)
    ///
    /// # Returns
    /// Fitted real basis coefficients (length = basis_size)
    pub fn fit(&self, values: &[Complex<f64>]) -> Vec<f64> {
        self.fitter.fit(values)
    }
}

/// Matsubara sampling for positive frequencies only
///
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
        
        // Create fitter
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

