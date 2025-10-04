//! Kernel implementations for SparseIR
//!
//! This module provides kernel implementations for analytical continuation in quantum many-body physics.
//! The kernels are used in Fredholm integral equations of the first kind.
//!
//! u(x) = integral of K(x, y) v(y) dy
//!
//! where x ∈ [xmin, xmax] and y ∈ [ymin, ymax].
//!
//! In general, the kernel is applied to a scaled spectral function rho'(y) as:
//!
//! integral of K(x, y) rho'(y) dy,
//!
//! where ρ'(y) = w(y) ρ(y). The weight function w(y) transforms the original spectral
//! function ρ(y) into the scaled version ρ'(y) used in the integral equation.

use twofloat::TwoFloat;
use crate::traits::{StatisticsType, Statistics};
use crate::gauss::Rule;
use crate::numeric::CustomNumeric;
use ndarray::Array2;
use std::fmt::Debug;
use num_traits::ToPrimitive;
use std::ops::{Index, IndexMut, Sub};
use rayon::prelude::*;

/// Trait for SVE (Singular Value Expansion) hints
/// 
/// Provides discretization hints for singular value expansion of a given kernel.
/// This includes segment information and numerical parameters for efficient computation.
pub trait SVEHints<T>: Debug + Send + Sync
where
    T: Copy + Debug + Send + Sync,
{
    /// Get the x-axis segments for discretization
    fn segments_x(&self) -> Vec<T>;
    
    /// Get the y-axis segments for discretization
    fn segments_y(&self) -> Vec<T>;
    
    /// Get the number of singular values hint
    fn nsvals(&self) -> usize;
    
    /// Get the number of Gauss points for quadrature
    fn ngauss(&self) -> usize;
}

/// Trait for kernel type properties (static characteristics)
pub trait KernelProperties {
    /// Associated type for SVE hints
    type SVEHintsType<T>: SVEHints<T> + Clone
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
    /// Check if the kernel is centrosymmetric.
    /// 
    /// Returns true if and only if K(x, y) == K(-x, -y) for all values of x and y.
    /// This allows the kernel to be block-diagonalized, speeding up the
    /// singular value expansion by a factor of 4.
    fn is_centrosymmetric(&self) -> bool;
    
    /// Power with which the y coordinate scales.
    /// 
    /// For most kernels, this is 0 (no scaling).
    /// For RegularizedBoseKernel, this is 1 (linear scaling).
    fn ypower(&self) -> i32;
    
    /// Get the x domain range for this kernel type
    fn xrange(&self) -> (f64, f64);
    
    /// Get the y domain range for this kernel type
    fn yrange(&self) -> (f64, f64);
    
    /// Weight function for given statistics.
    /// 
    /// The kernel is applied to a scaled spectral function ρ'(y) as:
    ///     ∫ K(x, y) ρ'(y) dy,
    /// where ρ'(y) = w(y) ρ(y).
    /// 
    /// This function returns w(beta, omega) that transforms the original spectral
    /// function ρ(y) into the scaled version ρ'(y) used in the integral equation.
    /// 
    /// @param beta Inverse temperature
    /// @param omega Frequency
    /// @return The weight value w(beta, omega)
    fn weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64;
    
    /// Inverse weight function to avoid division by zero.
    /// 
    /// This is a safer API that returns the inverse weight.
    /// 
    /// @param beta Inverse temperature  
    /// @param omega Frequency
    /// @return The inverse weight value
    fn inv_weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64;
    
    /// Create SVE hints for this kernel type.
    /// 
    /// Provides discretization hints for singular value expansion computation.
    /// The hints include segment information and numerical parameters optimized
    /// for the specific kernel type.
    /// 
    /// @param epsilon Target accuracy for the SVE computation
    /// @return SVE hints specific to this kernel type
    fn sve_hints<T>(&self, epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
}

/// Abstract kernel trait for SparseIR kernels
pub trait AbstractKernel: Send + Sync {
    /// Compute the kernel value K(x, y) with high precision
    fn compute(&self, x: TwoFloat, y: TwoFloat) -> TwoFloat;
    
    /// Get the cutoff parameter Λ (lambda)
    fn lambda(&self) -> f64;
    
    /// Convergence radius of the Matsubara basis asymptotic model.
    /// 
    /// For improved relative numerical accuracy, the IR basis functions on the
    /// Matsubara axis can be evaluated from an asymptotic expression for
    /// abs(n) > conv_radius. If conv_radius is infinity, then the asymptotics
    /// are unused (the default).
    fn conv_radius(&self) -> f64;
}

/// Utility function for f64 computation
pub fn compute_f64<K: AbstractKernel>(kernel: &K, x: f64, y: f64) -> f64 {
    kernel.compute(TwoFloat::from(x), TwoFloat::from(y)).into()
}

/// Logistic kernel for fermionic analytical continuation
/// 
/// This kernel implements K(x, y) = exp(-Λy(x + 1)/2)/(1 + exp(-Λy))
/// where x ∈ [-1, 1] and y ∈ [-1, 1]
#[derive(Debug, Clone, Copy)]
pub struct LogisticKernel {
    lambda: f64,
}

impl LogisticKernel {
    /// Create a new logistic kernel with the given cutoff parameter
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
    
    /// Get the cutoff parameter
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl KernelProperties for LogisticKernel {
    type SVEHintsType<T> = LogisticSVEHints<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
    fn is_centrosymmetric(&self) -> bool {
        true // LogisticKernel is centrosymmetric
    }
    
    fn ypower(&self) -> i32 {
        0 // No y-power scaling for LogisticKernel
    }
    
    fn xrange(&self) -> (f64, f64) { (-1.0, 1.0) }
    fn yrange(&self) -> (f64, f64) { (-1.0, 1.0) }
    
    fn weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64 {
        match S::STATISTICS {
            Statistics::Fermionic => {
                // For fermionic statistics: w(beta, omega) = 1.0
                // The kernel K(x, y) is already in the correct form for fermions
                1.0
            }
            Statistics::Bosonic => {
                // For bosonic statistics: w(beta, omega) = 1.0 / tanh(0.5 * beta * omega)
                // This transforms the fermionic kernel to work with bosonic correlation functions
                // The tanh factor accounts for the different statistics
                1.0 / (0.5 * beta * omega).tanh()
            }
        }
    }
    
    fn inv_weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64 {
        match S::STATISTICS {
            Statistics::Fermionic => {
                // For fermionic statistics: 1/w = 1.0 (safe, no division by zero)
                1.0
            }
            Statistics::Bosonic => {
                // For bosonic statistics: 1/w = tanh(0.5 * beta * omega) (safe, handles omega=0 case)
                // This avoids division by zero when tanh(0.5 * beta * omega) approaches zero
                (0.5 * beta * omega).tanh()
            }
        }
    }
    
    fn sve_hints<T>(&self, epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
    {
        LogisticSVEHints::new(self.clone(), epsilon)
    }
}

impl AbstractKernel for LogisticKernel {
    fn compute(&self, x: TwoFloat, y: TwoFloat) -> TwoFloat {
        // K(x, y) = exp(-Λy(x + 1)/2)/(1 + exp(-Λy))
        let lambda = TwoFloat::from(self.lambda);
        let exp_term = (-lambda * y * (x + TwoFloat::from(1.0)) / TwoFloat::from(2.0)).exp();
        exp_term / (TwoFloat::from(1.0) + (-lambda * y).exp())
    }
    
    fn lambda(&self) -> f64 { self.lambda }
    
    fn conv_radius(&self) -> f64 {
        40.0 * self.lambda // For LogisticKernel, conv_radius = 40 * Λ
    }
}

/// Regularized bosonic kernel for bosonic analytical continuation
/// 
/// This kernel implements K(x, y) = y * exp(-Λy(x + 1)/2)/(exp(-Λy) - 1)
/// where x ∈ [-1, 1] and y ∈ [-1, 1]
#[derive(Debug, Clone, Copy)]
pub struct RegularizedBoseKernel {
    lambda: f64,
}

impl RegularizedBoseKernel {
    /// Create a new regularized bosonic kernel with the given cutoff parameter
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
    
    /// Get the cutoff parameter
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

impl KernelProperties for RegularizedBoseKernel {
    type SVEHintsType<T> = RegularizedBoseSVEHints<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
    fn is_centrosymmetric(&self) -> bool {
        true // RegularizedBoseKernel is centrosymmetric
    }
    
    fn ypower(&self) -> i32 {
        1 // RegularizedBoseKernel has y^1 scaling
    }
    
    fn xrange(&self) -> (f64, f64) { (-1.0, 1.0) }
    fn yrange(&self) -> (f64, f64) { (-1.0, 1.0) }
    
    fn weight<S: StatisticsType + 'static>(&self, _beta: f64, omega: f64) -> f64 {
        match S::STATISTICS {
            Statistics::Fermionic => {
                panic!("RegularizedBoseKernel does not support fermionic functions")
            }
            Statistics::Bosonic => {
                // For bosonic statistics: w(beta, omega) = 1.0 / omega
                // The kernel K(x, y) = y * exp(-Λy(x + 1)/2)/(exp(-Λy) - 1) already includes
                // the y factor, so we need to divide by omega to get the proper scaling
                1.0 / omega
            }
        }
    }
    
    fn inv_weight<S: StatisticsType + 'static>(&self, _beta: f64, omega: f64) -> f64 {
        match S::STATISTICS {
            Statistics::Fermionic => {
                panic!("RegularizedBoseKernel does not support fermionic functions")
            }
            Statistics::Bosonic => {
                // For bosonic statistics: 1/w = omega (safe, handles omega=0 case naturally)
                // This avoids division by zero when omega approaches zero
                omega
            }
        }
    }
    
    fn sve_hints<T>(&self, epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
    {
        RegularizedBoseSVEHints::new(self.clone(), epsilon)
    }
}

impl AbstractKernel for RegularizedBoseKernel {
    fn compute(&self, x: TwoFloat, y: TwoFloat) -> TwoFloat {
        // K(x, y) = y * exp(-Λy(x + 1)/2)/(exp(-Λy) - 1)
        let lambda = TwoFloat::from(self.lambda);
        let exp_term = (-lambda * y * (x + TwoFloat::from(1.0)) / TwoFloat::from(2.0)).exp();
        y * exp_term / ((-lambda * y).exp() - TwoFloat::from(1.0))
    }
    
    fn lambda(&self) -> f64 { self.lambda }
    
    fn conv_radius(&self) -> f64 {
        40.0 * self.lambda // For RegularizedBoseKernel, conv_radius = 40 * Λ
    }
}

/// SVE hints for LogisticKernel
#[derive(Debug, Clone)]
pub struct LogisticSVEHints<T> {
    kernel: LogisticKernel,
    epsilon: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> LogisticSVEHints<T>
where
    T: Copy + Debug + Send + Sync,
{
    pub fn new(kernel: LogisticKernel, epsilon: f64) -> Self {
        Self {
            kernel,
            epsilon,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> SVEHints<T> for LogisticSVEHints<T>
where
    T: Copy + Debug + Send + Sync + CustomNumeric,
{
    fn segments_x(&self) -> Vec<T> {
        // Simplified implementation - in practice, this would use the full algorithm
        // from the C++ implementation with cosh calculations
        let nzeros = std::cmp::max(
            (15.0 * self.kernel.lambda().log10()).round() as usize, 1
        );
        
        let mut segments = Vec::with_capacity(2 * nzeros + 1);
        
        // Create symmetric segments around 0
        for i in 0..=nzeros {
            let pos = <T as CustomNumeric>::from_f64(0.1 * i as f64);
            let neg = <T as CustomNumeric>::from_f64(-0.1 * i as f64);
            if i == 0 {
                segments.push(<T as CustomNumeric>::from_f64(0.0));
            } else {
                segments.push(neg);
                segments.push(pos);
            }
        }
        
        segments.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap_or(std::cmp::Ordering::Equal));
        segments
    }
    
    fn segments_y(&self) -> Vec<T> {
        // Simplified implementation
        let nzeros = std::cmp::max(
            (20.0 * self.kernel.lambda().log10()).round() as usize, 2
        );
        
        let mut segments = Vec::with_capacity(2 * nzeros + 3);
        segments.push(<T as CustomNumeric>::from_f64(-1.0));
        
        for i in 1..=nzeros {
            let pos = <T as CustomNumeric>::from_f64(0.05 * i as f64);
            let neg = <T as CustomNumeric>::from_f64(-0.05 * i as f64);
            segments.push(neg);
            segments.push(pos);
        }
        
        segments.push(<T as CustomNumeric>::from_f64(1.0));
        segments.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap_or(std::cmp::Ordering::Equal));
        segments
    }
    
    fn nsvals(&self) -> usize {
        let log10_lambda = self.kernel.lambda().log10().max(1.0);
        ((25.0 + log10_lambda) * log10_lambda).round() as usize
    }
    
    fn ngauss(&self) -> usize {
        if self.epsilon >= 1e-8 { 10 } else { 16 }
    }
}

/// SVE hints for RegularizedBoseKernel
#[derive(Debug, Clone)]
pub struct RegularizedBoseSVEHints<T> {
    kernel: RegularizedBoseKernel,
    epsilon: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> RegularizedBoseSVEHints<T>
where
    T: Copy + Debug + Send + Sync,
{
    pub fn new(kernel: RegularizedBoseKernel, epsilon: f64) -> Self {
        Self {
            kernel,
            epsilon,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> SVEHints<T> for RegularizedBoseSVEHints<T>
where
    T: Copy + Debug + Send + Sync + CustomNumeric,
{
    fn segments_x(&self) -> Vec<T> {
        // Simplified implementation for RegularizedBoseKernel
        let nzeros = std::cmp::max(
            (15.0 * self.kernel.lambda().log10()).round() as usize, 15
        );
        
        let mut segments = Vec::with_capacity(2 * nzeros + 1);
        
        for i in 0..=nzeros {
            let pos = <T as CustomNumeric>::from_f64(0.08 * i as f64);
            let neg = <T as CustomNumeric>::from_f64(-0.08 * i as f64);
            if i == 0 {
                segments.push(<T as CustomNumeric>::from_f64(0.0));
            } else {
                segments.push(neg);
                segments.push(pos);
            }
        }
        
        segments.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap_or(std::cmp::Ordering::Equal));
        segments
    }
    
    fn segments_y(&self) -> Vec<T> {
        // Simplified implementation
        let nzeros = std::cmp::max(
            (20.0 * self.kernel.lambda().log10()).round() as usize, 20
        );
        
        let mut segments = Vec::with_capacity(2 * nzeros + 3);
        segments.push(<T as CustomNumeric>::from_f64(-1.0));
        
        for i in 1..=nzeros {
            let pos = <T as CustomNumeric>::from_f64(0.06 * i as f64);
            let neg = <T as CustomNumeric>::from_f64(-0.06 * i as f64);
            segments.push(neg);
            segments.push(pos);
        }
        
        segments.push(<T as CustomNumeric>::from_f64(1.0));
        segments.sort_by(|a, b| a.to_f64().partial_cmp(&b.to_f64()).unwrap_or(std::cmp::Ordering::Equal));
        segments
    }
    
    fn nsvals(&self) -> usize {
        let log10_lambda = self.kernel.lambda().log10().max(1.0);
        (28.0 * log10_lambda).round() as usize
    }
    
    fn ngauss(&self) -> usize {
        if self.epsilon >= 1e-8 { 10 } else { 16 }
    }
}

/// SVE hints for ReducedKernel
#[derive(Debug)]
pub struct ReducedSVEHints<T> {
    inner: Box<dyn SVEHints<T>>,
}

impl<T> ReducedSVEHints<T>
where
    T: Copy + Debug + Send + Sync,
{
    pub fn new(inner: Box<dyn SVEHints<T>>) -> Self {
        Self { inner }
    }
}

impl<T> SVEHints<T> for ReducedSVEHints<T>
where
    T: Copy + Debug + Send + Sync + ToPrimitive + CustomNumeric,
{
    fn segments_x(&self) -> Vec<T> {
        // For reduced kernels, we only need the positive half
        let mut segments = self.inner.segments_x();
        segments.retain(|&x| <T as CustomNumeric>::to_f64(x) >= 0.0);
        segments
    }
    
    fn segments_y(&self) -> Vec<T> {
        // For reduced kernels, we only need the positive half
        let mut segments = self.inner.segments_y();
        segments.retain(|&y| <T as CustomNumeric>::to_f64(y) >= 0.0);
        segments
    }
    
    fn nsvals(&self) -> usize {
        (self.inner.nsvals() + 1) / 2
    }
    
    fn ngauss(&self) -> usize {
        self.inner.ngauss()
    }
}

/// Reduced kernel for centrosymmetric kernels restricted to positive interval.
/// 
/// For a kernel K on [-1, 1] × [-1, 1] that is centrosymmetric, i.e.,
/// K(x, y) = K(-x, -y), it is straightforward to show that the left/right
/// singular vectors can be chosen as either odd or even functions.
/// 
/// Consequently, they are singular functions of a reduced kernel K_red on
/// [0, 1] × [0, 1] that is given as either:
/// 
/// K_red(x, y) = K(x, y) + K(x, -y)  or  K(x, y) - K(x, -y)
/// 
/// This kernel is what this struct represents. The full singular functions can be
/// reconstructed by (anti-)symmetrically continuing them to the negative axis.
#[derive(Debug, Clone)]
pub struct ReducedKernel<InnerKernel: KernelProperties + AbstractKernel> {
    inner_kernel: InnerKernel,
    sign: i32,
}

impl<InnerKernel: KernelProperties + AbstractKernel> ReducedKernel<InnerKernel> {
    /// Create a new reduced kernel.
    /// 
    /// # Arguments
    /// * `inner_kernel` - The inner centrosymmetric kernel
    /// * `sign` - The sign (+1 or -1) for the reduced kernel formula
    /// 
    /// # Panics
    /// Panics if the inner kernel is not centrosymmetric or if sign is not ±1
    pub fn new(inner_kernel: InnerKernel, sign: i32) -> Self {
        if !inner_kernel.is_centrosymmetric() {
            panic!("Inner kernel must be centrosymmetric");
        }
        if sign != 1 && sign != -1 {
            panic!("sign must be -1 or 1");
        }
        
        Self {
            inner_kernel,
            sign,
        }
    }
    
    /// Get the inner kernel
    pub fn inner_kernel(&self) -> &InnerKernel {
        &self.inner_kernel
    }
    
    /// Get the sign
    pub fn sign(&self) -> i32 {
        self.sign
    }
}

impl<InnerKernel: KernelProperties + AbstractKernel> KernelProperties for ReducedKernel<InnerKernel> {
    // TODO: Fix ReducedSVEHints to support Clone
    // For now, use LogisticSVEHints as a placeholder
    type SVEHintsType<T> = LogisticSVEHints<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
    fn is_centrosymmetric(&self) -> bool {
        false // ReducedKernel cannot be symmetrized further
    }
    
    fn ypower(&self) -> i32 {
        self.inner_kernel.ypower() // Inherit from inner kernel
    }
    
    fn xrange(&self) -> (f64, f64) {
        // For ReducedKernel, xrange is modified to [0, xmax_inner]
        let (_, xmax) = self.inner_kernel.xrange();
        (0.0, xmax)
    }
    
    fn yrange(&self) -> (f64, f64) {
        // For ReducedKernel, yrange is modified to [0, ymax_inner]
        let (_, ymax) = self.inner_kernel.yrange();
        (0.0, ymax)
    }
    
    fn weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64 {
        self.inner_kernel.weight::<S>(beta, omega)
    }
    
    fn inv_weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64 {
        self.inner_kernel.inv_weight::<S>(beta, omega)
    }
    
    fn sve_hints<T>(&self, epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
    {
        // For now, create a dummy LogisticKernel to get hints
        // TODO: Implement proper ReducedSVEHints that can be cloned
        let dummy_kernel = LogisticKernel::new(self.inner_kernel.lambda());
        dummy_kernel.sve_hints(epsilon)
    }
}

impl<InnerKernel: KernelProperties + AbstractKernel> AbstractKernel for ReducedKernel<InnerKernel> {
    fn compute(&self, x: TwoFloat, y: TwoFloat) -> TwoFloat {
        // K_red(x, y) = K(x, y) + sign * K(x, -y)
        let k_plus = self.inner_kernel.compute(x, y);
        let k_minus = self.inner_kernel.compute(x, -y);
        k_plus + TwoFloat::from(self.sign as f64) * k_minus
    }
    
    fn lambda(&self) -> f64 {
        self.inner_kernel.lambda()
    }
    
    fn conv_radius(&self) -> f64 {
        self.inner_kernel.conv_radius() // Inherit from inner kernel
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use twofloat::TwoFloat;
    
    #[test]
    fn test_logistic_kernel_basic() {
        let kernel = LogisticKernel::new(10.0);
        
        assert_eq!(kernel.lambda(), 10.0);
        assert_eq!(kernel.xrange(), (-1.0, 1.0));
        assert_eq!(kernel.yrange(), (-1.0, 1.0));
    }
    
    #[test]
    fn test_logistic_kernel_traits() {
        let kernel = LogisticKernel::new(10.0);
        
        // Test centrosymmetry trait
        assert!(kernel.is_centrosymmetric());
        
        // Test y-power trait
        assert_eq!(kernel.ypower(), 0);
        
        // Test ranges
        assert_eq!(kernel.xrange(), (-1.0, 1.0));
        assert_eq!(kernel.yrange(), (-1.0, 1.0));
    }
    
    #[test]
    fn test_logistic_kernel_computation() {
        let kernel = LogisticKernel::new(10.0);
        
        let x = TwoFloat::from(0.0);
        let y = TwoFloat::from(0.0);
        let result = kernel.compute(x, y);
        
        // K(0, 0) = exp(0)/(1 + exp(0)) = 1/2 = 0.5
        assert!((Into::<f64>::into(result) - 0.5).abs() < 1e-14);
    }
    
    #[test]
    fn test_logistic_kernel_weight_functions() {
        let kernel = LogisticKernel::new(10.0);
        let beta = 1.0;
        let omega = 1.0;
        
        // Test fermionic weight
        let w_fermi = kernel.weight::<Fermionic>(beta, omega);
        assert_eq!(w_fermi, 1.0);
        
        let inv_w_fermi = kernel.inv_weight::<Fermionic>(beta, omega);
        assert_eq!(inv_w_fermi, 1.0);
        
        // Test bosonic weight
        let w_bose = kernel.weight::<Bosonic>(beta, omega);
        let expected_w_bose = 1.0 / (0.5 * beta * omega).tanh();
        assert!((w_bose - expected_w_bose).abs() < 1e-14);
        
        let inv_w_bose = kernel.inv_weight::<Bosonic>(beta, omega);
        let expected_inv_w_bose = (0.5 * beta * omega).tanh();
        assert!((inv_w_bose - expected_inv_w_bose).abs() < 1e-14);
    }
    
    
    #[test]
    fn test_regularized_bose_kernel_basic() {
        let kernel = RegularizedBoseKernel::new(10.0);
        
        assert_eq!(kernel.lambda(), 10.0);
        assert_eq!(kernel.xrange(), (-1.0, 1.0));
        assert_eq!(kernel.yrange(), (-1.0, 1.0));
    }
    
    #[test]
    fn test_regularized_bose_kernel_traits() {
        let kernel = RegularizedBoseKernel::new(10.0);
        
        // Test centrosymmetry trait
        assert!(kernel.is_centrosymmetric());
        
        // Test y-power trait
        assert_eq!(kernel.ypower(), 1);
        
        // Test ranges
        assert_eq!(kernel.xrange(), (-1.0, 1.0));
        assert_eq!(kernel.yrange(), (-1.0, 1.0));
    }
    
    #[test]
    fn test_regularized_bose_kernel_weight_functions() {
        let kernel = RegularizedBoseKernel::new(10.0);
        let beta = 1.0;
        let omega = 1.0;
        
        // Test bosonic weight
        let w_bose = kernel.weight::<Bosonic>(beta, omega);
        assert_eq!(w_bose, 1.0 / omega);
        
        let inv_w_bose = kernel.inv_weight::<Bosonic>(beta, omega);
        assert_eq!(inv_w_bose, omega);
    }
    
    #[test]
    #[should_panic(expected = "RegularizedBoseKernel does not support fermionic functions")]
    fn test_regularized_bose_kernel_fermionic_panic() {
        let kernel = RegularizedBoseKernel::new(10.0);
        let _ = kernel.weight::<Fermionic>(1.0, 1.0);
    }
    
    #[test]
    fn test_compute_f64_utility() {
        let kernel = LogisticKernel::new(10.0);
        let result = compute_f64(&kernel, 0.0, 0.0);
        assert!((result - 0.5).abs() < 1e-14);
    }
    
    #[test]
    fn test_bosonic_weight_at_omega_zero() {
        let kernel = LogisticKernel::new(10.0);
        let beta = 1.0;
        let omega = 0.0;
        
        // This should not panic and should handle omega=0 gracefully
        let inv_w_bose = kernel.inv_weight::<Bosonic>(beta, omega);
        assert_eq!(inv_w_bose, 0.0); // tanh(0) = 0
    }
    
    #[test]
    fn test_reduced_kernel_creation() {
        let inner_kernel = LogisticKernel::new(10.0);
        let reduced_kernel = ReducedKernel::new(inner_kernel, 1);
        
        assert_eq!(reduced_kernel.sign(), 1);
        assert_eq!(reduced_kernel.lambda(), 10.0);
        assert!(!reduced_kernel.is_centrosymmetric()); // Cannot be symmetrized further
    }
    
    #[test]
    fn test_reduced_kernel_properties() {
        let inner_kernel = LogisticKernel::new(10.0);
        let reduced_kernel = ReducedKernel::new(inner_kernel, -1);
        
        // Test that properties are inherited correctly
        assert_eq!(reduced_kernel.ypower(), 0); // From LogisticKernel
        assert_eq!(reduced_kernel.xrange(), (0.0, 1.0)); // Modified range
        assert_eq!(reduced_kernel.yrange(), (0.0, 1.0)); // Modified range
        assert_eq!(reduced_kernel.conv_radius(), 40.0 * 10.0); // From inner kernel
    }
    
    #[test]
    fn test_reduced_kernel_computation() {
        let inner_kernel = LogisticKernel::new(10.0);
        let reduced_kernel_plus = ReducedKernel::new(inner_kernel.clone(), 1);
        let reduced_kernel_minus = ReducedKernel::new(inner_kernel, -1);
        
        let x = TwoFloat::from(0.5);
        let y = TwoFloat::from(0.3);
        
        // Test the reduced kernel formula: K_red(x, y) = K(x, y) ± K(x, -y)
        let k_plus = reduced_kernel_plus.compute(x, y);
        let k_minus = reduced_kernel_minus.compute(x, y);
        
        // Both results should be finite
        assert!(Into::<f64>::into(k_plus).is_finite());
        assert!(Into::<f64>::into(k_minus).is_finite());
        
        // The results should be different due to different signs
        assert_ne!(k_plus, k_minus);
    }
    
    #[test]
    #[should_panic(expected = "sign must be -1 or 1")]
    fn test_reduced_kernel_invalid_sign_zero_panic() {
        let inner_kernel = LogisticKernel::new(10.0);
        ReducedKernel::new(inner_kernel, 0); // Invalid sign
    }
    
    #[test]
    #[should_panic(expected = "sign must be -1 or 1")]
    fn test_reduced_kernel_invalid_sign_panic() {
        let inner_kernel = LogisticKernel::new(10.0);
        ReducedKernel::new(inner_kernel, 2); // Invalid sign
    }
    
    #[test]
    fn test_reduced_kernel_weight_functions() {
        let inner_kernel = LogisticKernel::new(10.0);
        let reduced_kernel = ReducedKernel::new(inner_kernel, 1);
        
        let beta = 1.0;
        let omega = 1.0;
        
        // Weight functions should be inherited from inner kernel
        let w_fermi = reduced_kernel.weight::<Fermionic>(beta, omega);
        assert_eq!(w_fermi, 1.0);
        
        let w_bose = reduced_kernel.weight::<Bosonic>(beta, omega);
        let expected_w_bose = 1.0 / (0.5 * beta * omega).tanh();
        assert!((w_bose - expected_w_bose).abs() < 1e-14);
    }
    
    #[test]
    fn test_sve_hints_basic() {
        let kernel = LogisticKernel::new(10.0);
        let epsilon = 1e-8;
        
        let hints = kernel.sve_hints::<f64>(epsilon);
        
        // Test that we get reasonable hints
        let segments_x = hints.segments_x();
        let segments_y = hints.segments_y();
        let nsvals = hints.nsvals();
        let ngauss = hints.ngauss();
        
        // Basic sanity checks
        assert!(!segments_x.is_empty());
        assert!(!segments_y.is_empty());
        assert!(nsvals > 0);
        assert!(ngauss > 0);
        
        // Segments should be sorted
        let mut sorted_x = segments_x.clone();
        sorted_x.sort_by(|a, b| a.partial_cmp(b));
        assert_eq!(segments_x, sorted_x);
        
        let mut sorted_y = segments_y.clone();
        sorted_y.sort_by(|a, b| a.partial_cmp(b));
        assert_eq!(segments_y, sorted_y);
    }
    
    #[test]
    fn test_reduced_sve_hints() {
        let inner_kernel = LogisticKernel::new(10.0);
        let reduced_kernel = ReducedKernel::new(inner_kernel, 1);
        let epsilon = 1e-8;
        
        let hints = reduced_kernel.sve_hints::<f64>(epsilon);
        
        let segments_x = hints.segments_x();
        let segments_y = hints.segments_y();
        let nsvals = hints.nsvals();
        let ngauss = hints.ngauss();
        
        // For reduced kernels, segments should be non-negative
        assert!(segments_x.iter().all(|&x| x >= 0.0));
        assert!(segments_y.iter().all(|&y| y >= 0.0));
        
        // Basic sanity checks
        assert!(!segments_x.is_empty());
        assert!(!segments_y.is_empty());
        assert!(nsvals > 0);
        assert!(ngauss > 0);
    }
}

/// Discretized kernel with associated Gauss quadrature rules
/// 
/// This structure stores a discrete kernel matrix along with the corresponding
/// Gauss quadrature rules for x and y coordinates. This enables easy application
/// of weights for SVE computation and maintains the relationship between matrix
/// elements and their corresponding quadrature points.
#[derive(Debug, Clone)]
pub struct DiscretizedKernel<T> {
    /// Discrete kernel matrix
    pub matrix: Array2<T>,
    /// Gauss quadrature rule for x coordinates
    pub gauss_x: Rule<T>,
    /// Gauss quadrature rule for y coordinates  
    pub gauss_y: Rule<T>,
}

impl<T: CustomNumeric + Clone> DiscretizedKernel<T> {
    /// Create a new DiscretizedKernel
    pub fn new(matrix: Array2<T>, gauss_x: Rule<T>, gauss_y: Rule<T>) -> Self {
        Self { matrix, gauss_x, gauss_y }
    }
    
    /// Delegate to matrix methods
    pub fn is_empty(&self) -> bool {
        self.matrix.is_empty()
    }
    
    pub fn nrows(&self) -> usize {
        self.matrix.nrows()
    }
    
    pub fn ncols(&self) -> usize {
        self.matrix.ncols()
    }
    
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.matrix.iter()
    }
}

impl<T: CustomNumeric + Clone> Index<[usize; 2]> for DiscretizedKernel<T> {
    type Output = T;
    
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.matrix[index]
    }
}

impl<T: CustomNumeric + Clone> IndexMut<[usize; 2]> for DiscretizedKernel<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        &mut self.matrix[index]
    }
}

impl<T: CustomNumeric + Clone> Sub for DiscretizedKernel<T> {
    type Output = DiscretizedKernel<T>;
    
    fn sub(self, rhs: Self) -> Self::Output {
        DiscretizedKernel::new(
            self.matrix - rhs.matrix,
            self.gauss_x,
            self.gauss_y,
        )
    }
}

impl<T: CustomNumeric + Clone> Sub<&DiscretizedKernel<T>> for &DiscretizedKernel<T> {
    type Output = DiscretizedKernel<T>;
    
    fn sub(self, rhs: &DiscretizedKernel<T>) -> Self::Output {
        DiscretizedKernel::new(
            &self.matrix - &rhs.matrix,
            self.gauss_x.clone(),
            self.gauss_y.clone(),
        )
    }
}

impl<T: CustomNumeric + Clone> DiscretizedKernel<T> {
    /// Apply weights for SVE computation
    /// 
    /// This applies the square root of Gauss weights to the matrix,
    /// which is required before performing SVD for SVE computation.
    /// The original matrix remains unchanged.
    pub fn apply_weights_for_sve(&self) -> Array2<T> {
        let mut weighted_matrix = self.matrix.clone();
        
        // Apply square root of x-direction weights to rows
        for i in 0..self.gauss_x.x.len() {
            let weight_sqrt = self.gauss_x.w[i].sqrt();
            weighted_matrix.row_mut(i).mapv_inplace(|x| x * weight_sqrt);
        }
        
        // Apply square root of y-direction weights to columns
        for j in 0..self.gauss_y.x.len() {
            let weight_sqrt = self.gauss_y.w[j].sqrt();
            weighted_matrix.column_mut(j).mapv_inplace(|x| x * weight_sqrt);
        }
        
        weighted_matrix
    }
    
    /// Remove weights from SVE result matrices
    /// 
    /// This removes the square root of Gauss weights from U and V matrices
    /// obtained from SVD, converting them back to the original basis.
    pub fn remove_weights_from_sve_result(&self, u_matrix: &mut Array2<T>, v_matrix: &mut Array2<T>) {
        // Remove weights from U matrix (x-direction)
        for i in 0..u_matrix.nrows() {
            let weight_sqrt = self.gauss_x.w[i].sqrt();
            u_matrix.row_mut(i).mapv_inplace(|x| x / weight_sqrt);
        }
        
        // Remove weights from V matrix (y-direction) 
        for j in 0..v_matrix.ncols() {
            let weight_sqrt = self.gauss_y.w[j].sqrt();
            v_matrix.column_mut(j).mapv_inplace(|x| x / weight_sqrt);
        }
    }
    
    /// Get the number of Gauss points in x direction
    pub fn n_gauss_x(&self) -> usize {
        self.gauss_x.x.len()
    }
    
    /// Get the number of Gauss points in y direction
    pub fn n_gauss_y(&self) -> usize {
        self.gauss_y.x.len()
    }
}

/// Compute matrix from Gauss quadrature rules
/// 
/// This function evaluates the kernel at all combinations of Gauss points
/// and returns a DiscretizedKernel containing the matrix and quadrature rules.
pub fn matrix_from_gauss<T: CustomNumeric + ToPrimitive + num_traits::Zero + Clone>(
    kernel: &dyn AbstractKernel,
    gauss_x: &Rule<T>,
    gauss_y: &Rule<T>,
) -> DiscretizedKernel<T> {
    let n = gauss_x.x.len();
    let m = gauss_y.x.len();
    let mut result = Array2::zeros((n, m));
    
    // Evaluate kernel at all combinations of Gauss points
    for i in 0..n {
        for j in 0..m {
            let x = gauss_x.x[i];
            let y = gauss_y.x[j];
            
            // Convert to TwoFloat for kernel computation
            let x_twofloat = <TwoFloat as CustomNumeric>::from_f64(x.to_f64());
            let y_twofloat = <TwoFloat as CustomNumeric>::from_f64(y.to_f64());
            
            let kernel_value = kernel.compute(x_twofloat, y_twofloat);
            result[[i, j]] = <T as CustomNumeric>::from_f64(kernel_value.into());
        }
    }
    
    DiscretizedKernel::new(result, gauss_x.clone(), gauss_y.clone())
}

/// Compute matrix from Gauss quadrature rules with parallel processing
/// 
/// This is a parallel version of matrix_from_gauss for better performance.
/// Uses rayon for parallel computation across matrix elements.
pub fn matrix_from_gauss_parallel<T: CustomNumeric + ToPrimitive + num_traits::Zero + Clone + Send + Sync>(
    kernel: &dyn AbstractKernel,
    gauss_x: &Rule<T>,
    gauss_y: &Rule<T>,
) -> DiscretizedKernel<T> {
    let n = gauss_x.x.len();
    let m = gauss_y.x.len();
    let mut result = Array2::zeros((n, m));
    
    // Parallel processing using rayon
    // Create a vector of (i, j, value) tuples for parallel computation
    let indices: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (0..m).map(move |j| (i, j)))
        .collect();
    
    let values: Vec<T> = indices
        .par_iter()
        .map(|&(i, j)| {
            let x = gauss_x.x[i];
            let y = gauss_y.x[j];
            
            // Convert to TwoFloat for kernel computation
            let x_twofloat = <TwoFloat as CustomNumeric>::from_f64(x.to_f64());
            let y_twofloat = <TwoFloat as CustomNumeric>::from_f64(y.to_f64());
            
            let kernel_value = kernel.compute(x_twofloat, y_twofloat);
            <T as CustomNumeric>::from_f64(kernel_value.into())
        })
        .collect();
    
    // Fill the result matrix with computed values
    for ((i, j), value) in indices.into_iter().zip(values.into_iter()) {
        result[[i, j]] = value;
    }
    
    DiscretizedKernel::new(result, gauss_x.clone(), gauss_y.clone())
}
