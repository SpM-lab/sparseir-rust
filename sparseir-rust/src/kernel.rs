//! Kernel implementations for SparseIR
//!
//! This module provides kernel implementations for analytical continuation in quantum many-body physics.
//! The kernels are used in Fredholm integral equations of the first kind:
//!
//!     u(x) = ∫ K(x, y) v(y) dy
//!
//! where x ∈ [xmin, xmax] and y ∈ [ymin, ymax].
//!
//! In general, the kernel is applied to a scaled spectral function ρ'(y) as:
//!
//!     ∫ K(x, y) ρ'(y) dy,
//!
//! where ρ'(y) = w(y) ρ(y). The weight function w(y) transforms the original spectral
//! function ρ(y) into the scaled version ρ'(y) used in the integral equation.

use twofloat::TwoFloat;
use crate::traits::{StatisticsType, Statistics, Fermionic, Bosonic};

/// Trait for kernel type properties (static characteristics)
pub trait KernelProperties {
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
    
    /// Inverse weight function (1/w) to avoid division by zero.
    /// 
    /// This is a safer API that returns 1/w(beta, omega) directly, avoiding
    /// potential division by zero issues when w(beta, omega) approaches zero.
    /// 
    /// @param beta Inverse temperature  
    /// @param omega Frequency
    /// @return The inverse weight value 1/w(beta, omega)
    fn inv_weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64;
}

/// Abstract kernel trait for SparseIR kernels
pub trait AbstractKernel {
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

/// Reduced kernel for centrosymmetric kernels restricted to positive interval.
/// 
/// For a kernel K on [-1, 1] × [-1, 1] that is centrosymmetric, i.e.,
/// K(x, y) = K(-x, -y), it is straightforward to show that the left/right
/// singular vectors can be chosen as either odd or even functions.
/// 
/// Consequently, they are singular functions of a reduced kernel K_red on
/// [0, 1] × [0, 1] that is given as either:
/// 
///     K_red(x, y) = K(x, y) ± K(x, -y)
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
}
