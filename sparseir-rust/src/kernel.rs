//! Kernel implementations for SparseIR

use twofloat::TwoFloat;
use crate::traits::{StatisticsType, Statistics, Fermionic, Bosonic};

/// Abstract kernel trait for SparseIR kernels
pub trait AbstractKernel {
    /// Compute the kernel value K(x, y) with high precision
    fn compute(&self, x: TwoFloat, y: TwoFloat) -> TwoFloat;
    
    /// Get the x domain range
    fn xrange(&self) -> (f64, f64);
    
    /// Get the y domain range
    fn yrange(&self) -> (f64, f64);
    
    /// Get the cutoff parameter Λ (lambda)
    fn lambda(&self) -> f64;
    
    /// Legacy weight function (for compatibility)
    /// Returns the weight value that the kernel should be divided by
    fn weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64;
    
    /// New safer API: inverse weight (1/w) to avoid division by zero
    fn inv_weight<S: StatisticsType + 'static>(&self, beta: f64, omega: f64) -> f64;
    
    /// New convenient API: compute kernel value divided by weight
    fn compute_weighted<S: StatisticsType + 'static>(&self, x: TwoFloat, y: TwoFloat, beta: f64, omega: f64) -> TwoFloat;
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

impl AbstractKernel for LogisticKernel {
    fn compute(&self, x: TwoFloat, y: TwoFloat) -> TwoFloat {
        // K(x, y) = exp(-Λy(x + 1)/2)/(1 + exp(-Λy))
        let lambda = TwoFloat::from(self.lambda);
        let exp_term = (-lambda * y * (x + TwoFloat::from(1.0)) / TwoFloat::from(2.0)).exp();
        exp_term / (TwoFloat::from(1.0) + (-lambda * y).exp())
    }
    
    fn xrange(&self) -> (f64, f64) { (-1.0, 1.0) }
    fn yrange(&self) -> (f64, f64) { (-1.0, 1.0) }
    fn lambda(&self) -> f64 { self.lambda }
    
    fn weight<S: StatisticsType + 'static>(&self, _beta: f64, omega: f64) -> f64 {
        match S::STATISTICS {
            Statistics::Fermionic => {
                // Fermionic: w(beta, omega) = 1.0 (kernel value is divided by this)
                1.0
            }
            Statistics::Bosonic => {
                // Bosonic: w(beta, omega) = 1.0 / tanh(0.5 * beta * omega) (kernel value is divided by this)
                1.0 / (0.5 * _beta * omega).tanh()
            }
        }
    }
    
    fn inv_weight<S: StatisticsType + 'static>(&self, _beta: f64, omega: f64) -> f64 {
        match S::STATISTICS {
            Statistics::Fermionic => {
                // Fermionic: 1/w = 1.0 (safe, no division by zero)
                1.0
            }
            Statistics::Bosonic => {
                // Bosonic: 1/w = tanh(0.5 * beta * omega) (safe, handles omega=0 case)
                (0.5 * _beta * omega).tanh()
            }
        }
    }
    
    fn compute_weighted<S: StatisticsType + 'static>(&self, x: TwoFloat, y: TwoFloat, beta: f64, omega: f64) -> TwoFloat {
        let k_value = self.compute(x, y);
        let inv_w = TwoFloat::from(self.inv_weight::<S>(beta, omega));
        k_value * inv_w
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

impl AbstractKernel for RegularizedBoseKernel {
    fn compute(&self, x: TwoFloat, y: TwoFloat) -> TwoFloat {
        // K(x, y) = y * exp(-Λy(x + 1)/2)/(exp(-Λy) - 1)
        let lambda = TwoFloat::from(self.lambda);
        let exp_term = (-lambda * y * (x + TwoFloat::from(1.0)) / TwoFloat::from(2.0)).exp();
        y * exp_term / ((-lambda * y).exp() - TwoFloat::from(1.0))
    }
    
    fn xrange(&self) -> (f64, f64) { (-1.0, 1.0) }
    fn yrange(&self) -> (f64, f64) { (-1.0, 1.0) }
    fn lambda(&self) -> f64 { self.lambda }
    
    fn weight<S: StatisticsType + 'static>(&self, _beta: f64, omega: f64) -> f64 {
        match S::STATISTICS {
            Statistics::Fermionic => {
                panic!("RegularizedBoseKernel does not support fermionic functions")
            }
            Statistics::Bosonic => {
                // Bosonic: w(beta, omega) = 1.0 / omega (kernel value is divided by this)
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
                // Bosonic: 1/w = omega (safe, handles omega=0 case naturally)
                omega
            }
        }
    }
    
    fn compute_weighted<S: StatisticsType + 'static>(&self, x: TwoFloat, y: TwoFloat, beta: f64, omega: f64) -> TwoFloat {
        let k_value = self.compute(x, y);
        let inv_w = TwoFloat::from(self.inv_weight::<S>(beta, omega));
        k_value * inv_w
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
    fn test_logistic_kernel_weighted_computation() {
        let kernel = LogisticKernel::new(10.0);
        let x = TwoFloat::from(0.0);
        let y = TwoFloat::from(0.0);
        let beta = 1.0;
        let omega = 1.0;
        
        let result_fermi = kernel.compute_weighted::<Fermionic>(x, y, beta, omega);
        let k_value = kernel.compute(x, y);
        let expected_fermi = k_value * TwoFloat::from(kernel.inv_weight::<Fermionic>(beta, omega));
        assert!((result_fermi - expected_fermi).abs() < TwoFloat::from(1e-14));
        
        let result_bose = kernel.compute_weighted::<Bosonic>(x, y, beta, omega);
        let expected_bose = k_value * TwoFloat::from(kernel.inv_weight::<Bosonic>(beta, omega));
        assert!((result_bose - expected_bose).abs() < TwoFloat::from(1e-14));
    }
    
    #[test]
    fn test_regularized_bose_kernel_basic() {
        let kernel = RegularizedBoseKernel::new(10.0);
        
        assert_eq!(kernel.lambda(), 10.0);
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
        
        let result = kernel.compute_weighted::<Bosonic>(TwoFloat::from(0.0), TwoFloat::from(0.0), beta, omega);
        // The result should be 0 since inv_weight is 0
        assert_eq!(Into::<f64>::into(result), 0.0);
    }
}
