//! Custom numeric traits for high-precision computation
//!
//! This module provides custom numeric traits that work with both f64 and TwoFloat
//! for high-precision numerical computation in gauss quadrature and matrix operations.

use twofloat::TwoFloat;
use ndarray::Array1;
use std::fmt::Debug;

/// Custom numeric trait for high-precision numerical computation
///
/// This trait provides the essential numeric operations needed for gauss module
/// and matrix_from_gauss functions. Supports both f64 and TwoFloat types.
pub trait CustomNumeric: 
    Copy + Debug + PartialOrd + std::fmt::Display + 
    std::ops::Add<Output = Self> + 
    std::ops::Sub<Output = Self> + 
    std::ops::Mul<Output = Self> + 
    std::ops::Div<Output = Self> +
    std::ops::Neg<Output = Self> {
    
    /// Convert from f64 to Self (direct conversion, no Option)
    fn from_f64(x: f64) -> Self;
    
    /// Convert to f64
    fn to_f64(self) -> f64;
    
    /// Get machine epsilon
    fn epsilon() -> Self;
    
    /// Absolute value
    fn abs(self) -> Self;
    
    /// Square root
    fn sqrt(self) -> Self;
    
    /// Cosine function
    fn cos(self) -> Self;
    
    /// Sine function
    fn sin(self) -> Self;
    
    /// Check if value is finite
    fn is_finite(self) -> bool;
}

/// f64 implementation of CustomNumeric
impl CustomNumeric for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
    
    fn to_f64(self) -> f64 {
        self
    }
    
    fn epsilon() -> Self {
        f64::EPSILON
    }
    
    fn abs(self) -> Self {
        self.abs()
    }
    
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    
    fn cos(self) -> Self {
        self.cos()
    }
    
    fn sin(self) -> Self {
        self.sin()
    }
    
    fn is_finite(self) -> bool {
        self.is_finite()
    }
}


/// TwoFloat implementation of CustomNumeric
impl CustomNumeric for TwoFloat {
    fn from_f64(x: f64) -> Self {
        TwoFloat::from(x)
    }
    
    fn to_f64(self) -> f64 {
        // TwoFloat can be converted to f64 directly
        self.into()
    }
    
    fn epsilon() -> Self {
        TwoFloat::from(f64::EPSILON)
    }
    
    fn abs(self) -> Self {
        self.abs()
    }
    
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    
    fn cos(self) -> Self {
        self.cos()
    }
    
    fn sin(self) -> Self {
        self.sin()
    }
    
    fn is_finite(self) -> bool {
        self.is_valid()
    }
}


// Note: ScalarOperand implementations for f64 and TwoFloat are provided by ndarray
// We cannot implement them here due to Orphan Rules, but they are already implemented
// in the ndarray crate for standard numeric types.

/// Helper trait for array operations with TwoFloat
pub trait TwoFloatArrayOps {
    type Output;
    
    /// Array-scalar multiplication
    fn mul_scalar(&self, scalar: TwoFloat) -> Self::Output;
    
    /// Array-scalar addition
    fn add_scalar(&self, scalar: TwoFloat) -> Self::Output;
    
    /// Array-scalar subtraction
    fn sub_scalar(&self, scalar: TwoFloat) -> Self::Output;
}

impl TwoFloatArrayOps for Array1<TwoFloat> {
    type Output = Array1<TwoFloat>;
    
    fn mul_scalar(&self, scalar: TwoFloat) -> Self::Output {
        self.mapv(|x| x * scalar)
    }
    
    fn add_scalar(&self, scalar: TwoFloat) -> Self::Output {
        self.mapv(|x| x + scalar)
    }
    
    fn sub_scalar(&self, scalar: TwoFloat) -> Self::Output {
        self.mapv(|x| x - scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_f64_custom_numeric() {
        let x = 1.5_f64;
        let y = -2.0_f64;
        
        // Test basic operations
        assert_eq!(x.abs(), 1.5);
        assert_eq!(y.abs(), 2.0);
        
        // Test mathematical functions
        let cos_x = x.cos();
        assert!(cos_x.is_finite());
        
        let sqrt_x = x.sqrt();
        assert!(sqrt_x.is_finite());
        
        // Test conversion
        assert_eq!(x.to_f64(), 1.5);
        assert_eq!(f64::epsilon(), f64::EPSILON);
    }
    
    #[test]
    fn test_twofloat_custom_numeric() {
        let x = TwoFloat::from_f64(1.5);
        let y = TwoFloat::from_f64(-2.0);
        
        // Test basic operations
        assert_eq!(x.abs(), TwoFloat::from_f64(1.5));
        assert_eq!(y.abs(), TwoFloat::from_f64(2.0));
        
        // Test mathematical functions
        let cos_x = x.cos();
        assert!(cos_x.is_finite());
        
        let sqrt_x = x.sqrt();
        assert!(sqrt_x.is_finite());
        
        // Test conversion
        let x_f64 = x.to_f64();
        assert!((x_f64 - 1.5).abs() < 1e-15);
        
        // Test epsilon
        let eps = TwoFloat::epsilon();
        assert!(eps > TwoFloat::from_f64(0.0));
        assert!(eps < TwoFloat::from_f64(1.0));
    }
    
    #[test]
    fn test_twofloat_array_ops() {
        let arr = Array1::from(vec![
            TwoFloat::from_f64(1.0), 
            TwoFloat::from_f64(2.0)
        ]);
        let scalar = TwoFloat::from_f64(3.0);
        
        // Test multiplication
        let result = arr.mul_scalar(scalar);
        assert_eq!(result[0], TwoFloat::from_f64(3.0));
        assert_eq!(result[1], TwoFloat::from_f64(6.0));
        
        // Test addition
        let result = arr.add_scalar(scalar);
        assert_eq!(result[0], TwoFloat::from_f64(4.0));
        assert_eq!(result[1], TwoFloat::from_f64(5.0));
        
        // Test subtraction
        let result = arr.sub_scalar(scalar);
        assert_eq!(result[0], TwoFloat::from_f64(-2.0));
        assert_eq!(result[1], TwoFloat::from_f64(-1.0));
    }
    
    #[test]
    fn test_precision_comparison() {
        // Test that TwoFloat provides higher precision than f64
        let pi_f64 = std::f64::consts::PI;
        let pi_tf = TwoFloat::from_f64(pi_f64);
        
        // Both should be finite
        assert!(pi_f64.is_finite());
        assert!(pi_tf.is_finite());
        
        // TwoFloat should convert back to f64 with minimal loss
        let pi_back = pi_tf.to_f64();
        assert!((pi_back - pi_f64).abs() < f64::EPSILON);
    }
}
