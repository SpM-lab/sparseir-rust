//! TwoFloat-specific traits for high-precision numerical computation
//!
//! This module provides custom traits that allow TwoFloat to work with the gauss module
//! without requiring the standard Float, FromPrimitive, ToPrimitive, and ScalarOperand traits.

use twofloat::TwoFloat;
use ndarray::Array1;
use std::fmt::{Debug, Display};

/// Custom numeric trait for high-precision numerical computation
///
/// This trait provides the essential numeric operations needed for gauss module
/// without requiring the standard num_traits implementations.
/// Supports f64 and TwoFloat types.
pub trait CustomNumeric: 
    Copy + Debug + PartialOrd + Display + 
    std::ops::Add<Output = Self> + 
    std::ops::Sub<Output = Self> + 
    std::ops::Mul<Output = Self> + 
    std::ops::Div<Output = Self> +
    std::ops::Neg<Output = Self> {
    /// Convert from f64 to Self
    fn from_f64(x: f64) -> Self;
    
    /// Get machine epsilon
    fn epsilon() -> Self;
    
    /// Absolute value
    fn abs(self) -> Self;
    
    /// Cosine function
    fn cos(self) -> Self;
    
    /// Sine function  
    fn sin(self) -> Self;
    
    /// Square root
    fn sqrt(self) -> Self;
    
    /// Check if value is finite
    fn is_finite(self) -> bool;
}

/// f64 implementation of CustomNumeric
impl CustomNumeric for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
    
    fn epsilon() -> Self {
        f64::EPSILON
    }
    
    fn abs(self) -> Self {
        self.abs()
    }
    
    fn cos(self) -> Self {
        self.cos()
    }
    
    fn sin(self) -> Self {
        self.sin()
    }
    
    fn sqrt(self) -> Self {
        self.sqrt()
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
    
    fn epsilon() -> Self {
        TwoFloat::from_f64(f64::EPSILON)
    }
    
    fn abs(self) -> Self {
        self.abs()
    }
    
    fn cos(self) -> Self {
        self.cos()
    }
    
    fn sin(self) -> Self {
        self.sin()
    }
    
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    
    fn is_finite(self) -> bool {
        self.is_valid()
    }
}

// Note: ScalarOperand implementation for TwoFloat would require a wrapper type
// due to Rust's orphan rules. This is handled differently in the gauss module.

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
    fn test_f64_numeric_trait() {
        let x = f64::from_f64(1.5);
        let y = f64::from_f64(-2.0);
        
        // Test basic operations
        assert_eq!(x.abs(), 1.5);
        assert_eq!(y.abs(), 2.0);
        
        // Test mathematical functions
        let cos_x = x.cos();
        assert!(cos_x.is_finite());
        
        let sqrt_x = x.sqrt();
        assert!(sqrt_x.is_finite());
    }
    
    #[test]
    fn test_twofloat_numeric_trait() {
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
    }
    
    #[test]
    fn test_twofloat_array_ops() {
        let arr = Array1::from(vec![TwoFloat::from_f64(1.0), TwoFloat::from_f64(2.0)]);
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
    fn test_twofloat_epsilon() {
        let eps = TwoFloat::epsilon();
        assert!(eps > TwoFloat::from_f64(0.0));
        assert!(eps < TwoFloat::from_f64(1.0));
    }
}
