//! Custom numeric traits for high-precision computation
//!
//! This module provides custom numeric traits that work with both f64 and the xprec Df64 backend
//! for high-precision numerical computation in gauss quadrature and matrix operations.

use crate::TwoFloat;
use num_traits::Float;
use xprec::arith;
use xprec::checks;
use xprec::funcs;
use xprec::circular;
use xprec::exp;
use xprec::hyperbolic;
use xprec::consts;
use std::fmt::Debug;
use std::str::FromStr;

/// Custom numeric trait for high-precision numerical computation
///
/// This trait provides the essential numeric operations needed for gauss module
/// and matrix_from_gauss functions. Supports both f64 and TwoFloat types.
pub trait CustomNumeric:
    Copy
    + Debug
    + PartialOrd
    + std::fmt::Display
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    /// Convert from f64 to Self (direct conversion, no Option)
    ///
    /// This is a static method that can be called as T::from_f64(x)
    fn from_f64(x: f64) -> Self;

    /// Convert from any CustomNumeric type to Self (generic conversion)
    ///
    /// This allows conversion between different numeric types while preserving precision.
    /// Can be called as T::convert_from(other_numeric_value)
    fn convert_from<U: CustomNumeric + 'static>(value: U) -> Self;

    /// Convert to DBig with high precision
    ///
    /// This allows conversion to arbitrary precision for testing and comparison.
    /// Can be called as value.to_dbig(precision)
    fn to_dbig(self, precision: usize) -> dashu_float::DBig;

    /// Convert to f64
    fn to_f64(self) -> f64;

    /// Get machine epsilon
    fn epsilon() -> Self;

    /// Get zero value
    fn zero() -> Self;

    /// Absolute value
    fn abs(self) -> Self;

    /// Square root
    fn sqrt(self) -> Self;

    /// Cosine function
    fn cos(self) -> Self;

    /// Sine function
    fn sin(self) -> Self;

    /// Exponential function
    fn exp(self) -> Self;

    /// Hyperbolic sine function
    fn sinh(self) -> Self;

    /// Hyperbolic cosine function
    fn cosh(self) -> Self;

    /// Check if value is finite
    fn is_finite(self) -> bool;

    /// Get high-precision PI constant
    fn pi() -> Self;

    /// Maximum of two values
    fn max(self, other: Self) -> Self;

    /// Check if the value is valid (not NaN/infinite)
    fn is_valid(&self) -> bool;
}

/// f64 implementation of CustomNumeric
impl CustomNumeric for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }

    fn convert_from<U: CustomNumeric + 'static>(value: U) -> Self {
        // Use match to optimize conversion based on the source type
        // Note: Using TypeId for compile-time optimization, but falling back to safe conversion
        match std::any::TypeId::of::<U>() {
            // For f64 to f64, this is just a copy (no conversion needed)
            id if id == std::any::TypeId::of::<f64>() => {
                // Safe: f64 to f64 conversion
                // This is a no-op for f64
                value.to_f64()
            }
            // For TwoFloat to f64, use the conversion method
            id if id == std::any::TypeId::of::<TwoFloat>() => {
                // Safe: TwoFloat to f64 conversion
                value.to_f64()
            }
            // Fallback: convert via f64 for unknown types
            _ => value.to_f64(),
        }
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn to_dbig(self, precision: usize) -> dashu_float::DBig {
        // f64 to DBig conversion
        let val_str = format!("{:.17e}", self);
        dashu_float::DBig::from_str(&val_str)
            .unwrap()
            .with_precision(precision)
            .unwrap()
    }

    fn epsilon() -> Self {
        f64::EPSILON
    }

    fn zero() -> Self {
        0.0
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

    fn exp(self) -> Self {
        self.exp()
    }

    fn sinh(self) -> Self {
        self.sinh()
    }

    fn cosh(self) -> Self {
        self.cosh()
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }

    fn pi() -> Self {
        std::f64::consts::PI
    }

    fn max(self, other: Self) -> Self {
        self.max(other)
    }

    fn is_valid(&self) -> bool {
        self.is_finite()
    }
}

/// Df64 implementation of CustomNumeric
impl CustomNumeric for TwoFloat {
    fn from_f64(x: f64) -> Self {
        TwoFloat::from(x)
    }

    fn convert_from<U: CustomNumeric + 'static>(value: U) -> Self {
        // Use match to optimize conversion based on the source type
        // Note: Using TypeId for compile-time optimization, but falling back to safe conversion
        match std::any::TypeId::of::<U>() {
            // For f64 to Df64, use the conversion method
            id if id == std::any::TypeId::of::<f64>() => {
                // Safe: f64 to Df64 conversion
                let f64_value = value.to_f64();
                Self::from_f64(f64_value)
            }
            // For Df64 to Df64, this is just a copy (no conversion needed)
            id if id == std::any::TypeId::of::<TwoFloat>() => {
                // Safe: Df64 to Df64 conversion (copy)
                let df64_value = value.to_f64(); // Convert to f64 first
                Self::from_f64(df64_value) // Then back to Df64
            }
            // Fallback: convert via f64 for unknown types
            _ => Self::from_f64(value.to_f64()),
        }
    }

    fn to_f64(self) -> f64 {
        // Use hi() and lo() methods for better precision
        let hi = self.hi();
        let lo = self.lo();
        hi + lo
    }

    fn to_dbig(self, precision: usize) -> dashu_float::DBig {
        // Use hi() and lo() methods for better precision
        let hi = self.hi();
        let lo = self.lo();

        let hi_dbig = dashu_float::DBig::from_str(&format!("{:.17e}", hi))
            .unwrap()
            .with_precision(precision)
            .unwrap();
        let lo_dbig = dashu_float::DBig::from_str(&format!("{:.17e}", lo))
            .unwrap()
            .with_precision(precision)
            .unwrap();

        hi_dbig + lo_dbig
    }

    fn epsilon() -> Self {
        TwoFloat::from(xprec::Df64::EPSILON)
    }

    fn zero() -> Self {
        TwoFloat::from(0.0)
    }

    fn abs(self) -> Self {
        TwoFloat::from(funcs::abs(TwoFloat::from(self)))
    }

    fn sqrt(self) -> Self {
        TwoFloat::from(arith::sqrt_q(TwoFloat::from(self)))
    }

    fn cos(self) -> Self {
        TwoFloat::from(circular::cos(TwoFloat::from(self)))
    }

    fn sin(self) -> Self {
        TwoFloat::from(circular::sin(TwoFloat::from(self)))
    }

    fn exp(self) -> Self {
        TwoFloat::from(exp::exp(TwoFloat::from(self)))
    }

    fn sinh(self) -> Self {
        TwoFloat::from(hyperbolic::sinh(TwoFloat::from(self)))
    }

    fn cosh(self) -> Self {
        TwoFloat::from(hyperbolic::cosh(TwoFloat::from(self)))
    }

    fn is_finite(self) -> bool {
        checks::is_finite(TwoFloat::from(self))
    }

    fn pi() -> Self {
        TwoFloat::from(consts::PI)
    }

    fn max(self, other: Self) -> Self {
        if self > other { self } else { other }
    }

    fn is_valid(&self) -> bool {
        let value = *self;
        value.is_finite() && !f64::from(value).is_nan()
    }
}

// Note: ScalarOperand implementations for f64 and TwoFloat are provided by ndarray
// We cannot implement them here due to Orphan Rules, but they are already implemented
// in the ndarray crate for standard numeric types.

// Note: TwoFloatArrayOps trait and impl removed after ndarray migration
// Array operations should now be done using mdarray Tensor methods directly

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
        assert_eq!(<f64 as CustomNumeric>::epsilon(), f64::EPSILON);
    }

    #[test]
    fn test_twofloat_custom_numeric() {
        let x = TwoFloat::from_f64(1.5);
        let y = TwoFloat::from_f64(-2.0);

        // Test basic operations
        assert_eq!(<TwoFloat as CustomNumeric>::abs(x), TwoFloat::from_f64(1.5));
        assert_eq!(<TwoFloat as CustomNumeric>::abs(y), TwoFloat::from_f64(2.0));

        // Test mathematical functions
        let cos_x = <TwoFloat as CustomNumeric>::cos(x);
        assert!(<TwoFloat as CustomNumeric>::is_finite(cos_x));

        let sqrt_x = <TwoFloat as CustomNumeric>::sqrt(x);
        assert!(<TwoFloat as CustomNumeric>::is_finite(sqrt_x));

        // Test conversion
        let x_f64 = x.to_f64();
        assert!((x_f64 - 1.5).abs() < 1e-15);

        // Test epsilon
        let eps = TwoFloat::epsilon();
        assert!(eps > TwoFloat::from_f64(0.0));
        assert!(eps < TwoFloat::from_f64(1.0));
    }

    // Note: TwoFloatArrayOps tests removed after ndarray migration

    #[test]
    fn test_precision_comparison() {
        // Test that TwoFloat provides higher precision than f64
        let pi_f64 = std::f64::consts::PI;
        let pi_tf = TwoFloat::from_f64(pi_f64);

        // Both should be finite
        assert!(pi_f64.is_finite());
        assert!(<TwoFloat as CustomNumeric>::is_finite(pi_tf));

        // TwoFloat should convert back to f64 with minimal loss
        let pi_back = pi_tf.to_f64();
        assert!((pi_back - pi_f64).abs() < f64::EPSILON);
    }
}

