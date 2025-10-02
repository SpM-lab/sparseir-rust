//! Precision type definitions and implementations

// Precision trait implementation for f64

// For now, we focus on f64 implementation only
// ExtendedFloat wrapper for TwoFloat can be added later

/// Trait for precision types used in high-precision SVD computations
pub trait Precision: 
    From<f64> + Into<f64> + Copy + Clone + 
    std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + 
    std::ops::Mul<Output = Self> + std::ops::Div<Output = Self> + 
    std::ops::Neg<Output = Self> + std::ops::AddAssign + std::ops::SubAssign +
    std::cmp::PartialEq + std::cmp::PartialOrd +
    num_traits::Zero + num_traits::One + num_traits::Float {
    /// Machine epsilon for this precision type
    const EPSILON: Self;
    /// Minimum positive value
    const MIN_POSITIVE: Self;
    /// Maximum finite value
    const MAX_VALUE: Self;
    
    /// Square root function
    fn sqrt(self) -> Self;
    
    /// Absolute value function
    fn abs(self) -> Self;
    
    /// Maximum of two values
    fn max(self, other: Self) -> Self;
    
    /// Minimum of two values
    fn min(self, other: Self) -> Self;
    
}

// f64 implementation
impl Precision for f64 {
    const EPSILON: f64 = f64::EPSILON;
    const MIN_POSITIVE: f64 = f64::MIN_POSITIVE;
    const MAX_VALUE: f64 = f64::MAX;
    
    #[inline]
    fn sqrt(self) -> f64 {
        self.sqrt()
    }
    
    #[inline]
    fn abs(self) -> f64 {
        self.abs()
    }
    
    #[inline]
    fn max(self, other: f64) -> f64 {
        self.max(other)
    }
    
    #[inline]
    fn min(self, other: f64) -> f64 {
        self.min(other)
    }
}

// Note: TwoFloat implementation is complex and requires orphan rule workarounds.
// For now, we focus on f64 implementation for the initial version.

#[cfg(feature = "f128")]
mod f128_impl {
    use super::Precision;
    
    // f128 implementation (if available)
    impl Precision for f128 {
        const EPSILON: f128 = f128::EPSILON;
        const MIN_POSITIVE: f128 = f128::MIN_POSITIVE;
        const MAX_VALUE: f128 = f128::MAX;
        
        #[inline]
        fn sqrt(self) -> f128 {
            self.sqrt()
        }
        
        #[inline]
        fn abs(self) -> f128 {
            self.abs()
        }
        
        #[inline]
        fn max(self, other: f128) -> f128 {
            self.max(other)
        }
        
        #[inline]
        fn min(self, other: f128) -> f128 {
            self.min(other)
        }
    }
}
