//! Precision type definitions and implementations

// Precision trait implementation for f64

// For now, we focus on f64 implementation only
// ExtendedFloat wrapper for TwoFloat can be added later

use approx::AbsDiffEq;

/// Trait for precision types used in high-precision SVD computations
pub trait Precision: 
    From<f64> + Into<f64> + Copy + Clone + 
    std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + 
    std::ops::Mul<Output = Self> + std::ops::Div<Output = Self> + 
    std::ops::Neg<Output = Self> + std::ops::AddAssign + std::ops::SubAssign +
    std::cmp::PartialEq + std::cmp::PartialOrd +
    num_traits::Zero + num_traits::One + num_traits::Float {
    /// Machine epsilon for this precision type
    fn epsilon() -> Self;
    /// Minimum positive value
    fn min_positive() -> Self;
    /// Maximum finite value
    fn max_value() -> Self;
    
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
    fn epsilon() -> f64 { f64::EPSILON }
    fn min_positive() -> f64 { f64::MIN_POSITIVE }
    fn max_value() -> f64 { f64::MAX }
    
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

// TwoFloat wrapper to avoid orphan rule issues
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TwoFloatPrecision(twofloat::TwoFloat);

impl TwoFloatPrecision {
    pub fn new(hi: f64, lo: f64) -> Self {
        Self(twofloat::TwoFloat::new_add(hi, lo))
    }
    
    pub fn from_f64(x: f64) -> Self {
        Self(twofloat::TwoFloat::from(x))
    }
    
    pub fn to_f64(self) -> f64 {
        self.0.into()
    }
    
    pub fn epsilon() -> Self {
        Self::from_f64(f64::EPSILON)
    }
    
    pub fn min_positive() -> Self {
        Self::from_f64(f64::MIN_POSITIVE)
    }
    
    pub fn max_value() -> Self {
        Self::from_f64(f64::MAX)
    }
}

// TwoFloat implementation
impl Precision for TwoFloatPrecision {
    fn epsilon() -> TwoFloatPrecision { TwoFloatPrecision::epsilon() }
    fn min_positive() -> TwoFloatPrecision { TwoFloatPrecision::min_positive() }
    fn max_value() -> TwoFloatPrecision { TwoFloatPrecision::max_value() }
    
    #[inline]
    fn sqrt(self) -> TwoFloatPrecision {
        // Use twofloat's sqrt implementation
        TwoFloatPrecision(self.0.sqrt())
    }
    
    #[inline]
    fn abs(self) -> TwoFloatPrecision {
        TwoFloatPrecision(self.0.abs())
    }
    
    #[inline]
    fn max(self, other: TwoFloatPrecision) -> TwoFloatPrecision {
        if self >= other { self } else { other }
    }
    
    #[inline]
    fn min(self, other: TwoFloatPrecision) -> TwoFloatPrecision {
        if self <= other { self } else { other }
    }
}

// Conversion traits
impl From<f64> for TwoFloatPrecision {
    fn from(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl Into<f64> for TwoFloatPrecision {
    fn into(self) -> f64 {
        self.to_f64()
    }
}

// Arithmetic operations
impl std::ops::Add for TwoFloatPrecision {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl std::ops::Sub for TwoFloatPrecision {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl std::ops::Mul for TwoFloatPrecision {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}

impl std::ops::Div for TwoFloatPrecision {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }
}

impl std::ops::Neg for TwoFloatPrecision {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl std::ops::AddAssign for TwoFloatPrecision {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl std::ops::SubAssign for TwoFloatPrecision {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl std::ops::Rem for TwoFloatPrecision {
    type Output = Self;
    fn rem(self, other: Self) -> Self {
        Self(self.0 % other.0)
    }
}

impl std::ops::RemAssign for TwoFloatPrecision {
    fn rem_assign(&mut self, other: Self) {
        self.0 %= other.0;
    }
}

impl std::ops::MulAssign for TwoFloatPrecision {
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl std::ops::DivAssign for TwoFloatPrecision {
    fn div_assign(&mut self, other: Self) {
        self.0 /= other.0;
    }
}

impl num_traits::Zero for TwoFloatPrecision {
    fn zero() -> Self {
        Self(twofloat::TwoFloat::from(0.0))
    }
    
    fn is_zero(&self) -> bool {
        self.0 == twofloat::TwoFloat::from(0.0)
    }
}

impl num_traits::One for TwoFloatPrecision {
    fn one() -> Self {
        Self(twofloat::TwoFloat::from(1.0))
    }
}

impl num_traits::Num for TwoFloatPrecision {
    type FromStrRadixErr = String;
    
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        // TwoFloat doesn't support parsing, so we'll use f64 as intermediate
        let val = f64::from_str_radix(str, radix).map_err(|e| e.to_string())?;
        Ok(Self::from_f64(val))
    }
}

impl num_traits::NumCast for TwoFloatPrecision {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        n.to_f64().map(Self::from_f64)
    }
}

impl num_traits::ToPrimitive for TwoFloatPrecision {
    fn to_i64(&self) -> Option<i64> {
        let f64_val: f64 = self.0.into();
        f64_val.to_i64()
    }
    
    fn to_u64(&self) -> Option<u64> {
        let f64_val: f64 = self.0.into();
        f64_val.to_u64()
    }
    
    fn to_isize(&self) -> Option<isize> {
        let f64_val: f64 = self.0.into();
        f64_val.to_isize()
    }
    
    fn to_i8(&self) -> Option<i8> {
        let f64_val: f64 = self.0.into();
        f64_val.to_i8()
    }
    
    fn to_i16(&self) -> Option<i16> {
        let f64_val: f64 = self.0.into();
        f64_val.to_i16()
    }
    
    fn to_i32(&self) -> Option<i32> {
        let f64_val: f64 = self.0.into();
        f64_val.to_i32()
    }
    
    fn to_i128(&self) -> Option<i128> {
        let f64_val: f64 = self.0.into();
        f64_val.to_i128()
    }
    
    fn to_usize(&self) -> Option<usize> {
        let f64_val: f64 = self.0.into();
        f64_val.to_usize()
    }
    
    fn to_u8(&self) -> Option<u8> {
        let f64_val: f64 = self.0.into();
        f64_val.to_u8()
    }
    
    fn to_u16(&self) -> Option<u16> {
        let f64_val: f64 = self.0.into();
        f64_val.to_u16()
    }
    
    fn to_u32(&self) -> Option<u32> {
        let f64_val: f64 = self.0.into();
        f64_val.to_u32()
    }
    
    fn to_u128(&self) -> Option<u128> {
        let f64_val: f64 = self.0.into();
        f64_val.to_u128()
    }
    
    fn to_f32(&self) -> Option<f32> {
        let f64_val: f64 = self.0.into();
        f64_val.to_f32()
    }
    
    fn to_f64(&self) -> Option<f64> {
        Some(self.0.into())
    }
}

impl num_traits::Float for TwoFloatPrecision {
    fn nan() -> Self {
        Self(twofloat::TwoFloat::from(f64::NAN))
    }
    
    fn infinity() -> Self {
        Self(twofloat::TwoFloat::from(f64::INFINITY))
    }
    
    fn neg_infinity() -> Self {
        Self(twofloat::TwoFloat::from(f64::NEG_INFINITY))
    }
    
    fn neg_zero() -> Self {
        Self(twofloat::TwoFloat::from(-0.0))
    }
    
    fn min_value() -> Self {
        Self(twofloat::TwoFloat::from(f64::MIN))
    }
    
    fn min_positive_value() -> Self {
        Self(twofloat::TwoFloat::from(f64::MIN_POSITIVE))
    }
    
    fn epsilon() -> Self {
        Self(twofloat::TwoFloat::from(f64::EPSILON))
    }
    
    fn max_value() -> Self {
        Self(twofloat::TwoFloat::from(f64::MAX))
    }
    
    fn is_nan(self) -> bool {
        let f64_val: f64 = self.0.into();
        f64_val.is_nan()
    }
    
    fn is_infinite(self) -> bool {
        let f64_val: f64 = self.0.into();
        f64_val.is_infinite()
    }
    
    fn is_finite(self) -> bool {
        let f64_val: f64 = self.0.into();
        f64_val.is_finite()
    }
    
    fn is_normal(self) -> bool {
        let f64_val: f64 = self.0.into();
        f64_val.is_normal()
    }
    
    fn classify(self) -> std::num::FpCategory {
        let f64_val: f64 = self.0.into();
        f64_val.classify()
    }
    
    fn floor(self) -> Self {
        Self(self.0.floor())
    }
    
    fn ceil(self) -> Self {
        Self(self.0.ceil())
    }
    
    fn round(self) -> Self {
        Self(self.0.round())
    }
    
    fn trunc(self) -> Self {
        Self(self.0.trunc())
    }
    
    fn fract(self) -> Self {
        Self(self.0.fract())
    }
    
    fn abs(self) -> Self {
        Self(self.0.abs())
    }
    
    fn signum(self) -> Self {
        Self(self.0.signum())
    }
    
    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }
    
    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }
    
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0 * a.0 + b.0)
    }
    
    fn recip(self) -> Self {
        Self(self.0.recip())
    }
    
    fn powi(self, n: i32) -> Self {
        Self(self.0.powi(n))
    }
    
    fn powf(self, n: Self) -> Self {
        Self(self.0.powf(n.0))
    }
    
    fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }
    
    fn exp(self) -> Self {
        Self(self.0.exp())
    }
    
    fn exp2(self) -> Self {
        Self(self.0.exp2())
    }
    
    fn ln(self) -> Self {
        Self(self.0.ln())
    }
    
    fn log(self, base: Self) -> Self {
        Self(self.0.log(base.0))
    }
    
    fn log2(self) -> Self {
        Self(self.0.log2())
    }
    
    fn log10(self) -> Self {
        Self(self.0.log10())
    }
    
    fn max(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }
    
    fn min(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }
    
    fn abs_sub(self, other: Self) -> Self {
        if self >= other { self - other } else { num_traits::Zero::zero() }
    }
    
    fn cbrt(self) -> Self {
        Self(self.0.cbrt())
    }
    
    fn hypot(self, other: Self) -> Self {
        Self(self.0.hypot(other.0))
    }
    
    fn sin(self) -> Self {
        Self(self.0.sin())
    }
    
    fn cos(self) -> Self {
        Self(self.0.cos())
    }
    
    fn tan(self) -> Self {
        Self(self.0.tan())
    }
    
    fn asin(self) -> Self {
        Self(self.0.asin())
    }
    
    fn acos(self) -> Self {
        Self(self.0.acos())
    }
    
    fn atan(self) -> Self {
        Self(self.0.atan())
    }
    
    fn atan2(self, other: Self) -> Self {
        Self(self.0.atan2(other.0))
    }
    
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.0.sin_cos();
        (Self(s), Self(c))
    }
    
    fn exp_m1(self) -> Self {
        Self(self.0.exp_m1())
    }
    
    fn ln_1p(self) -> Self {
        Self(self.0.ln_1p())
    }
    
    fn sinh(self) -> Self {
        Self(self.0.sinh())
    }
    
    fn cosh(self) -> Self {
        Self(self.0.cosh())
    }
    
    fn tanh(self) -> Self {
        Self(self.0.tanh())
    }
    
    fn asinh(self) -> Self {
        Self(self.0.asinh())
    }
    
    fn acosh(self) -> Self {
        Self(self.0.acosh())
    }
    
    fn atanh(self) -> Self {
        Self(self.0.atanh())
    }
    
    fn integer_decode(self) -> (u64, i16, i8) {
        let f64_val: f64 = self.0.into();
        f64_val.integer_decode()
    }
}

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

// Implement AbsDiffEq for TwoFloatPrecision
impl AbsDiffEq for TwoFloatPrecision {
    type Epsilon = TwoFloatPrecision;
    
    fn default_epsilon() -> Self::Epsilon {
        Self::epsilon()
    }
    
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let diff = (*self - *other).abs();
        diff <= epsilon
    }
}
