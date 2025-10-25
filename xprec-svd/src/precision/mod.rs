//! Precision type definitions and implementations

// Precision trait implementation for f64 and xprec-rs Df64 backend

use approx::AbsDiffEq;
use num_traits::Zero;
use simba::scalar::{ComplexField, RealField};
use std::fmt;
use xprec::{CompensatedArithmetic, Df64};

/// Trait for precision types used in high-precision SVD computations
pub trait Precision:
    From<f64>
    + Into<f64>
    + Copy
    + Clone
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::cmp::PartialEq
    + std::cmp::PartialOrd
    + num_traits::Zero
    + num_traits::One
    + num_traits::Float
{
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
    fn epsilon() -> f64 {
        f64::EPSILON
    }
    fn min_positive() -> f64 {
        f64::MIN_POSITIVE
    }
    fn max_value() -> f64 {
        f64::MAX
    }

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

/// Extended precision wrapper backed by `xprec-rs::Df64`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Df64Precision(Df64);

impl Df64Precision {
    pub fn from_f64(value: f64) -> Self {
        Self(Df64::from(value))
    }

    /// Construct from compensated parts.
    #[inline]
    pub fn from_parts(hi: f64, lo: f64) -> Self {
        // SAFETY: `Df64::new_full` requires the compensation invariant.
        // Callers are responsible for providing valid compensated parts,
        // mirroring the expectations of the original TwoFloat API.
        Self(unsafe { Df64::new_full(hi, lo) })
    }

    pub fn to_f64(self) -> f64 {
        f64::from(self.0)
    }

    /// Return the high component of the compensated value.
    #[inline]
    pub fn hi(self) -> f64 {
        f64::from(self.0)
    }

    /// Return the low component of the compensated value.
    #[inline]
    pub fn lo(self) -> f64 {
        CompensatedArithmetic::compensate(&self.0)
    }

    /// Return both components as a tuple.
    #[inline]
    pub fn components(self) -> (f64, f64) {
        (self.hi(), self.lo())
    }

    pub fn epsilon() -> Self {
        Self(Df64::EPSILON)
    }

    pub fn min_positive() -> Self {
        Self(Df64::MIN_POSITIVE)
    }

    pub fn max_value() -> Self {
        Self(Df64::MAX)
    }

    /// High-precision circle constant π.
    #[inline]
    pub fn pi() -> Self {
        Self(<Df64 as RealField>::pi())
    }
}

impl Precision for Df64Precision {
    fn epsilon() -> Self {
        Self::epsilon()
    }

    fn min_positive() -> Self {
        Self::min_positive()
    }

    fn max_value() -> Self {
        Self::max_value()
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    #[inline]
    fn abs(self) -> Self {
        Self(self.0.abs())
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }
}

impl From<f64> for Df64Precision {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<Df64Precision> for f64 {
    fn from(value: Df64Precision) -> Self {
        value.to_f64()
    }
}

impl std::ops::Add for Df64Precision {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Df64Precision {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::Mul for Df64Precision {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl std::ops::Div for Df64Precision {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl std::ops::Rem for Df64Precision {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0 % rhs.0)
    }
}

impl std::ops::Neg for Df64Precision {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl std::ops::AddAssign for Df64Precision {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::ops::SubAssign for Df64Precision {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl std::ops::MulAssign for Df64Precision {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl std::ops::DivAssign for Df64Precision {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl std::ops::RemAssign for Df64Precision {
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}

impl num_traits::Zero for Df64Precision {
    fn zero() -> Self {
        Self(Df64::ZERO)
    }

    fn is_zero(&self) -> bool {
        *self == Self(Df64::ZERO)
    }
}

impl num_traits::One for Df64Precision {
    fn one() -> Self {
        Self(Df64::ONE)
    }
}

impl num_traits::Num for Df64Precision {
    type FromStrRadixErr = String;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        if radix != 10 {
            return Err(format!("radix {} not supported for Df64Precision", radix));
        }
        str.parse::<f64>()
            .map(Self::from_f64)
            .map_err(|err| err.to_string())
    }
}

impl num_traits::NumCast for Df64Precision {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        num_traits::ToPrimitive::to_f64(&n).map(Self::from_f64)
    }
}

impl num_traits::ToPrimitive for Df64Precision {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    fn to_isize(&self) -> Option<isize> {
        self.0.to_isize()
    }

    fn to_i8(&self) -> Option<i8> {
        self.0.to_i8()
    }

    fn to_i16(&self) -> Option<i16> {
        self.0.to_i16()
    }

    fn to_i32(&self) -> Option<i32> {
        self.0.to_i32()
    }

    fn to_i128(&self) -> Option<i128> {
        self.0.to_i128()
    }

    fn to_usize(&self) -> Option<usize> {
        self.0.to_usize()
    }

    fn to_u8(&self) -> Option<u8> {
        self.0.to_u8()
    }

    fn to_u16(&self) -> Option<u16> {
        self.0.to_u16()
    }

    fn to_u32(&self) -> Option<u32> {
        self.0.to_u32()
    }

    fn to_u128(&self) -> Option<u128> {
        self.0.to_u128()
    }

    fn to_f32(&self) -> Option<f32> {
        self.0.to_f32()
    }

    fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }
}

impl num_traits::Float for Df64Precision {
    fn nan() -> Self {
        Self(Df64::NAN)
    }

    fn infinity() -> Self {
        Self(Df64::INFINITY)
    }

    fn neg_infinity() -> Self {
        Self(Df64::NEG_INFINITY)
    }

    fn neg_zero() -> Self {
        Self(Df64::from(-0.0))
    }

    fn min_value() -> Self {
        Self(Df64::MIN)
    }

    fn min_positive_value() -> Self {
        Self(Df64::MIN_POSITIVE)
    }

    fn epsilon() -> Self {
        Self(Df64::EPSILON)
    }

    fn max_value() -> Self {
        Self(Df64::MAX)
    }

    fn is_nan(self) -> bool {
        f64::from(self.0).is_nan()
    }

    fn is_infinite(self) -> bool {
        f64::from(self.0).is_infinite()
    }

    fn is_finite(self) -> bool {
        f64::from(self.0).is_finite()
    }

    fn is_normal(self) -> bool {
        f64::from(self.0).is_normal()
    }

    fn classify(self) -> std::num::FpCategory {
        f64::from(self.0).classify()
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
        Self(self.0.mul_add(a.0, b.0))
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
        if self >= other {
            self - other
        } else {
            Self::zero()
        }
    }

    fn cbrt(self) -> Self {
        let one_third = Df64::from(1.0 / 3.0);
        Self(self.0.powf(one_third))
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
        f64::from(self.0).integer_decode()
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

impl AbsDiffEq for Df64Precision {
    type Epsilon = Self;

    fn default_epsilon() -> Self::Epsilon {
        Self::epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let diff = (*self - *other).abs();
        diff <= epsilon
    }
}

impl fmt::Display for Df64Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.17e}", self.to_f64())
    }
}
