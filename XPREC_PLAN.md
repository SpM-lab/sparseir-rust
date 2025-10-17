# libxprec-rust: Pure Rust Migration Plan

## Overview

This document outlines the plan to migrate libxprec (C++ double-double arithmetic library) to pure Rust. The goal is to provide high-performance extended precision arithmetic with equivalent or better performance than the C++ implementation, while maintaining API compatibility where appropriate.

## Current State Analysis

### Existing C++ Implementation (libxprec)
- **Core**: Double-double (quad precision emulation) arithmetic
- **Performance**: Optimized with FMA instructions, 15x slower than native double
- **Precision**: ~106 bits mantissa (vs 53 bits for double)
- **Features**:
  - Basic arithmetic: `+`, `-`, `*`, `/`, `reciprocal`
  - Special functions: `exp`, `sqrt`, `sin`, `cos`, `sinh`, `cosh`, `tanh`
  - Random number generation
  - Eigen integration
  - C interface for FFI
- **Error bounds**: Tight analytical bounds from Joldeș et al. (2018) and Muller & Rideau (2022)

### Existing Rust Implementation (twofloat)
Located at: `/Users/hiroshi/git/sparseir-rust/twofloat/`
- **Status**: Functional double-double implementation
- **Algorithms**: Based on Joldeș et al. (2018) paper
- **Features**: Basic arithmetic, math functions, serialization
- **Limitations**: 
  - Mathematical functions use same precision as result (not fully accurate)
  - Less optimized than libxprec
  - No explicit FMA optimization strategy
  - No integration with mdarray/linalg ecosystem

## Migration Strategy: Fresh Implementation (Option B)

**Decision**: Implement from scratch following libxprec architecture.

**Rationale**:
- **Clean design**: Start with optimal architecture for Rust idioms
- **Full control**: Direct implementation of algorithms from papers
- **Performance**: Optimize for Rust compiler from the ground up
- **No legacy**: Avoid inheriting design decisions from twofloat
- **Learning**: Deep understanding of every algorithm
- **Maintainability**: Clean, well-documented codebase tailored to our needs

**Trade-offs**:
- Higher initial development effort
- Need to implement all features from scratch
- Longer time to first working version
- More comprehensive testing required

**Mitigation**:
- Use libxprec C++ code as reference implementation
- Leverage algorithms from Joldeș et al. (2018) papers
- Start with minimal viable implementation, iterate
- Comprehensive test suite comparing against libxprec

## Critical Implementation Rules

### ⚠️ MANDATORY REQUIREMENTS ⚠️

These rules MUST be followed without exception:

#### 1. Algorithm Fidelity
**RULE**: All arithmetic algorithms must be identical to libxprec implementation.

- **Line-by-line correspondence**: Each Rust function must match the corresponding C++ function in libxprec
- **No simplifications**: Do not "optimize" or "simplify" algorithms without explicit justification
- **Same precision**: Use identical error bounds and flop counts
- **Reference required**: Every algorithm must cite the exact C++ file and line numbers it's ported from

**Example**:
```rust
/// Port of libxprec/src/floats.cpp:42-48
/// Algorithm 4 from Joldeș et al. (2018), p.10
/// Cost: 10 flops, Error bound: 2u²
#[inline]
fn add_dd_d(x: DDouble, y: f64) -> DDouble {
    // Exact translation of C++ code
    let (s, e) = two_sum(x.hi(), y);
    let v = x.lo() + e;
    fast_two_sum(s, v)
}
```

**Verification**:
- Side-by-side code review with C++ source
- Bit-exact output comparison for test cases
- Document any intentional deviations (with rationale and approval)

#### 2. Test Coverage
**RULE**: All tests from libxprec must be ported without omission.

- **Complete test suite**: Port every test from libxprec/test/ directory
- **Same test cases**: Use identical input values and test scenarios
- **No skipping**: Do not skip tests because "Rust is different" or "not applicable"
- **Add, don't subtract**: New tests can be added, but existing ones cannot be removed

**Test inventory checklist**:
```
libxprec/test/
├── [ ] arith.cpp          → tests/arith.rs
├── [ ] circular.cpp       → tests/circular.rs  
├── [ ] compare-mpfloat.hpp → tests/compare_mpfloat.rs
├── [ ] convert.cpp        → tests/convert.rs
├── [ ] eigen.cpp          → tests/eigen.rs (if implementing Eigen integration)
├── [ ] exp.cpp            → tests/exp.rs
├── [ ] gauss.cpp          → tests/gauss.rs
├── [ ] hyperbolic.cpp     → tests/hyperbolic.rs
├── [ ] inline.cpp         → tests/inline.rs
├── [ ] limits.cpp         → tests/limits.rs
├── [ ] mpfloat.cpp        → tests/mpfloat.rs (MPFR reference tests)
├── [ ] random.cpp         → tests/random.rs
├── [ ] round.cpp          → tests/round.rs
└── [ ] sqrt.cpp           → tests/sqrt.rs
```

**Documentation**: Each test module must document which C++ test file it corresponds to.

#### 3. Precision Requirements
**RULE**: Error tolerances must NOT be relaxed from libxprec values.

- **Exact tolerances**: Use the same epsilon/tolerance values as C++ tests
- **No rounding up**: If libxprec uses `1e-30`, do not use `1e-29` or `1e-15`
- **Document bounds**: Every function must state its proven error bound in doc comments
- **Verify bounds**: Test must verify error is within the stated bound

**Tolerance specification**:
```rust
/// Error bound: 2u² where u² ≈ 1.32e-32
const TOLERANCE_2U2: f64 = 2.64e-32;

#[test]
fn test_add_dd_d() {
    let x = DDouble::from_parts(1.0, 1e-20);
    let y = 0.5;
    let result = x + y;
    let expected = DDouble::from_parts(1.5, 1e-20);
    
    // Must use proven error bound, not arbitrary tolerance
    assert_approx_eq!(result, expected, TOLERANCE_2U2);
}
```

**Forbidden practices**:
- ❌ Using `assert!((a - b).abs() < 1e-10)` without justification
- ❌ Increasing tolerance because "test is flaky"
- ❌ Skipping precision checks for "performance"
- ❌ Using different tolerance for Rust than C++

#### 4. Verification Strategy

**Before marking any component as "complete"**:

1. **Algorithm Review**:
   - [ ] Compare Rust code side-by-side with C++ source
   - [ ] Verify identical structure and operations
   - [ ] Document any differences (with explanation)

2. **Test Verification**:
   - [ ] Count tests in C++ file
   - [ ] Count tests in Rust file
   - [ ] Verify counts match
   - [ ] Run diff on test outputs (Rust vs C++)

3. **Precision Verification**:
   - [ ] Extract tolerance values from C++ tests
   - [ ] Verify Rust tests use same values
   - [ ] Run MPFR comparison tests (high-precision reference)
   - [ ] Verify error bounds match paper citations

4. **Documentation**:
   - [ ] Every function cites C++ source location
   - [ ] Every test cites C++ test location
   - [ ] Every error bound cites paper theorem
   - [ ] Deviations (if any) are explicitly documented

#### 5. Review Process

**Pull Request Requirements**:
- Must include "Fidelity Checklist" comparing with C++ code
- Must include test coverage report showing 100% of C++ tests ported
- Must include precision verification results
- Must include benchmark comparison with C++ libxprec

**Example PR Checklist**:
```markdown
## Fidelity Checklist
- [ ] Algorithm matches libxprec/src/exp.cpp:42-89
- [ ] All 15 tests from test/exp.cpp ported
- [ ] Tolerance values match (1e-30 for exp, 1e-28 for expm1)
- [ ] Error bounds verified against Joldeș et al. (2018) Theorem 3.2
- [ ] Benchmark shows performance within 10% of C++
```

### Consequences of Rule Violations

**If any rule is violated**:
1. PR will be rejected
2. Code must be revised to match libxprec
3. Tests must be restored to full coverage
4. Tolerances must be corrected to proven bounds

**Rationale**: The purpose of this port is to provide a **faithful** Rust implementation of libxprec, not a "Rust-ified" approximation. Users depend on the proven numerical properties and exhaustive testing of the C++ library.

## Proposed Architecture

### Directory Structure
```
xprec/
├── Cargo.toml
├── README.md
├── PLAN.md (this file)
├── src/
│   ├── lib.rs              # Main module, public API
│   ├── ddouble.rs           # Core DDouble type (double-double)
│   ├── exdouble.rs          # ExDouble wrapper for normal double
│   ├── power_of_two.rs     # PowerOfTwo type for exact scaling
│   ├── arith/
│   │   ├── mod.rs
│   │   ├── add.rs           # Addition algorithms (Algorithm 1,2,4,6)
│   │   ├── mul.rs           # Multiplication (Algorithm 3)
│   │   ├── div.rs           # Division and reciprocal
│   │   └── fma.rs           # FMA-optimized variants
│   ├── functions/
│   │   ├── mod.rs
│   │   ├── exp.rs           # Exponential (Taylor series)
│   │   ├── sqrt.rs          # Square root (Newton-Raphson)
│   │   ├── trig.rs          # Trigonometric functions
│   │   ├── hyperbolic.rs    # Hyperbolic functions
│   │   └── gauss.rs         # Gaussian functions
│   ├── random.rs            # Random number generation
│   ├── constants.rs         # Mathematical constants (π, e, etc.)
│   ├── convert.rs           # Type conversions
│   ├── format.rs            # Display, Debug formatting
│   └── linalg.rs           # Integration with mdarray/mdarray-linalg
├── benches/
│   └── arithmetic.rs        # Performance benchmarks
└── tests/
    ├── arith.rs             # Arithmetic tests
    ├── functions.rs         # Special functions tests
    └── precision.rs         # Precision/accuracy tests
```

## num-traits Integration

### Core Traits Implementation

DDouble will implement key traits from the `num-traits` crate to ensure seamless integration with the Rust numeric ecosystem:

#### Basic Traits
```rust
use num_traits::{Zero, One, Num, Float, FloatConst};

impl Zero for DDouble {
    fn zero() -> Self { DDouble::from(0.0) }
    fn is_zero(&self) -> bool { self.hi == 0.0 && self.lo == 0.0 }
}

impl One for DDouble {
    fn one() -> Self { DDouble::from(1.0) }
}

impl Num for DDouble {
    type FromStrRadixErr = ParseDDoubleError;
    // ... implementation
}
```

#### Float Trait (Primary Integration Point)
```rust
impl Float for DDouble {
    // Classification
    fn nan() -> Self { ... }
    fn infinity() -> Self { ... }
    fn neg_infinity() -> Self { ... }
    fn neg_zero() -> Self { ... }
    fn min_value() -> Self { ... }
    fn max_value() -> Self { ... }
    fn min_positive_value() -> Self { ... }
    
    // Predicates
    fn is_nan(self) -> bool { ... }
    fn is_infinite(self) -> bool { ... }
    fn is_finite(self) -> bool { ... }
    fn is_normal(self) -> bool { ... }
    fn classify(self) -> FpCategory { ... }
    
    // Arithmetic
    fn floor(self) -> Self { ... }
    fn ceil(self) -> Self { ... }
    fn round(self) -> Self { ... }
    fn trunc(self) -> Self { ... }
    fn fract(self) -> Self { ... }
    fn abs(self) -> Self { ... }
    fn signum(self) -> Self { ... }
    fn copysign(self, sign: Self) -> Self { ... }
    
    // Math functions
    fn mul_add(self, a: Self, b: Self) -> Self { ... }
    fn recip(self) -> Self { ... }
    fn powi(self, n: i32) -> Self { ... }
    fn powf(self, n: Self) -> Self { ... }
    fn sqrt(self) -> Self { ... }
    fn exp(self) -> Self { ... }
    fn exp2(self) -> Self { ... }
    fn ln(self) -> Self { ... }
    fn log(self, base: Self) -> Self { ... }
    fn log2(self) -> Self { ... }
    fn log10(self) -> Self { ... }
    fn max(self, other: Self) -> Self { ... }
    fn min(self, other: Self) -> Self { ... }
    fn abs_sub(self, other: Self) -> Self { ... }
    fn cbrt(self) -> Self { ... }
    fn hypot(self, other: Self) -> Self { ... }
    fn sin(self) -> Self { ... }
    fn cos(self) -> Self { ... }
    fn tan(self) -> Self { ... }
    fn asin(self) -> Self { ... }
    fn acos(self) -> Self { ... }
    fn atan(self) -> Self { ... }
    fn atan2(self, other: Self) -> Self { ... }
    fn sin_cos(self) -> (Self, Self) { ... }
    fn exp_m1(self) -> Self { ... }
    fn ln_1p(self) -> Self { ... }
    fn sinh(self) -> Self { ... }
    fn cosh(self) -> Self { ... }
    fn tanh(self) -> Self { ... }
    fn asinh(self) -> Self { ... }
    fn acosh(self) -> Self { ... }
    fn atanh(self) -> Self { ... }
    
    // Conversion
    fn integer_decode(self) -> (u64, i16, i8) { ... }
}
```

#### FloatConst Trait (Mathematical Constants)
```rust
impl FloatConst for DDouble {
    fn E() -> Self { consts::E }
    fn FRAC_1_PI() -> Self { consts::FRAC_1_PI }
    fn FRAC_1_SQRT_2() -> Self { consts::FRAC_1_SQRT_2 }
    fn FRAC_2_PI() -> Self { consts::FRAC_2_PI }
    fn FRAC_2_SQRT_PI() -> Self { consts::FRAC_2_SQRT_PI }
    fn FRAC_PI_2() -> Self { consts::FRAC_PI_2 }
    fn FRAC_PI_3() -> Self { consts::FRAC_PI_3 }
    fn FRAC_PI_4() -> Self { consts::FRAC_PI_4 }
    fn FRAC_PI_6() -> Self { consts::FRAC_PI_6 }
    fn FRAC_PI_8() -> Self { consts::FRAC_PI_8 }
    fn LN_2() -> Self { consts::LN_2 }
    fn LN_10() -> Self { consts::LN_10 }
    fn LOG2_E() -> Self { consts::LOG2_E }
    fn LOG10_E() -> Self { consts::LOG10_E }
    fn PI() -> Self { consts::PI }
    fn SQRT_2() -> Self { consts::SQRT_2 }
    fn TAU() -> Self { consts::TAU }
}
```

#### Additional Useful Traits
```rust
impl NumCast for DDouble {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> { ... }
}

impl ToPrimitive for DDouble {
    fn to_i64(&self) -> Option<i64> { ... }
    fn to_u64(&self) -> Option<u64> { ... }
    fn to_f32(&self) -> Option<f32> { ... }
    fn to_f64(&self) -> Option<f64> { ... }
}

impl FromPrimitive for DDouble {
    fn from_i64(n: i64) -> Option<Self> { ... }
    fn from_u64(n: u64) -> Option<Self> { ... }
    fn from_f32(n: f32) -> Option<Self> { ... }
    fn from_f64(n: f64) -> Option<Self> { ... }
}

impl Signed for DDouble {
    fn abs(&self) -> Self { ... }
    fn abs_sub(&self, other: &Self) -> Self { ... }
    fn signum(&self) -> Self { ... }
    fn is_positive(&self) -> bool { ... }
    fn is_negative(&self) -> bool { ... }
}
```

### Benefits of num-traits Integration

1. **Generic Programming**: Functions can be written as `fn process<T: Float>(x: T)` and work with both `f64` and `DDouble`
2. **Ecosystem Compatibility**: Works with libraries that accept `num-traits` bounds
3. **Standard Interface**: Users familiar with `f64` can use `DDouble` with same API
4. **Easy Migration**: Drop-in replacement in generic code

### Implementation Priority

This should be implemented early (during arithmetic operations phase) to ensure the API is ergonomic from the start.

## Performance Targets

Based on libxprec C++ benchmarks:

| Operation      | Target (vs f64) | With FMA | Error Bound |
|----------------|-----------------|----------|-------------|
| `dd + double`  | 10-15x slower   | 10x      | 2u²         |
| `dd + dd`      | 15-25x slower   | 20x      | 3u²         |
| `dd * double`  | 5-10x slower    | 6x       | 2u²         |
| `dd * dd`      | 8-15x slower    | 9x       | 4u²         |
| `dd / double`  | 10-15x slower   | 10x      | 3u²         |
| `dd / dd`      | 25-35x slower   | 28x      | 6u²         |
| `reciprocal`   | 20-25x slower   | 19x      | 2.3u²       |

Where u² ≈ 1.32e-32 (square of double epsilon).

## Technical Decisions

### FMA (Fused Multiply-Add)
**Decision**: Support both FMA and non-FMA variants.
- Use compile-time feature detection
- Provide `#[cfg(target_feature = "fma")]` variants
- Fallback to standard operations when FMA unavailable
- Document performance differences

### Unsafe Code
**Decision**: Minimize but allow where necessary for performance.
- No unsafe in core arithmetic algorithms
- Unsafe allowed for:
  - FFI (C interface)
  - SIMD operations (if implemented)
  - Transmute for bit-level operations (carefully audited)
- All unsafe code must have safety comments

### Error Handling
**Decision**: Use Rust's type system, not exceptions.
- Return `Option<DDouble>` for operations that can fail (e.g., `sqrt` of negative)
- Use `f64` semantics for infinity/NaN propagation
- No panics in release builds (except for contract violations in debug)

### Precision vs Performance
**Decision**: Accuracy first, then optimize.
- Implement correct algorithms first
- Profile and optimize hot paths
- Provide "fast" variants where appropriate (with documented trade-offs)

## Implementation Guidelines

### Code Organization Principles

1. **Separation of Concerns**
   - Keep algorithm implementations in separate modules
   - Clear distinction between public API and internal helpers
   - Test modules co-located with implementation

2. **Error Bound Documentation**
   - Every function documents its error bound in terms of u²
   - Include citation to paper/theorem
   - Example:
     ```rust
     /// Fast two-sum algorithm (Algorithm 1, Joldeș et al. 2018)
     /// 
     /// Error bound: 0u² (exact within double-double precision)
     /// Cost: 3 flops
     #[inline]
     fn two_sum(a: f64, b: f64) -> (f64, f64) { ... }
     ```

3. **Performance Annotations**
   - Mark hot path functions with `#[inline]` or `#[inline(always)]`
   - Use `#[cold]` for error paths
   - Document flop count in comments

4. **Testing Strategy**
   - Unit test each algorithm in isolation
   - Integration tests for composed operations
   - Property-based tests using `proptest`
   - Reference tests against libxprec C++ output

### Algorithm Implementation Order

#### Priority 1: Foundation (Week 1-2)
1. **Two-Sum** (Algorithm 1): Building block for all additions
2. **Fast-Two-Sum**: When ordering is known
3. **Two-Product** with FMA (Algorithm 3): Core multiplication
4. **Quick-Two-Sum**: Special case optimization

#### Priority 2: Basic Arithmetic (Week 2-3)
5. **DDouble + f64** (Algorithm 4)
6. **DDouble + DDouble** (Algorithm 6)
7. **DDouble * f64**
8. **DDouble * DDouble**
9. **Division algorithms**

#### Priority 3: Advanced Operations (Week 4-5)
10. **sqrt** via Newton-Raphson
11. **reciprocal** optimization
12. **fma** for DDouble

#### Priority 4: Special Functions (Week 6-9)
13. **exp** family (exp, expm1, exp2)
14. **log** family (log, log1p, log2, log10)
15. **power** (pow, powi, sqrt)
16. **trigonometric** (sin, cos, tan, asin, acos, atan, atan2)
17. **hyperbolic** (sinh, cosh, tanh, asinh, acosh, atanh)

### Reference Implementation Mapping

Map each libxprec C++ file to Rust modules:

| C++ File          | Rust Module      | Description                    |
|-------------------|------------------|--------------------------------|
| `arith.hpp`       | `arith/mod.rs`   | Core arithmetic algorithms     |
| `functions.hpp`   | `functions/mod.rs`| Helper functions              |
| `floats.cpp`      | `ddouble.rs`     | Main DDouble implementation    |
| `sqrt.cpp`        | `functions/sqrt.rs`| Square root                  |
| `exp.cpp`         | `functions/exp.rs` | Exponential functions        |
| `circular.cpp`    | `functions/trig.rs`| Trigonometric functions      |
| `hyperbolic.cpp`  | `functions/hyperbolic.rs`| Hyperbolic functions    |
| `gauss.cpp`       | `functions/gauss.rs`| Gaussian functions          |
| `random.hpp`      | `random.rs`      | RNG integration                |
| `cinterface.cpp`  | `ffi.rs`         | C API                          |

### Key Algorithms from Papers

#### Algorithm 1: Fast2Sum (Joldeș et al. 2018, p.5)
```rust
/// Assumes |a| >= |b|
#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let z = s - a;
    let t = b - z;
    (s, t)
}
```

#### Algorithm 2: 2Sum (Joldeș et al. 2018, p.6)
```rust
/// No ordering assumption
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let aprime = s - b;
    let bprime = s - aprime;
    let delta_a = a - aprime;
    let delta_b = b - bprime;
    let t = delta_a + delta_b;
    (s, t)
}
```

#### Algorithm 3: 2Prod (Joldeș et al. 2018, p.8)
```rust
/// Requires FMA
#[inline]
#[cfg(target_feature = "fma")]
fn two_prod_fma(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = f64::mul_add(a, b, -p);
    (p, e)
}

/// Fallback without FMA
#[inline]
#[cfg(not(target_feature = "fma"))]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    // Split a and b into high/low parts
    // ... Dekker split implementation
    let e = // ... error calculation
    (p, e)
}
```

#### Algorithm 4: Add-DD-D (Joldeș et al. 2018, p.10)
```rust
/// DDouble + f64
#[inline]
fn add_dd_d(x: DDouble, y: f64) -> DDouble {
    let (s, e) = two_sum(x.hi(), y);
    let v = x.lo() + e;
    fast_two_sum(s, v)
}
```

#### Algorithm 6: Add-DD-DD (Joldeș et al. 2018, p.11)
```rust
/// DDouble + DDouble
#[inline]
fn add_dd_dd(x: DDouble, y: DDouble) -> DDouble {
    let (s, e1) = two_sum(x.hi(), y.hi());
    let (t, e2) = two_sum(x.lo(), y.lo());
    let c = e1 + t.hi();
    let (v, w) = fast_two_sum(s, c);
    let z = t.lo() + e2 + w;
    fast_two_sum(v, z)
}
```

## Dependencies

### Core Dependencies
- `num-traits`: Generic numeric traits
- `libm`: Math functions for no_std (optional)

### Optional Dependencies
- `serde`: Serialization support
- `rand`: Random number generation
- `mdarray`: Multi-dimensional array integration
- `mdarray-linalg`: Linear algebra operations

### Development Dependencies
- `criterion`: Benchmarking
- `proptest`: Property-based testing
- `mpfr` (via FFI): Reference implementation for tests

## Documentation Plan

### User Documentation
- [ ] Comprehensive README with quick start
- [ ] API documentation (rustdoc) for all public items
- [ ] Usage guide:
  - When to use extended precision
  - Performance characteristics
  - Accuracy expectations
- [ ] Examples:
  - Basic arithmetic
  - Matrix operations
  - Numerical algorithms (e.g., iterative solvers)
- [ ] Migration guide from twofloat

### Developer Documentation
- [ ] Architecture overview (this document)
- [ ] Algorithm references (papers, page numbers)
- [ ] Contribution guidelines
- [ ] Benchmark methodology

## Success Metrics

1. **Correctness**: All operations match libxprec accuracy within 1ulp
2. **Performance**: Within 10-20% of C++ libxprec with FMA
3. **Ergonomics**: Natural Rust API, zero-cost abstractions where possible
4. **Integration**: Works seamlessly with existing Rust ecosystem
5. **Adoption**: Used in sparseir-rust and xprec-svd as drop-in replacement


## Design Decisions

### 1. Naming Convention
**Decision**: Use `DDouble` (libxprec naming)

**Rationale**:
- Consistency with C++ libxprec
- Clear abbreviation: "Double-Double"
- Avoid confusion with `f64` (Rust's double)

**Optional compatibility**: Provide type alias `type TwoFloat = DDouble` for users migrating from twofloat crate.

### 2. Precision Extensibility
**Decision**: Design for double-double only, but keep extensibility in mind

**Rationale**:
- Focus on getting double-double right first
- API should not preclude future triple-double or quad-double
- Use trait-based design where appropriate

**Future work**: Consider `trait ExtendedFloat` for generic extended precision types.

### 3. SIMD Opportunities
**Decision**: Start without SIMD, profile and evaluate later

**Rationale**:
- Data dependencies in double-double arithmetic limit SIMD benefits
- Focus on correct, maintainable implementation first
- Modern compilers already vectorize simple operations
- Can add SIMD as optimization later if profiling shows benefit

**Research areas**: 
- Parallel processing of DDouble arrays
- Vectorized special functions (exp, sin, etc. on arrays)

### 4. License
**Decision**: Dual MIT/Apache-2.0 license

**Rationale**:
- Standard in Rust ecosystem
- Compatible with libxprec's MIT license
- Maximum flexibility for users
- Follows Rust API guidelines

### 5. Const Generics for Arrays
**Decision**: Use const generics for compile-time optimizations where beneficial

**Example**:
```rust
impl<const N: usize> DDouble {
    pub fn from_array(arr: [f64; N]) -> [DDouble; N] { ... }
}
```

**Rationale**: Zero-cost abstractions, compile-time guarantees

### 6. Error Representation
**Decision**: Follow IEEE 754 semantics for special values

**Handling**:
- `NaN` propagates through operations
- `Infinity` follows standard arithmetic rules
- No exceptions/panics for overflow/underflow
- Return `Option<DDouble>` for operations with preconditions (e.g., `sqrt` of negative)

**Rationale**: Matches Rust's `f64` behavior, familiar to users

## Testing Strategy and Reference Data Generation

### libxprec's Testing Approach

libxprec uses a **two-tier testing strategy** combining unit tests with high-precision reference comparisons.

#### 1. MPFR-Based Reference Implementation

**Core Tool**: `MPFloat` class - a C++ wrapper around **MPFR (Multiple Precision Floating-Point Reliable Library)**

**Key Details**:
- **Precision**: 120 bits mantissa (vs 106 bits for double-double, 53 bits for double)
- **Rounding**: MPFR_RNDN (round to nearest)
- **Purpose**: Ground truth for testing DDouble arithmetic
- **Location**: `libxprec/test/mpfloat.hpp` and `mpfloat.cpp`

**Implementation**:
```cpp
class MPFloat {
    static const mpfr_prec_t precision = 120;
    static const mpfr_rnd_t round = MPFR_RNDN;
    // ... wrapper methods for MPFR operations
};
```

#### 2. Test Macros for Comparison

**Primary Macros** (defined in `compare-mpfloat.hpp`):

```cpp
// Relative error comparison
CMP_UNARY(fn, x, eps)
// Tests: |fn(DDouble(x)) - fn(MPFloat(x))| / |fn(MPFloat(x))| < eps

// Absolute error comparison  
CMP_UNARY_ABS(fn, x, eps)
// Tests: |fn(DDouble(x)) - fn(MPFloat(x))| < eps

// Binary operations
CMP_BINARY(fn, x, y, eps)
CMP_BINARY_1(fn, x, y, eps)  // Second arg is double
CMP_BINARY_EX(fn, x, y, eps) // Uses ExDouble
```

#### 3. Error Tolerance Standards

**Reference ULP** (Unit in the Last Place for double-double):
```cpp
const double ulp = 2.4651903288156619e-32;  // ≈ 2^-105
```

**Typical Tolerances by Operation**:

| Operation | Tolerance | Source File |
|-----------|-----------|-------------|
| Addition/Subtraction (DD+DD) | 1.6 × ulp | `arith.cpp:33-36` |
| Multiplication (DD×DD) | 2 × ulp | `arith.cpp:37` |
| Division (DD÷DD) | 3 × ulp | `arith.cpp:38` |
| `exp()` | 1 × ulp | `exp.cpp:34-43` |
| `expm1()` | 1 × ulp | `exp.cpp:59-76` |
| `log()` | 1 × ulp | `exp.cpp:82-93` |
| `log1p()` | 1.5 × ulp | `exp.cpp:109` |
| `sin/cos` (small x) | 1 × ulp | `circular.cpp:36` |
| `sin/cos` (large x) | 1.5 × ulp × \|x\| | `circular.cpp:43` |
| `sqrt()` | 0.5 × ulp | `sqrt.cpp` |
| `sinh/cosh/tanh` | 1-2 × ulp | `hyperbolic.cpp` |
| `pow(x, n)` | 1-1.5 × ulp | `exp.cpp:16-24` |

**Note**: Tolerances depend on:
- **Input magnitude**: Larger inputs may have relaxed absolute tolerances
- **Algorithm complexity**: More steps accumulate more error
- **Proven error bounds**: From Joldeș et al. (2018) and Muller & Rideau (2022)

#### 4. Test Coverage in libxprec

**Complete Test Suite** (`libxprec/test/`):

```
├── arith.cpp          # Basic arithmetic (±, ×, ÷)
│   └── Tests: DD+DD, DD+D, ExDouble, mixed arithmetic, small additions
├── circular.cpp       # Trigonometric functions
│   └── Tests: sin, cos, tan, asin, acos, atan, atan2, sincos
├── convert.cpp        # Type conversions
│   └── Tests: DDouble ↔ double, string parsing, rounding
├── exp.cpp            # Exponential and logarithmic functions
│   └── Tests: exp, expm1, log, log1p, pow
├── gauss.cpp          # Gaussian functions
│   └── Tests: erf, erfc, Gaussian integrals
├── hyperbolic.cpp     # Hyperbolic functions
│   └── Tests: sinh, cosh, tanh, asinh, acosh, atanh
├── inline.cpp         # Internal utilities (two_sum, two_prod, etc.)
│   └── Tests: Algorithm correctness at bit level
├── limits.cpp         # Edge cases and special values
│   └── Tests: NaN, Infinity, denormals, overflow/underflow
├── mpfloat.cpp        # MPFR wrapper tests
│   └── Tests: MPFloat class itself
├── random.cpp         # Random number generation
│   └── Tests: Statistical properties, distributions
├── round.cpp          # Rounding modes
│   └── Tests: floor, ceil, trunc, round
├── sqrt.cpp           # Square root
│   └── Tests: sqrt accuracy across wide range
└── eigen.cpp          # Eigen library integration (optional)
    └── Tests: Linear algebra operations
```

**Total**: 14 test files, hundreds of test cases covering:
- Wide input ranges (1e-290 to 1e300)
- Edge cases (0, ±∞, NaN, denormals)
- Systematic scanning (e.g., `while ((x *= 0.9) > 1e-290)`)

#### 5. Systematic Test Generation Pattern

**Example from `exp.cpp`**:
```cpp
TEST_CASE("exp", "[exp]")
{
    const double ulp = 2.4651903288156619e-32;
    
    // Small values: very accurate
    DDouble x = 0.25;
    while ((x *= 0.947) > 1e-290) {
        CMP_UNARY(exp, x, 1.0 * ulp);   // Test exp(x)
        CMP_UNARY(exp, -x, 1.0 * ulp);  // Test exp(-x)
    }
    
    // Large values: still 1 ULP target
    x = 0.125;
    while ((x *= 1.0041) < 708.0) {
        CMP_UNARY(exp, x, 1.0 * ulp);
        if (x < 670)
            CMP_UNARY(exp, -x, 1.0 * ulp);
    }
}
```

**Pattern**: Geometric progression through input space ensures comprehensive coverage without hardcoded test vectors.

### Migration to Rust

#### 1. MPFR Bindings for Rust

**Option A**: Use existing crate: [`rug`](https://crates.io/crates/rug)
```rust
use rug::{Float, ops::Pow};

// High-precision reference (120 bits)
let ref_val = Float::with_val(120, x).exp();
```

**Option B**: Use [`rust-mpfr`](https://crates.io/crates/rust-mpfr)
```rust
use mpfr::Mpfr;

let mut ref_val = Mpfr::new(120);
ref_val.assign(x);
ref_val.exp_mut();
```

**Recommendation**: **Option A (`rug`)** - more ergonomic, actively maintained, used in production Rust code.

#### 2. Test Macro Translation

**C++ Macro**:
```cpp
CMP_UNARY(exp, 1.0, 1e-32);
```

**Rust Equivalent**:
```rust
fn cmp_unary<F>(op_dd: F, op_mpf: fn(Float) -> Float, 
                x: f64, tol: f64) 
where F: Fn(DDouble) -> DDouble
{
    let result_dd = op_dd(DDouble::from(x));
    let result_mpf = op_mpf(Float::with_val(120, x));
    
    let rel_error = relative_error(result_dd, result_mpf);
    assert!(rel_error < tol, 
            "exp({}) failed: rel_error = {} > {}", x, rel_error, tol);
}

// Usage:
cmp_unary(|x| x.exp(), |x| x.exp(), 1.0, 1e-32);
```

#### 3. Rust Test Structure

**Proposed Organization**:
```
xprec/tests/
├── mpfloat.rs          # MPFloat wrapper (using rug)
├── compare.rs          # Comparison utilities and macros
├── arith.rs            # Port of arith.cpp
├── circular.rs         # Port of circular.cpp
├── convert.rs          # Port of convert.cpp
├── exp.rs              # Port of exp.cpp
├── gauss.rs            # Port of gauss.cpp
├── hyperbolic.rs       # Port of hyperbolic.cpp
├── inline.rs           # Port of inline.cpp
├── limits.rs           # Port of limits.cpp
├── random.rs           # Port of random.cpp
├── round.rs            # Port of round.cpp
└── sqrt.rs             # Port of sqrt.cpp
```

#### 4. Porting Checklist for Each Test File

For **each** `libxprec/test/*.cpp` file:

1. **Count test cases**: `grep -c "TEST_CASE" <file.cpp>`
2. **Identify tolerance values**: Extract all `ulp` multipliers
3. **Port test structure**: Translate C++ `TEST_CASE` to Rust `#[test]`
4. **Verify test count**: Ensure Rust file has same number of test functions
5. **Run diff**: Compare outputs for identical inputs
6. **Document mapping**: Add comment linking to C++ source

**Example Documentation**:
```rust
/// Port of libxprec/test/exp.cpp:31-54
/// Tests exponential function across wide input range
/// Tolerance: 1 ULP (2.465e-32) for x ∈ [1e-290, 708]
#[test]
fn test_exp() {
    // ... test implementation
}
```

#### 5. Continuous Verification

**Strategy**:
- **CI Integration**: Run MPFR comparison tests on every PR
- **Benchmark Tracking**: Monitor performance vs C++ libxprec
- **Coverage Reports**: Ensure 100% of C++ tests are ported
- **Fuzz Testing**: Random inputs beyond systematic tests (using `cargo fuzz`)

**Acceptance Criteria**:
- All tests pass with identical or tighter tolerances than C++
- No test cases omitted from C++ suite
- Documentation links every test to its C++ source
- CI fails if any tolerance is relaxed without justification

### Reference Data for Development

**During Development**: 
- Use `rug` crate to generate on-the-fly reference values
- No static test vectors needed (tests are generative)
- Match libxprec's dynamic testing approach

**Optional**: Generate static test vectors for embedded/no-std environments:
```rust
// Generate CSV file with (input, output) pairs at 120-bit precision
// For environments without MPFR
```

**Rationale**: libxprec doesn't use static test data - neither should we. This ensures tests remain maintainable and comprehensive.

## References

1. M. Joldeș, et al., "Tight and rigorous error bounds for basic building blocks of double-word arithmetic," ACM Trans. Math. Softw. 44, 1-27 (2018)
2. J.-M. Muller and L. Rideau, "Accurate and efficient computation of even and odd functions," ACM Trans. Math. Softw. 48, 1, 9 (2022)
3. libxprec repository: https://github.com/tuwien-cms/libxprec
4. twofloat crate: https://crates.io/crates/twofloat

## Next Steps

1. **Create xprec crate**: `cargo new --lib xprec`
2. **Set up project structure**: Create module hierarchy as per architecture plan
3. **Implement core types**: `DDouble`, `ExDouble`, `PowerOfTwo`
4. **Implement basic algorithms**: `two_sum`, `fast_two_sum`, `two_prod`
5. **Set up testing infrastructure**: Unit tests, property-based tests, benchmarks
6. **Implement num-traits**: Early integration for consistent API

---

**Document Version**: 4.2
**Changes**: 
- Focused on Option B (Fresh Implementation)
- Added comprehensive num-traits integration section
- Removed timeline sections (phases and week-by-week roadmap)
- Streamlined to core design and architecture
- **Added Critical Implementation Rules** (mandatory requirements):
  - Algorithm fidelity: Must match libxprec exactly
  - Test coverage: All tests must be ported without omission
  - Precision requirements: No relaxing of error tolerances
  - Verification strategy and review process
- **Updated libxprec reference**: Updated to v0.8-24-g296235c (2025-10-01)
  - Intel compiler compatibility improvements
  - CI enhancements for Intel C++ compiler
  - Pragmas fixes for better cross-compiler support
- **Added comprehensive Testing Strategy section**:
  - Detailed analysis of libxprec's MPFR-based testing approach
  - MPFloat reference implementation (120-bit precision)
  - Complete tolerance specifications by operation
  - Test coverage inventory (14 test files)
  - Migration strategy for Rust using `rug` crate
  - Porting checklist and verification procedures
**Last Updated**: 2025-10-17
**Status**: Ready for implementation

