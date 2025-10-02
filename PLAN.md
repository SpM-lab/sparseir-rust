# SparseIR Rust Reimplementation Plan

**In English**

## Goals

Reimplement sparseir in Rust while keeping the C-API defined in `libsparseir/include/sparseir/sparseir.h` unchanged.

## Current State Analysis

### Existing C++ Library Structure

#### Major Components
1. **Kernel** (`kernel.hpp`)
   - `AbstractKernel`: Base class
   - `LogisticKernel`: For fermionic analytical continuation
   - `RegularizedBoseKernel`: For bosonic analytical continuation
   - Integral kernels with cutoff parameter Λ
   - Weight function definitions (statistics-dependent)

2. **SVE (Singular Value Expansion)** (`sve.hpp`)
   - `SVEResult`: Singular value expansion result
   - `SamplingSVE`: Sampling-based SVE
   - `CentrosymmSVE`: For centrosymmetric kernels
   - Precision control (ε, cutoff values)
   - Working precision selection (double, double-double)

3. **Basis** (`basis.hpp`)
   - `AbstractBasis<S>`: Statistics-dependent base class
   - `FiniteTempBasis<S>`: Finite temperature IR basis
   - Statistics (Fermionic/Bosonic)
   - Inverse temperature β, frequency cutoff ωmax
   - Basis function management (u, v, uhat)

4. **Functions** (`funcs.hpp`)
   - `AbstractTauFunctions<S>`: Periodic function base class
   - `PeriodicFunctions<S, ImplType>`: Statistics-dependent periodic functions
   - `PiecewiseLegendrePolyVector`: Piecewise Legendre polynomials
   - Imaginary time domain, real frequency domain, Matsubara frequency domain
   - Batch evaluation support

5. **Sampling** (`sampling.hpp`)
   - `AbstractSampling`: Sampling base class
   - `TauSampling`: τ sampling
   - `MatsubaraSampling`: Matsubara sampling
   - Evaluation and fitting (multi-dimensional array support)

6. **DLR (Discrete Lehmann Representation)** (`dlr.hpp`)
   - `MatsubaraPoles<S>`: Matsubara pole management
   - `DiscreteLehmannRepresentation<S>`: DLR basis
   - IR-DLR transformations
   - Poles on the real frequency axis

#### Dependency Analysis

**Eigen3 Usage** (29 header files)
- Linear algebra operations: matrix and vector operations
- SVD computation: `JacobiSVD`
- Tensor operations: `Eigen::Tensor`
- Complex number operations: `Eigen::MatrixXcd`

**libxprec Usage** (37 locations)
- Extended precision arithmetic: `xprec::DDouble`
- High-precision numerical integration
- Computations requiring numerical stability

**BLAS** (Required)
- High-performance matrix operations
- Essential for optimal performance
- ILP64 support (for large matrices)

### C-API Characteristics

#### Opaque Types
```c
struct _spir_kernel;
typedef struct _spir_kernel spir_kernel;
// Similarly: spir_funcs, spir_basis, spir_sampling, spir_sve_result
```

#### Major API Categories
1. **Kernel Creation and Operations**
   - `spir_logistic_kernel_new()`
   - `spir_reg_bose_kernel_new()`
   - `spir_kernel_domain()`

2. **SVE Computation**
   - `spir_sve_result_new()`
   - `spir_sve_result_get_size()`
   - `spir_sve_result_get_svals()`

3. **Basis Creation and Operations**
   - `spir_basis_new()`
   - `spir_basis_get_u()`, `spir_basis_get_v()`, `spir_basis_get_uhat()`
   - Sampling point retrieval functions

4. **Function Evaluation**
   - `spir_funcs_eval()`
   - `spir_funcs_batch_eval()`
   - `spir_funcs_eval_matsu()`

5. **Sampling**
   - `spir_tau_sampling_new()`
   - `spir_matsu_sampling_new()`
   - `spir_sampling_eval_dd()`, `spir_sampling_fit_dd()`

6. **DLR Transformations**
   - `spir_dlr_new()`
   - `spir_ir2dlr_dd()`, `spir_dlr2ir_dd()`

7. **BLAS Function Registration**
   - `spir_register_blas_functions()`: Register custom BLAS kernels
   - `spir_register_ilp64_functions()`: Register ILP64 BLAS functions
   - `spir_clear_blas_functions()`: Reset to default BLAS

## Rust Reimplementation Strategy

### Phase 1: Foundation Building

#### 1.1 Project Structure (Updated)
**Note**: The project is organized from foundational components to higher-level interfaces
```
sparseir-rust/
├── sparseir-poly/          # Piecewise polynomial representations (foundation)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── piecewise.rs    # Basic piecewise polynomial
│       ├── legendre.rs     # Legendre polynomial implementation
│       ├── vector.rs       # PiecewiseLegendrePolyVector
│       ├── fourier.rs      # Fourier transform for polynomials
│       ├── specfuncs.rs    # Special functions (Bessel, Gamma, Legendre)
│       ├── gauss.rs        # Gaussian integration
│       └── traits.rs       # Polynomial trait definitions
├── sparseir-svd/           # SVD specialized functionality (independent)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── sve.rs          # Singular value expansion
│       ├── tsvd.rs         # High-precision TSVD
│       ├── kernel_svd.rs   # Kernel SVD
│       └── precision.rs    # twofloat integration
├── sparseir-rust/          # Rust interface for SparseIR functionality
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── linalg.rs    # Linear algebra wrappers
│       ├── kernel.rs       # Kernel implementation
│       ├── basis.rs        # Basis implementation
│       ├── funcs.rs        # Function implementation
│       ├── sampling.rs     # Sampling implementation
│       ├── dlr.rs          # DLR implementation
│       └── traits.rs       # Common trait definitions
├── sparseir-capi/          # C-API layer and integration crate
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs          # Main integration point
│       ├── ffi.rs          # FFI functions
│       └── types.rs        # C-API type definitions
│   └── tests/
│       ├── integration_tests.rs  # Integration tests
│       └── c_api_tests.rs        # C-API compatibility tests
└── examples/
    ├── basic_usage.rs
    └── c_api_usage.rs
```

#### 1.2 Dependencies and Alternatives

**Linear Algebra Libraries**
```toml
[dependencies]
# Main linear algebra libraries
ndarray = "0.15"          # Multi-dimensional arrays (Eigen::MatrixXd alternative)
nalgebra = "0.32"         # Linear algebra (Eigen::VectorXd alternative)
# Performance: Equivalent to or better than Eigen3

# Numerical computation
num-complex = "0.4"       # Complex number operations
num-traits = "0.2"        # Numerical traits

# Extended precision arithmetic
twofloat = "0.4"           # double-double precision (libxprec alternative)
# - Achieves high precision as sum of two f64 values
# - Equivalent functionality to libxprec::DDouble

# Special functions
special = "0.2"           # Bessel functions, gamma functions, etc.
statrs = "0.16"           # Statistical functions

# Parallel processing (for future TSVD optimization)
# rayon = "1.7"             # Data parallel processing

# FFI (C-API compatibility)
libc = "0.2"              # C standard library

# BLAS (Required for performance)
blas = "0.22"             # BLAS bindings (default)
# Custom BLAS kernels and ILP64 support via C-API function pointer registration
```

**Crate Dependencies**

**sparseir-poly** (Polynomial representations - foundation)
- `special` (Bessel, Gamma functions)
- Custom Legendre polynomial implementation
- Gaussian integration

**sparseir-svd** (SVD functionality)
- High-precision TSVD implementation
- Independent mathematical functionality

**sparseir-rust** (Rust interface for SparseIR functionality)
- `sparseir-poly` (internal)
- `sparseir-svd` (internal)
- `ndarray`, `nalgebra` (linear algebra)
- `twofloat` (extended precision)
- `special`, `statrs` (special functions)

**sparseir-capi** (C-API layer and integration)
- `sparseir-rust` (internal)
- `sparseir-svd` (internal)
- `sparseir-poly` (internal)
- `libc` (FFI)
- Main integration point for all SparseIR functionality

**Eigen3 Alternative Details**
- **ndarray**: Multi-dimensional arrays, tensor operations
- **nalgebra**: Linear algebra, SVD computation
- **BLAS integration**: Default `blas` crate with optional custom kernels
- **Custom BLAS support**: C-API function pointer registration for ILP64/custom kernels
- **Performance**: Equivalent to or better than Eigen3
- **Memory efficiency**: Optimization through Rust's ownership system

**libxprec Alternative**
- **twofloat crate**: Uses `TwoFloat` type
- **double-double precision**: Achieves high precision as sum of two f64 values
- **Numerical stability**: For high-precision numerical integration

**Piecewise Polynomial Representation (sparseir-poly)**
- **Independent crate**: Self-contained polynomial functionality
- **Legendre polynomials**: Custom implementation (no existing Rust crates)
- **Vector operations**: Efficient piecewise polynomial vector operations
- **Fourier transforms**: Polynomial Fourier representations
- **Special functions**: Bessel, Gamma functions (using `special` crate)
- **Integration**: Gaussian integration for piecewise polynomials
- **Advantages**: Uses existing implemented crate, no maintenance required

### Phase 2: Core Implementation

#### 2.1 Mathematical Foundation
- **Linear algebra**: ndarray/nalgebra as Eigen3 alternative
- **Extended precision**: libxprec alternative implementation
- **Special functions**: Utilizing existing Rust crates
- **Numerical integration**: Gaussian integration, Legendre polynomials

#### 2.2 Major Data Structure Design

**Statistics Type Definition**
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Statistics {
    Fermionic,
    Bosonic,
}

// Statistics distinction at type level
pub trait StatisticsType {
    const STATISTICS: Statistics;
}

pub struct Fermionic;
impl StatisticsType for Fermionic {
    const STATISTICS: Statistics = Statistics::Fermionic;
}

pub struct Bosonic;
impl StatisticsType for Bosonic {
    const STATISTICS: Statistics = Statistics::Bosonic;
}
```

**Kernel Implementation**
```rust
// Abstract kernel trait
pub trait AbstractKernel {
    fn compute(&self, x: f64, y: f64) -> f64;
    fn compute_ddouble(&self, x: TwoFloat, y: TwoFloat) -> TwoFloat;
    fn xrange(&self) -> (f64, f64);
    fn yrange(&self) -> (f64, f64);
    fn lambda(&self) -> f64;
}

// Logistic kernel
pub struct LogisticKernel {
    lambda: f64,
}

impl AbstractKernel for LogisticKernel {
    fn compute(&self, x: f64, y: f64) -> f64 {
        // K(x, y) = exp(-Λy(x + 1)/2)/(1 + exp(-Λy))
        let lambda = self.lambda;
        let exp_term = (-lambda * y * (x + 1.0) / 2.0).exp();
        exp_term / (1.0 + (-lambda * y).exp())
    }
    
    fn compute_ddouble(&self, x: TwoFloat, y: TwoFloat) -> TwoFloat {
        // High-precision implementation
        let lambda = TwoFloat::from(self.lambda);
        let exp_term = (-lambda * y * (x + 1.0) / 2.0).exp();
        exp_term / (1.0 + (-lambda * y).exp())
    }
    
    fn xrange(&self) -> (f64, f64) { (-1.0, 1.0) }
    fn yrange(&self) -> (f64, f64) { (-1.0, 1.0) }
    fn lambda(&self) -> f64 { self.lambda }
}

// Regularized bosonic kernel
pub struct RegularizedBoseKernel {
    lambda: f64,
}

impl AbstractKernel for RegularizedBoseKernel {
    fn compute(&self, x: f64, y: f64) -> f64 {
        // K(x, y) = y * exp(-Λy(x + 1)/2)/(exp(-Λy) - 1)
        let lambda = self.lambda;
        let exp_term = (-lambda * y * (x + 1.0) / 2.0).exp();
        y * exp_term / ((-lambda * y).exp() - 1.0)
    }
    // ... other method implementations
}
```

**SVE Implementation with Kernel Specialization**
```rust
pub struct SVEResult {
    pub u: PiecewiseLegendrePolyVector,
    pub s: Array1<f64>,
    pub v: PiecewiseLegendrePolyVector,
    pub epsilon: f64,
}

// Kernel-specialized SVE trait
pub trait AbstractSVE<K: AbstractKernel> {
    fn compute(&self) -> Result<SVEResult, SpirError>;
}

// Logistic kernel specialized SVE
pub struct LogisticSamplingSVE {
    kernel: LogisticKernel,  // Concrete type, not trait object
    epsilon: f64,
    n_gauss: i32,
    // ... other fields
}

impl AbstractSVE<LogisticKernel> for LogisticSamplingSVE {
    fn compute(&self) -> Result<SVEResult, SpirError> {
        // SVE computation optimized for logistic kernel
    }
}

// Regularized bosonic kernel specialized SVE
pub struct RegularizedBoseSamplingSVE {
    kernel: RegularizedBoseKernel,  // Concrete type
    epsilon: f64,
    n_gauss: i32,
    // ... other fields
}

impl AbstractSVE<RegularizedBoseKernel> for RegularizedBoseSamplingSVE {
    fn compute(&self) -> Result<SVEResult, SpirError> {
        // SVE computation optimized for regularized bosonic kernel
    }
}

// Centrosymmetric kernel specialized SVE
pub struct CentrosymmSVE<K: AbstractKernel> {
    kernel: K,  // Generic but concrete type
    epsilon: f64,
    n_gauss: i32,
}

impl<K: AbstractKernel> AbstractSVE<K> for CentrosymmSVE<K> {
    fn compute(&self) -> Result<SVEResult, SpirError> {
        // Optimized SVE computation for centrosymmetric kernels
    }
}
```

**Basis Implementation with Kernel Specialization**
```rust
// Kernel-specialized basis trait
pub trait AbstractBasis<K: AbstractKernel, S: StatisticsType> {
    fn get_beta(&self) -> f64;
    fn get_accuracy(&self) -> f64;
    fn get_wmax(&self) -> f64;
    fn size(&self) -> usize;
    fn significance(&self) -> Array1<f64>;
    fn kernel_type(&self) -> &'static str;
}

// Logistic kernel specialized basis
pub struct LogisticFiniteTempBasis<S: StatisticsType> {
    pub beta: f64,
    pub lambda: f64,
    pub sve_result: SVEResult,
    pub u: PeriodicFunctions<S, PiecewiseLegendrePolyVector>,
    pub v: PiecewiseLegendrePolyVector,
    pub uhat: PiecewiseLegendreFTVector<S>,
    // No dynamic kernel storage needed
}

impl<S: StatisticsType> AbstractBasis<LogisticKernel, S> for LogisticFiniteTempBasis<S> {
    fn get_beta(&self) -> f64 { self.beta }
    fn get_accuracy(&self) -> f64 { self.sve_result.epsilon }
    fn get_wmax(&self) -> f64 { self.lambda / self.beta }
    fn size(&self) -> usize { self.u.size() }
    fn significance(&self) -> Array1<f64> { self.sve_result.s.clone() }
    fn kernel_type(&self) -> &'static str { "LogisticKernel" }
}

// Regularized bosonic kernel specialized basis
pub struct RegularizedBoseFiniteTempBasis<S: StatisticsType> {
    pub beta: f64,
    pub lambda: f64,
    pub sve_result: SVEResult,
    pub u: PeriodicFunctions<S, PiecewiseLegendrePolyVector>,
    pub v: PiecewiseLegendrePolyVector,
    pub uhat: PiecewiseLegendreFTVector<S>,
    // No dynamic kernel storage needed
}

impl<S: StatisticsType> AbstractBasis<RegularizedBoseKernel, S> for RegularizedBoseFiniteTempBasis<S> {
    fn get_beta(&self) -> f64 { self.beta }
    fn get_accuracy(&self) -> f64 { self.sve_result.epsilon }
    fn get_wmax(&self) -> f64 { self.lambda / self.beta }
    fn size(&self) -> usize { self.u.size() }
    fn significance(&self) -> Array1<f64> { self.sve_result.s.clone() }
    fn kernel_type(&self) -> &'static str { "RegularizedBoseKernel" }
}

// Type-safe basis construction
impl<S: StatisticsType> LogisticFiniteTempBasis<S> {
    pub fn new(
        beta: f64,
        omega_max: f64,
        epsilon: f64,
        kernel: LogisticKernel,  // Concrete type, not trait object
        sve_result: SVEResult,
        max_size: Option<usize>,
    ) -> Result<Self, SpirError> {
        // Basis construction with compile-time kernel specialization
    }
}
```

**Opaque Type Implementation**
```rust
// Opaque types for C-API
#[repr(C)]
pub struct SpirKernel {
    inner: Box<dyn AbstractKernel>,
}

#[repr(C)]
pub struct SpirSveResult {
    inner: SVEResult,
}

#[repr(C)]
pub struct SpirBasis {
    inner: Box<dyn AbstractBasis>,
}

#[repr(C)]
pub struct SpirFuncs {
    inner: Box<dyn AbstractTauFunctions>,
}

#[repr(C)]
pub struct SpirSampling {
    inner: Box<dyn AbstractSampling>,
}

// Abstract function trait
pub trait AbstractTauFunctions {
    fn eval(&self, x: f64) -> Array1<f64>;
    fn batch_eval(&self, xs: &Array1<f64>) -> Array2<f64>;
    fn size(&self) -> usize;
}

// Abstract sampling trait
pub trait AbstractSampling {
    fn n_sampling_points(&self) -> i32;
    fn basis_size(&self) -> usize;
    fn get_cond_num(&self) -> f64;
}
```

#### 2.3 Error Handling
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpirStatus {
    ComputationSuccess = 0,
    GetImplFailed = -1,
    InvalidDimension = -2,
    InputDimensionMismatch = -3,
    OutputDimensionMismatch = -4,
    NotSupported = -5,
    InvalidArgument = -6,
    InternalError = -7,
}

#[derive(Debug, thiserror::Error)]
pub enum SpirError {
    #[error("Invalid dimension: {0}")]
    InvalidDimension(String),
    
    #[error("Input dimension mismatch: expected {expected}, got {actual}")]
    InputDimensionMismatch { expected: usize, actual: usize },
    
    #[error("Output dimension mismatch: expected {expected}, got {actual}")]
    OutputDimensionMismatch { expected: usize, actual: usize },
    
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    
    #[error("Internal error: {0}")]
    InternalError(String),
    
    #[error("Not supported: {0}")]
    NotSupported(String),
}

impl From<SpirError> for SpirStatus {
    fn from(err: SpirError) -> Self {
        match err {
            SpirError::InvalidDimension(_) => SpirStatus::InvalidDimension,
            SpirError::InputDimensionMismatch { .. } => SpirStatus::InputDimensionMismatch,
            SpirError::OutputDimensionMismatch { .. } => SpirStatus::OutputDimensionMismatch,
            SpirError::InvalidArgument(_) => SpirStatus::InvalidArgument,
            SpirError::InternalError(_) => SpirStatus::InternalError,
            SpirError::NotSupported(_) => SpirStatus::NotSupported,
        }
    }
}
```

### Phase 3: C-API Implementation

#### 3.1 FFI Layer
- Implement C-API with `#[no_mangle]` functions
- Opaque type pointer management
- Memory safety assurance
- Unified error handling

#### 3.2 Memory Management
```rust
// Automatic memory management
impl Drop for SpirKernel {
    fn drop(&mut self) {
        // Resource cleanup
    }
}

// Release function for C-API
#[no_mangle]
pub extern "C" fn spir_kernel_release(kernel: *mut SpirKernel) {
    if !kernel.is_null() {
        unsafe {
            Box::from_raw(kernel);
        }
    }
}
```

### フェーズ4: テスト・検証

#### 4.1 単体テスト
- 各モジュールの個別テスト
- 数値精度の検証
- エッジケースのテスト

#### 4.2 統合テスト
- C-API互換性テスト
- 既存C++実装との結果比較
- パフォーマンステスト

#### 4.3 回帰テスト
- 既存Python/Fortranラッパーの動作確認
- 既存テストスイートの移植

## Implementation Order

### Phase 1: Foundation Building (2-3 weeks)
1. **Project Structure Construction**
   - Cargo.toml configuration
   - Module structure creation
   - Basic type definitions

2. **Mathematical Foundation Implementation**
   - Piecewise polynomial representations (`sparseir-poly`)
     - Special functions (Bessel, Gamma using `special` crate)
     - Custom Legendre polynomial implementation
     - Gaussian integration
   - SVD functionality (`sparseir-svd`) - independent
     - High-precision TSVD implementation

3. **Error Handling**
   - `SpirError` and `SpirStatus` implementation
   - Error conversion logic

### Phase 2: Core Functionality (4-6 weeks)
1. **Rust Interface Implementation** (`sparseir-rust`)
   - Extended precision arithmetic (utilizing `twofloat` crate)
   - Linear algebra wrappers (`linalg.rs`)
   - Kernel, basis, and sampling implementations

2. **Kernel Implementation**
   - `AbstractKernel` trait
   - `LogisticKernel` implementation
   - `RegularizedBoseKernel` implementation

3. **SVE Implementation**
   - `SVEResult` structure
   - `SamplingSVE` implementation
   - Singular value decomposition optimization

4. **Basis Implementation**
   - `FiniteTempBasis` implementation
   - Statistics type distinction
   - Basis function management

5. **Functions Implementation**
   - `PeriodicFunctions` implementation
   - Piecewise Legendre polynomials
   - Batch evaluation functionality

### Phase 3: Advanced Functionality (3-4 weeks)
1. **Sampling Implementation**
   - `TauSampling` implementation
   - `MatsubaraSampling` implementation
   - Multi-dimensional array support

2. **DLR Implementation**
   - `MatsubaraPoles` implementation
   - `DiscreteLehmannRepresentation` implementation
   - IR-DLR transformations

3. **Batch Processing Optimization**
   - Parallel processing introduction
   - Memory efficiency optimization

### Phase 4: C-API Implementation (2-3 weeks)
1. **FFI Layer Implementation**
   - Opaque type definitions
   - C-API function implementation
   - Memory management

2. **Testing Implementation**
   - Integration tests (`sparseir-capi/tests/integration_tests.rs`)
   - C-API compatibility tests (`sparseir-capi/tests/c_api_tests.rs`)
   - Result comparison with existing C++ implementation
   - Performance testing

### Phase 5: Optimization and Validation (2-3 weeks)
1. **Performance Optimization**
   - Profiling
   - Parallel processing optimization
   - Memory usage optimization

2. **Comprehensive Testing**
   - Unit tests
   - Integration tests
   - Regression tests

3. **Documentation**
   - API documentation
   - Usage examples
   - Migration guide

## Technical Challenges and Solutions

### 1. Extended Precision Arithmetic
**Challenge**: Alternative to libxprec's double-double precision
**Solution**:
- Utilize `twofloat` crate (`TwoFloat` type)
- Ensure compatibility with libxprec::DDouble
- Verify numerical stability

### 2. Special Functions
**Challenge**: High-precision implementation of Bessel functions, gamma functions, etc.
**Solution**:
- Utilize `special` crate for Bessel and Gamma functions
- Custom Legendre polynomial implementation (no existing Rust crates)
- Precision control for numerical integration

### 3. Memory Management
**Challenge**: Compatibility and safety with C-API
**Solution**:
- Safe memory management using `Box` and `Rc`
- Appropriate design of opaque types
- Automatic memory deallocation implementation

### 4. Performance
**Challenge**: Achieving performance equivalent to or better than Eigen3
**Solution**:
- Optimization of `ndarray` and `nalgebra`
- Default BLAS integration via `blas` crate
- Custom BLAS kernels via C-API function pointer registration
- ILP64 support for large matrices through external registration
- Future: Parallel processing for TSVD optimization
- Optimization through profiling

### 5. BLAS Function Registration
**Challenge**: Flexible BLAS kernel registration without build dependencies
**Solution**:
- C-API function pointer registration system
- Default `blas` crate for standard operations
- Custom kernel support via external registration
- ILP64 support through Fortran function pointer registration

### 6. Type-Level Specialization
**Challenge**: Avoiding dynamic dispatch overhead and unnecessary runtime structures
**Solution**:
- Kernel-specialized basis and SVE types
- Compile-time type specialization instead of runtime polymorphism
- Concrete types instead of trait objects where possible
- Zero-cost abstractions through monomorphization

### 7. Testing Strategy
**Challenge**: Comprehensive testing of C-API and integration
**Solution**:
- Integration tests in `sparseir-capi/tests/`
- C-API compatibility tests with existing implementations
- Performance benchmarking
- Regression testing against C++ implementation

### 8. Compatibility
**Challenge**: Compatibility with existing Python/Fortran wrappers
**Solution**:
- Complete C-API compatibility maintenance
- Comprehensive regression testing
- Gradual migration strategy

## Success Criteria

1. **Compatibility**: Complete compatibility with existing C-API
2. **Accuracy**: Numerical result consistency
3. **Performance**: Performance equivalent to or better than C++ implementation
4. **Safety**: Memory safety assurance
5. **Maintainability**: Clean code structure
6. **Flexibility**: Support for custom BLAS kernels and ILP64 without build dependencies
7. **Efficiency**: Zero-cost abstractions through type-level specialization

## Next Steps

1. Detailed analysis of existing code
2. Dependency investigation
3. Prototype implementation
4. Gradual implementation start
