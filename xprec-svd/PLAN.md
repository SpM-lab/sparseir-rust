# xprec-svd: High-Precision Truncated SVD Implementation

## Overview

`xprec-svd` is a high-precision truncated SVD (TSVD) library implemented in Rust, based on the algorithms used in `libsparseir` and `Eigen3`. It provides extended precision arithmetic support and optimized algorithms for numerical stability.

## Key Features

- **High-precision arithmetic**: Support for `f64`, `f128`, and `TwoFloat` (double-double) precision
- **RRQR preconditioning**: Rank-revealing QR with column pivoting for numerical stability
- **Jacobi SVD**: Two-sided Jacobi iterations for maximum accuracy
- **Truncated SVD**: Efficient computation of only the significant singular values
- **Generic implementation**: Template-based design for different precision types
- **Rust-native**: No C++ dependencies, pure Rust implementation

## Algorithm Analysis

Based on the C++ implementation in `libsparseir`, the TSVD algorithm follows this structure:

### 1. **RRQR (Rank-Revealing QR with Column Pivoting)**

```rust
// Core RRQR algorithm
pub fn rrqr<T: Float>(matrix: &mut Array2<T>, rtol: T) -> (QRPivoted<T>, usize) {
    // 1. Initialize column norms and pivot indices
    // 2. For each column i:
    //    a. Find column with maximum norm (pivoting)
    //    b. Apply Householder reflection
    //    c. Update remaining column norms
    //    d. Check rank condition: |A(i,i)| < rtol * |A(0,0)|
    // 3. Return QR factorization and effective rank
}
```

**Key components:**
- **Householder reflections**: `reflector()` and `reflector_apply()`
- **Column pivoting**: Select column with maximum norm
- **Rank detection**: Stop when diagonal element becomes small relative to tolerance
- **Norm updates**: Efficiently update column norms without full recomputation

### 2. **QR Truncation**

```rust
// Truncate QR result to effective rank
pub fn truncate_qr_result<T: Float>(qr: &QRPivoted<T>, k: usize) -> (Array2<T>, Array2<T>) {
    // Extract Q_trunc (m × k) and R_trunc (k × n) matrices
    // where k is the effective rank from RRQR
}
```

### 3. **Jacobi SVD on R^T**

```rust
// Apply Jacobi SVD to R^T (transpose of R)
pub fn jacobi_svd<T: Float>(matrix: &Array2<T>) -> (Array2<T>, Array1<T>, Array2<T>) {
    // Two-sided Jacobi iterations
    // - Compute 2×2 SVD for each pair of columns
    // - Apply rotations to both U and V
    // - Iterate until convergence
}
```

**Eigen3 JacobiSVD features:**
- **Computation options**: `ComputeThinU | ComputeThinV` for efficiency
- **QR preconditioning**: `ColPivHouseholderQRPreconditioner` (default)
- **Convergence criteria**: Based on 2×2 block diagonalization
- **Numerical stability**: Two-sided Jacobi is more reliable than bidiagonalization

### 4. **TSVD Assembly**

```rust
// Complete TSVD algorithm
pub fn tsvd<T: Float>(matrix: &Array2<T>, rtol: T) -> (Array2<T>, Array1<T>, Array2<T>) {
    // 1. Apply RRQR to A
    let (qr, k) = rrqr(matrix, rtol);
    
    // 2. Truncate QR result to rank k
    let (q_trunc, r_trunc) = truncate_qr_result(&qr, k);
    
    // 3. Compute SVD of R^T
    let (u_svd, s, v_svd) = jacobi_svd(&r_trunc.t());
    
    // 4. Reconstruct final U and V
    let u = q_trunc.dot(&v_svd);
    let v = permute_columns(&u_svd, &qr.jpvt);
    
    (u, s, v)
}
```

## Project Structure

```
xprec-svd/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library interface
│   ├── types.rs            # Core type definitions
│   ├── qr/
│   │   ├── mod.rs          # QR decomposition module
│   │   ├── rrqr.rs         # Rank-revealing QR implementation
│   │   ├── householder.rs  # Householder reflection utilities
│   │   └── truncate.rs     # QR truncation functions
│   ├── svd/
│   │   ├── mod.rs          # SVD decomposition module
│   │   ├── jacobi.rs       # Jacobi SVD implementation
│   │   ├── twobytwo.rs     # 2×2 SVD subroutines
│   │   └── convergence.rs  # Convergence criteria
│   ├── tsvd.rs             # Main TSVD algorithm
│   ├── precision/
│   │   ├── mod.rs          # Precision type definitions
│   │   ├── f64.rs          # f64 implementation
│   │   ├── f128.rs         # f128 implementation (if available)
│   │   └── twofloat.rs     # TwoFloat implementation
│   └── utils/
│       ├── mod.rs          # Utility functions
│       ├── norms.rs        # Vector/matrix norm computations
│       ├── pivoting.rs     # Column pivoting utilities
│       └── validation.rs   # Result validation
└── tests/
    ├── integration_tests.rs
    ├── qr_tests.rs
    ├── svd_tests.rs
    └── precision_tests.rs
```

## Core Types

```rust
// Precision trait for generic implementation
pub trait Precision: Float + From<f64> + Into<f64> {
    const EPSILON: Self;
    const MIN_POSITIVE: Self;
    const MAX_VALUE: Self;
}

// QR factorization result
pub struct QRPivoted<T: Precision> {
    pub factors: Array2<T>,      // Q and R stored in packed format
    pub taus: Array1<T>,         // Householder reflection coefficients
    pub jpvt: Array1<usize>,     // Column pivot indices
}

// SVD result
pub struct SVDResult<T: Precision> {
    pub u: Array2<T>,            // Left singular vectors
    pub s: Array1<T>,            // Singular values
    pub v: Array2<T>,            // Right singular vectors
    pub rank: usize,             // Effective rank
}

// TSVD configuration
pub struct TSVDConfig<T: Precision> {
    pub rtol: T,                 // Relative tolerance
    pub max_iterations: usize,   // Maximum Jacobi iterations
    pub convergence_threshold: T, // Convergence criterion
}
```

## Dependencies

```toml
[dependencies]
# Core linear algebra
ndarray = "0.15"              # Multi-dimensional arrays
nalgebra = "0.32"             # Linear algebra operations

# Extended precision
twofloat = "0.2"              # Double-double arithmetic
num-traits = "0.2"            # Numeric traits

# Optional high precision
# quad = "0.1"                # f128 support (if available)

# Utilities
thiserror = "1.0"             # Error handling
serde = { version = "1.0", features = ["derive"] }  # Serialization
```

## Implementation Phases

### Phase 1: Core Infrastructure
1. **Precision types**: Implement `Precision` trait for `f64`, `TwoFloat`
2. **Basic QR**: Simple QR decomposition without pivoting
3. **Householder reflections**: Core reflection utilities
4. **Basic SVD**: Simple 2×2 SVD implementation

### Phase 2: RRQR Implementation
1. **Column pivoting**: Implement pivot selection algorithm
2. **Norm updates**: Efficient column norm maintenance
3. **Rank detection**: Tolerance-based rank determination
4. **QR truncation**: Extract truncated Q and R matrices

### Phase 3: Jacobi SVD
1. **2×2 SVD**: Core 2×2 singular value decomposition
2. **Rotation matrices**: Givens rotation computations
3. **Convergence**: Iterative convergence criteria
4. **Full Jacobi**: Complete two-sided Jacobi algorithm

### Phase 4: TSVD Integration
1. **Algorithm assembly**: Combine RRQR + Jacobi SVD
2. **Precision handling**: Generic implementation across precision types
3. **Performance optimization**: SIMD and parallelization
4. **Validation**: Comprehensive test suite

### Phase 5: Advanced Features
1. **f128 support**: Quadruple precision (if available)
2. **Parallelization**: Multi-threaded implementations
3. **Memory optimization**: In-place operations where possible
4. **Documentation**: Complete API documentation

## Performance Considerations

### Memory Layout
- **Column-major storage**: Match BLAS conventions
- **In-place operations**: Minimize memory allocation
- **Cache-friendly access**: Optimize for modern CPU caches

### Numerical Stability
- **RRQR preconditioning**: Essential for ill-conditioned matrices
- **Extended precision**: Use higher precision for intermediate calculations
- **Careful norm updates**: Avoid catastrophic cancellation

### Algorithmic Optimizations
- **Early termination**: Stop when convergence is achieved
- **Adaptive precision**: Use higher precision only when needed
- **Vectorized operations**: Leverage SIMD instructions

## Testing Strategy

### Unit Tests
- **QR decomposition**: Test against known matrices
- **SVD accuracy**: Compare with reference implementations
- **Precision handling**: Verify extended precision correctness

### Integration Tests
- **TSVD pipeline**: End-to-end algorithm testing
- **Performance benchmarks**: Compare with Eigen3 and LAPACK
- **Numerical stability**: Test with ill-conditioned matrices

### Reference Comparisons
- **Eigen3 JacobiSVD**: Direct comparison with C++ implementation
- **LAPACK GESVD**: Compare with industry standard
- **Julia LinearAlgebra**: Cross-validation with Julia implementation

## API Design

```rust
// Main TSVD function
pub fn tsvd<T: Precision>(
    matrix: &Array2<T>,
    config: TSVDConfig<T>
) -> Result<SVDResult<T>, TSVDError> {
    // Implementation
}

// Convenience functions
pub fn tsvd_f64(matrix: &Array2<f64>, rtol: f64) -> Result<SVDResult<f64>, TSVDError> {
    tsvd(matrix, TSVDConfig::new(rtol))
}

pub fn tsvd_twofloat(matrix: &Array2<TwoFloat>, rtol: TwoFloat) -> Result<SVDResult<TwoFloat>, TSVDError> {
    tsvd(matrix, TSVDConfig::new(rtol))
}

// Low-level access
pub mod qr {
    pub fn rrqr<T: Precision>(matrix: &mut Array2<T>, rtol: T) -> (QRPivoted<T>, usize);
    pub fn truncate_qr<T: Precision>(qr: &QRPivoted<T>, k: usize) -> (Array2<T>, Array2<T>);
}

pub mod svd {
    pub fn jacobi<T: Precision>(matrix: &Array2<T>) -> SVDResult<T>;
    pub fn twobytwo<T: Precision>(a: T, b: T, c: T, d: T) -> (T, T, T, T, T, T);
}
```

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum TSVDError {
    #[error("Matrix is empty")]
    EmptyMatrix,
    
    #[error("Invalid tolerance: {0}")]
    InvalidTolerance(String),
    
    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },
    
    #[error("Numerical error: {message}")]
    NumericalError { message: String },
    
    #[error("Precision overflow")]
    PrecisionOverflow,
}
```

## Future Extensions

1. **BDCSVD**: Bidiagonal divide-and-conquer SVD for large matrices
2. **Randomized SVD**: Probabilistic algorithms for very large matrices
3. **GPU acceleration**: CUDA/OpenCL implementations
4. **Distributed computing**: Multi-node parallel SVD
5. **Specialized matrices**: Optimizations for sparse, structured matrices

## References

1. **libsparseir**: C++ implementation in `libsparseir/include/sparseir/impl/linalg_impl.ipp`
2. **Eigen3**: JacobiSVD implementation in `eigen/Eigen/src/SVD/JacobiSVD.h`
3. **Julia LinearAlgebra**: Reference implementation in Julia's standard library
4. **Golub & Van Loan**: "Matrix Computations" - theoretical foundation
5. **Higham**: "Accuracy and Stability of Numerical Algorithms" - numerical analysis

This implementation will provide a robust, high-precision TSVD library that can serve as both a standalone tool and a foundation for the `sparseir-rust` project.
