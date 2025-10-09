# xprec-svd

A high-precision truncated SVD (TSVD) library implemented in Rust, based on algorithms from libsparseir and Eigen3.

## Features

- **High-precision arithmetic**: Support for `f64` and `TwoFloatPrecision` (double-double) precision
- **RRQR preconditioning**: Rank-revealing QR with column pivoting for numerical stability
- **Jacobi SVD**: Two-sided Jacobi iterations for maximum accuracy
- **Truncated SVD**: Efficient computation of only the significant singular values
- **Generic implementation**: Trait-based design for different precision types
- **Rust-native**: No C++ dependencies, pure Rust implementation
- **Modular design**: Separate modules for QR decomposition, SVD, and utilities

## Quick Start

```rust
use xprec_svd::*;
use ndarray::array;

// Create a test matrix
let a = array![
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
];

// Compute truncated SVD with f64 precision
let result = tsvd_f64(&a, 1e-12).unwrap();

println!("Rank: {}", result.rank);
println!("Singular values: {:?}", result.s);

// For higher precision, use TwoFloat
let result_hp = tsvd_twofloat_from_f64(&a, 1e-15).unwrap();
println!("High-precision rank: {}", result_hp.rank);
```

## Algorithm

The TSVD algorithm follows this structure:

1. **RRQR (Rank-Revealing QR)**: Apply QR factorization with column pivoting to determine numerical rank
2. **QR Truncation**: Extract truncated Q and R matrices
3. **Jacobi SVD**: Compute SVD of R^T using two-sided Jacobi iterations
4. **Reconstruction**: Assemble final U and V matrices

## API

### Basic Usage

```rust
// f64 precision
let result = tsvd_f64(matrix, rtol)?;

// TwoFloat precision (direct input)
let matrix_tf = Array2::from_shape_fn(matrix.dim(), |(i, j)| {
    TwoFloatPrecision::from_f64(matrix[[i, j]])
});
let rtol_tf = TwoFloatPrecision::from_f64(rtol);
let result = tsvd_twofloat(&matrix_tf, rtol_tf)?;

// TwoFloat precision (from f64) - most convenient
let result = tsvd_twofloat_from_f64(matrix, rtol)?;

// Generic precision with configuration
let config = TSVDConfig::new(rtol);
let result = tsvd(matrix, config)?;
```

### Advanced Configuration

```rust
let config = TSVDConfig {
    rtol: 1e-12,
    max_iterations: 30,
    convergence_threshold: 1e-15,
};

let result = tsvd(matrix, config)?;
```

### Available Functions

- `tsvd_f64()`: Standard f64 precision
- `tsvd_twofloat()`: Direct TwoFloat precision input
- `tsvd_twofloat_from_f64()`: Convert f64 to TwoFloat automatically
- `tsvd()`: Generic function with custom configuration

## Dependencies

- `ndarray`: Multi-dimensional arrays
- `nalgebra`: Linear algebra operations
- `twofloat`: Double-double precision arithmetic
- `num-traits`: Numeric traits
- `approx`: Approximate equality comparisons
- `thiserror`: Error handling

## Modules

- `precision`: Precision trait and TwoFloatPrecision wrapper
- `qr`: Rank-revealing QR decomposition with column pivoting
- `svd`: Jacobi SVD implementation
- `tsvd`: Main truncated SVD interface
- `utils`: Utility functions for norms and matrix operations

## Error Handling

The library uses `TSVDError` enum for error handling:

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

## License

MIT
