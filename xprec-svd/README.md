# xprec-svd

A high-precision truncated SVD (TSVD) library implemented in Rust, based on algorithms from libsparseir and Eigen3.

## Features

- **High-precision arithmetic**: Support for `f64`, `f128`, and `TwoFloat` (double-double) precision
- **RRQR preconditioning**: Rank-revealing QR with column pivoting for numerical stability
- **Jacobi SVD**: Two-sided Jacobi iterations for maximum accuracy
- **Truncated SVD**: Efficient computation of only the significant singular values
- **Generic implementation**: Template-based design for different precision types
- **Rust-native**: No C++ dependencies, pure Rust implementation

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

// Compute truncated SVD
let result = tsvd_f64(&a, 1e-12).unwrap();

println!("Rank: {}", result.rank);
println!("Singular values: {:?}", result.s);
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

// TwoFloat precision
let result = tsvd_twofloat(matrix, rtol)?;

// Generic precision
let result = tsvd(matrix, TSVDConfig::new(rtol))?;
```

### Advanced Configuration

```rust
let config = TSVDConfig {
    rtol: 1e-12,
    max_iterations: 50,
    convergence_threshold: 1e-15,
};

let result = tsvd(matrix, config)?;
```

## Dependencies

- `ndarray`: Multi-dimensional arrays
- `nalgebra`: Linear algebra operations
- `twofloat`: Double-double precision arithmetic
- `num-traits`: Numeric traits

## License

MIT
