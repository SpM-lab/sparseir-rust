//! Matrix multiplication utilities using Faer backend
//!
//! This module provides thin wrappers around `mdarray-linalg-faer` for matrix operations.
//!
//! # Design Notes
//! - Uses Faer's high-performance parallel implementation
//! - For transpose, use mdarray's built-in `.transpose()` method directly

use mdarray::DTensor;

/// Parallel matrix multiplication: C = A * B
///
/// Uses Faer's high-performance parallel implementation.
///
/// # Arguments
/// * `a` - Left matrix (M x K)
/// * `b` - Right matrix (K x N)
///
/// # Returns
/// Result matrix (M x N)
///
/// # Panics
/// Panics if matrix dimensions are incompatible (A.cols != B.rows)
///
/// # Example
/// ```ignore
/// use mdarray::tensor;
/// use sparseir::gemm::matmul_par;
///
/// let a = tensor![[1.0, 2.0], [3.0, 4.0]];
/// let b = tensor![[5.0, 6.0], [7.0, 8.0]];
/// let c = matmul_par(&a, &b);
/// // c = [[19.0, 22.0], [43.0, 50.0]]
/// ```
pub fn matmul_par<T>(a: &DTensor<T, 2>, b: &DTensor<T, 2>) -> DTensor<T, 2>
where
    T: num_complex::ComplexFloat + faer_traits::ComplexField + num_traits::One + 'static,
{
    use mdarray_linalg::{MatMul, MatMulBuilder};
    use mdarray_linalg_faer::Faer;

    // Use Faer's parallel matmul
    Faer.matmul(a, b).parallelize().eval()
}

