//! Matrix multiplication utilities using Faer backend
//!
//! This module provides thin wrappers around `mdarray-linalg-faer` for matrix operations.
//! All functions work with `mdarray::DTensor<T, 2>`.
//!
//! # Design Notes
//! - All matrix operations use Faer's parallel implementations for performance
//! - For future BLAS support, only these wrapper functions need modification
//! - matvec operations are implemented via `reshape + matmul`

use mdarray::DTensor;

/// Parallel matrix multiplication: C = A * B
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
#[cfg(feature = "faer-backend")]
pub fn matmul_par<T>(a: &DTensor<T, 2>, b: &DTensor<T, 2>) -> DTensor<T, 2>
where
    T: num_complex::ComplexFloat + faer_traits::ComplexField + num_traits::One + 'static,
{
    use mdarray_linalg::{MatMul, MatMulBuilder};
    use mdarray_linalg_faer::Faer;

    // Use Faer's parallel matmul
    Faer.matmul(a, b).parallelize().eval()
}

/// Fallback matrix multiplication (non-parallel, for non-faer builds)
///
/// This is a simple O(n³) implementation for testing without Faer.
#[cfg(not(feature = "faer-backend"))]
pub fn matmul_par<T>(a: &DTensor<T, 2>, b: &DTensor<T, 2>) -> DTensor<T, 2>
where
    T: num_complex::ComplexFloat,
{
    let a_shape = *a.shape();
    let b_shape = *b.shape();
    
    assert_eq!(
        a_shape[0], b_shape[0],
        "Matrix dimensions incompatible: ({}, {}) * ({}, {})",
        a_shape[0], a_shape[1], b_shape[0], b_shape[1]
    );

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];
    
    DTensor::from_fn([m, n], |idx| {
        let (i, j) = (idx[0], idx[1]);
        let mut sum = T::zero();
        for p in 0..k {
            sum = sum + a[[i, p]] * b[[p, j]];
        }
        sum
    })
}

/// Matrix transpose: B = Aᵀ
///
/// # Arguments
/// * `a` - Input matrix (M x N)
///
/// # Returns
/// Transposed matrix (N x M)
///
/// # Example
/// ```ignore
/// use mdarray::tensor;
/// use sparseir::gemm::transpose;
///
/// let a = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let b = transpose(&a);
/// // b = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
/// ```
pub fn transpose<T: Clone>(a: &DTensor<T, 2>) -> DTensor<T, 2> {
    let shape = *a.shape();
    let m = shape.0;
    let n = shape.1;
    DTensor::<T, 2>::from_fn([n, m], |idx| a[[idx[1], idx[0]]].clone())
}

/// Solve linear system using SVD pseudoinverse: X = A⁺ * B
///
/// Solves the least-squares problem `min ||AX - B||` using SVD-based pseudoinverse.
/// This is equivalent to `X = pinv(A) * B` where singular values below
/// `rcond * max(S)` are set to zero.
///
/// # Arguments
/// * `a` - Coefficient matrix (M x N)
/// * `b` - Right-hand side matrix (M x K)
/// * `rcond` - Relative condition number threshold for truncating small singular values
///
/// # Returns
/// Solution matrix X (N x K)
///
/// # Panics
/// Panics if A.rows != B.rows
///
/// # Example
/// ```ignore
/// use mdarray::tensor;
/// use sparseir::gemm::solve_via_svd;
///
/// let a = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let b = tensor![[1.0], [2.0], [3.0]];
/// let x = solve_via_svd(&a, &b, 1e-10);
/// ```
pub fn solve_via_svd<T>(a: &DTensor<T, 2>, b: &DTensor<T, 2>, rcond: T) -> DTensor<T, 2>
where
    T: num_complex::ComplexFloat + faer_traits::ComplexField + num_traits::One + xprec_svd::Precision + 'static,
{
    let a_shape = *a.shape();
    let b_shape = *b.shape();
    
    assert_eq!(
        a_shape.0, b_shape.0,
        "Matrix dimensions incompatible: A.rows={} != B.rows={}",
        a_shape.0, b_shape.0
    );

    // Compute SVD: A = U * S * Vᵀ
    let svd_result = xprec_svd::jacobi_svd(a);
    
    let u = &svd_result.u;
    let s = &svd_result.s;
    let v = &svd_result.v;
    
    let n_singular = s.len();
    let max_s = s.iter().cloned().fold(T::zero(), |a, b| xprec_svd::Precision::max(a, b));
    let threshold = rcond * max_s;
    
    // Compute U^T * B
    let ut_b = matmul_par(&transpose(u), b);
    
    // Divide by singular values: S^{-1} * (U^T * B)
    // Only use singular values above threshold
    let ut_b_shape = *ut_b.shape();
    let m = ut_b_shape.0;
    let k = ut_b_shape.1;
    let s_inv_ut_b = DTensor::<T, 2>::from_fn([m, k], |idx| {
        let (i, j) = (idx[0], idx[1]);
        if i < n_singular && s[i] > threshold {
            ut_b[[i, j]] / s[i]
        } else {
            T::zero()
        }
    });
    
    // X = V * (S^{-1} * U^T * B)
    matmul_par(v, &s_inv_ut_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mdarray::tensor;

    #[test]
    fn test_matmul_par_basic() {
        let a: DTensor<f64, 2> = tensor![[1.0, 2.0], [3.0, 4.0]];
        let b: DTensor<f64, 2> = tensor![[5.0, 6.0], [7.0, 8.0]];
        let c = matmul_par(&a, &b);
        
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        assert!((c[[0, 0]] - 19.0).abs() < 1e-10);
        assert!((c[[0, 1]] - 22.0).abs() < 1e-10);
        assert!((c[[1, 0]] - 43.0).abs() < 1e-10);
        assert!((c[[1, 1]] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_par_non_square() {
        let a: DTensor<f64, 2> = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3
        let b: DTensor<f64, 2> = tensor![[7.0], [8.0], [9.0]]; // 3x1
        let c = matmul_par(&a, &b);
        
        // Expected: [[1*7+2*8+3*9], [4*7+5*8+6*9]]
        //         = [[50], [122]]
        assert!((c[[0, 0]] - 50.0).abs() < 1e-10);
        assert!((c[[1, 0]] - 122.0).abs() < 1e-10);
    }

    #[test]
    fn test_transpose_basic() {
        let a: DTensor<f64, 2> = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = transpose(&a);
        
        assert_eq!(*b.shape(), (3, 2));
        assert!((b[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((b[[0, 1]] - 4.0).abs() < 1e-10);
        assert!((b[[1, 0]] - 2.0).abs() < 1e-10);
        assert!((b[[1, 1]] - 5.0).abs() < 1e-10);
        assert!((b[[2, 0]] - 3.0).abs() < 1e-10);
        assert!((b[[2, 1]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_transpose_square() {
        let a: DTensor<f64, 2> = tensor![[1.0, 2.0], [3.0, 4.0]];
        let b = transpose(&a);
        
        assert_eq!(*b.shape(), (2, 2));
        assert!((b[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((b[[0, 1]] - 3.0).abs() < 1e-10);
        assert!((b[[1, 0]] - 2.0).abs() < 1e-10);
        assert!((b[[1, 1]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_via_svd_overdetermined() {
        // Solve Ax = b for overdetermined system
        // A = [[1, 2], [3, 4], [5, 6]]
        // b = [[1], [2], [3]]
        let a: DTensor<f64, 2> = tensor![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let b: DTensor<f64, 2> = tensor![[1.0], [2.0], [3.0]];
        let x = solve_via_svd(&a, &b, 1e-10);
        
        // Verify dimensions
        assert_eq!(*x.shape(), (2, 1));
        
        // Verify Ax ≈ b in least-squares sense
        let ax = matmul_par(&a, &x);
        let residual = (0..3).map(|i| {
            let diff = ax[[i, 0]] - b[[i, 0]];
            diff * diff
        }).sum::<f64>().sqrt();
        
        // Residual should be small (but not zero for overdetermined system)
        assert!(residual < 1e-5, "Residual too large: {}", residual);
    }

    #[test]
    fn test_solve_via_svd_square() {
        // Solve Ax = b for square system
        // A = [[2, 1], [1, 2]]
        // b = [[3], [3]]
        // Solution: x = [[1], [1]]
        let a: DTensor<f64, 2> = tensor![[2.0, 1.0], [1.0, 2.0]];
        let b: DTensor<f64, 2> = tensor![[3.0], [3.0]];
        let x = solve_via_svd(&a, &b, 1e-10);
        
        // Verify dimensions
        assert_eq!(*x.shape(), (2, 1));
        
        // Verify solution
        assert!((x[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((x[[1, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_via_svd_rank_deficient() {
        // Rank-deficient matrix (rank 1)
        // A = [[1, 2], [2, 4], [3, 6]]
        let a: DTensor<f64, 2> = tensor![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
        let b: DTensor<f64, 2> = tensor![[1.0], [2.0], [3.0]];
        let x = solve_via_svd(&a, &b, 1e-6);
        
        // Should find least-squares solution
        assert_eq!(*x.shape(), (2, 1));
        
        // Verify Ax ≈ b
        let ax = matmul_par(&a, &x);
        for i in 0..3 {
            let diff = (ax[[i, 0]] - b[[i, 0]]).abs();
            assert!(diff < 1e-5, "Row {}: diff = {}", i, diff);
        }
    }
}

