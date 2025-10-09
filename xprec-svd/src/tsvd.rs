//! Truncated SVD (TSVD) implementation

use mdarray::Tensor;
use crate::precision::Precision;
use crate::qr::{rrqr, truncate_qr_result};
use crate::svd::jacobi_svd;

/// Configuration for TSVD computation
#[derive(Debug, Clone)]
pub struct TSVDConfig<T: Precision> {
    /// Relative tolerance for rank determination
    pub rtol: T,
    /// Maximum number of Jacobi iterations
    pub max_iterations: usize,
    /// Convergence threshold for Jacobi SVD
    pub convergence_threshold: T,
}

impl<T: Precision> TSVDConfig<T> {
    pub fn new(rtol: T) -> Self {
        Self {
            rtol,
            max_iterations: 30,
            convergence_threshold: Precision::sqrt(<T as Precision>::epsilon()),
        }
    }
}

/// Error types for TSVD computation
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

/// Truncated SVD computation
/// 
/// Computes the truncated SVD using the algorithm:
/// 1. Apply RRQR to A to get QR factorization with rank k
/// 2. Truncate QR result to rank k
/// 3. Compute SVD of R^T (transpose of R)
/// 4. Reconstruct final U and V matrices
/// 
/// # Arguments
/// * `matrix` - Input matrix (m × n)
/// * `config` - TSVD configuration
/// 
/// # Returns
/// * `SVDResult` - Truncated SVD result
pub fn tsvd<T: Precision + 'static + std::fmt::Debug>(
    matrix: &Tensor<T, (usize, usize)>,
    config: TSVDConfig<T>,
) -> Result<crate::svd::jacobi::SVDResult<T>, TSVDError> {
    let shape = *matrix.shape();
    let (m, n) = shape;
    
    if m == 0 || n == 0 {
        return Err(TSVDError::EmptyMatrix);
    }
    
    if config.rtol <= T::zero() || config.rtol >= T::one() {
        return Err(TSVDError::InvalidTolerance(
            format!("Tolerance must be in (0, 1), got {}", config.rtol.into())
        ));
    }
    
    // Step 1: Apply RRQR to A
    let mut a_copy = matrix.clone();
    let (qr, k) = rrqr(&mut a_copy, config.rtol);
    
    if k == 0 {
        // Matrix has zero rank
        return Ok(crate::svd::jacobi::SVDResult {
            u: Tensor::from_elem((m, 0), T::zero()),
            s: Tensor::from_elem((0,), T::zero()),
            v: Tensor::from_elem((n, 0), T::zero()),
            rank: 0,
        });
    }
    
    // Step 2: Truncate QR result to rank k
    let (q_trunc, r_trunc) = truncate_qr_result(&qr, k);
    
    // Step 3: Compute SVD of R^T
    let r_shape = *r_trunc.shape();
    let r_transpose = Tensor::from_fn((r_shape.1, r_shape.0), |idx| r_trunc[[idx[1], idx[0]]]);
    let svd_result = jacobi_svd(&r_transpose);
    
    // Step 4: Reconstruct final U and V
    // U = Q_trunc * V_svd (manual matrix multiplication)
    let u_shape = *q_trunc.shape();
    let v_svd_shape = *svd_result.v.shape();
    let u = Tensor::from_fn((u_shape.0, v_svd_shape.1), |idx| {
        let mut sum = T::zero();
        for l in 0..u_shape.1 {
            sum = sum + q_trunc[[idx[0], l]] * svd_result.v[[l, idx[1]]];
        }
        sum
    });
    
    // V = P^(-1) * U_svd (where P^(-1) is the inverse permutation matrix)
    // First, compute inverse permutation
    let mut inv_perm = vec![0; n];
    for i in 0..n {
        inv_perm[qr.jpvt[[i]]] = i;
    }
    
    // Apply inverse permutation: V[i, j] = U_svd[inv_perm[i], j]
    let u_svd_shape = *svd_result.u.shape();
    let v = Tensor::from_fn((n, u_svd_shape.1), |idx| {
        svd_result.u[[inv_perm[idx[0]], idx[1]]]
    });
    
    Ok(crate::svd::jacobi::SVDResult {
        u,
        s: svd_result.s,
        v,
        rank: k,
    })
}

/// Convenience function for f64 precision
pub fn tsvd_f64(
    matrix: &Tensor<f64, (usize, usize)>,
    rtol: f64,
) -> Result<crate::svd::jacobi::SVDResult<f64>, TSVDError> {
    tsvd(matrix, TSVDConfig::new(rtol))
}

/// Convenience function for TwoFloat precision
pub fn tsvd_twofloat(
    matrix: &Tensor<crate::precision::TwoFloatPrecision, (usize, usize)>,
    rtol: crate::precision::TwoFloatPrecision,
) -> Result<crate::svd::jacobi::SVDResult<crate::precision::TwoFloatPrecision>, TSVDError> {
    let config = TSVDConfig::new(rtol);
    tsvd(matrix, config)
}

/// Convenience function to convert f64 matrix to TwoFloat and compute SVD
pub fn tsvd_twofloat_from_f64(
    matrix: &Tensor<f64, (usize, usize)>,
    rtol: f64,
) -> Result<crate::svd::jacobi::SVDResult<crate::precision::TwoFloatPrecision>, TSVDError> {
    // Convert f64 matrix to TwoFloatPrecision
    let shape = *matrix.shape();
    let (m, n) = shape;
    let matrix_tf = Tensor::from_fn((m, n), |idx| {
        crate::precision::TwoFloatPrecision::from_f64(matrix[[idx[0], idx[1]]])
    });
    
    tsvd_twofloat(&matrix_tf, crate::precision::TwoFloatPrecision::from_f64(rtol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_tsvd_identity() {
        let a = Array2::eye(3);
        let result = tsvd_f64(&a, 1e-10).unwrap();
        
        // Identity matrix should have singular values all equal to 1
        for &s in result.s.iter() {
            assert_abs_diff_eq!(s, 1.0, epsilon = 1e-10);
        }
        
        assert_eq!(result.rank, 3);
    }
    
    #[test]
    fn test_tsvd_rank_one() {
        let a = array![
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ];
        
        let result = tsvd_f64(&a, 1e-10).unwrap();
        
        // Should have rank 1
        assert_eq!(result.rank, 1);
        
        // Should have one non-zero singular value
        assert!(result.s[0] > 1.0);
    }
    
    #[test]
    fn test_tsvd_empty_matrix() {
        let a = Array2::<f64>::zeros((0, 0));
        let result = tsvd_f64(&a, 1e-10);
        
        assert!(matches!(result, Err(TSVDError::EmptyMatrix)));
    }
    
    #[test]
    fn test_tsvd_invalid_tolerance() {
        let a = Array2::eye(2);
        let result = tsvd_f64(&a, -1.0);
        
        assert!(matches!(result, Err(TSVDError::InvalidTolerance(_))));
    }
}
