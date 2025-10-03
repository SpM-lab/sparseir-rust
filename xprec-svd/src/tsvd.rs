//! Truncated SVD (TSVD) implementation

use ndarray::{Array1, Array2};
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
            convergence_threshold: Precision::sqrt(T::EPSILON),
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
    matrix: &Array2<T>,
    config: TSVDConfig<T>,
) -> Result<crate::svd::jacobi::SVDResult<T>, TSVDError> {
    let m = matrix.nrows();
    let n = matrix.ncols();
    
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
            u: Array2::zeros((m, 0)),
            s: Array1::zeros(0),
            v: Array2::zeros((n, 0)),
            rank: 0,
        });
    }
    
    // Step 2: Truncate QR result to rank k
    let (q_trunc, r_trunc) = truncate_qr_result(&qr, k);
    
    // Step 3: Compute SVD of R^T
    let r_transpose = r_trunc.t().to_owned();
    let svd_result = jacobi_svd(&r_transpose);
    
    // Step 4: Reconstruct final U and V
    // U = Q_trunc * V_svd
    let u = q_trunc.dot(&svd_result.v);
    
    // V = P * U_svd (where P is the permutation matrix)
    let mut v = Array2::zeros((n, k));
    for i in 0..k {
        for j in 0..n {
            v[[j, i]] = svd_result.u[[qr.jpvt[j], i]];
        }
    }
    
    Ok(crate::svd::jacobi::SVDResult {
        u,
        s: svd_result.s,
        v,
        rank: k,
    })
}

/// Convenience function for f64 precision
pub fn tsvd_f64(
    matrix: &Array2<f64>,
    rtol: f64,
) -> Result<crate::svd::jacobi::SVDResult<f64>, TSVDError> {
    tsvd(matrix, TSVDConfig::new(rtol))
}

// TwoFloat support can be added later when ExtendedFloat wrapper is implemented

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
