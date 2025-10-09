//! QR truncation utilities

use mdarray::Tensor;
use crate::precision::Precision;
use super::rrqr::QRPivoted;

/// Truncate QR factorization result to effective rank k
/// 
/// Given a QR factorization with effective rank k, extracts the truncated
/// Q matrix (m × k) and R matrix (k × n).
pub fn truncate_qr_result<T: Precision>(
    qr: &QRPivoted<T>,
    k: usize,
) -> (Tensor<T, (usize, usize)>, Tensor<T, (usize, usize)>) {
    let shape = *qr.factors.shape();
    let (m, n) = shape;
    
    // Extract Q_trunc (m × k) - Initialize as identity
    let mut q_trunc = Tensor::from_fn((m, k), |idx| {
        if idx[0] == idx[1] { T::one() } else { T::zero() }
    });
    
    // Apply Householder reflections using lmul (like libsparseir)
    let min_mn = m.min(k);
    for i in (0..min_mn).rev() {
        if qr.taus[[i]] != T::zero() {
            let tau = qr.taus[[i]];
            
            // Apply H_i to Q_trunc (like libsparseir's lmul)
            for j in 0..k {
                let mut vbj = q_trunc[[i, j]];
                for ii in (i + 1)..m {
                    vbj = vbj + qr.factors[[ii, i]] * q_trunc[[ii, j]];
                }
                vbj = tau * vbj;
                q_trunc[[i, j]] = q_trunc[[i, j]] - vbj;
                for ii in (i + 1)..m {
                    q_trunc[[ii, j]] = q_trunc[[ii, j]] - qr.factors[[ii, i]] * vbj;
                }
            }
        }
    }
    
    // Extract R_trunc (k × n)
    let r_trunc = Tensor::from_fn((k, n), |idx| {
        if idx[1] >= idx[0] {
            qr.factors[[idx[0], idx[1]]]
        } else {
            T::zero()
        }
    });
    
    (q_trunc, r_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qr::rrqr::rrqr;
    use mdarray::tensor;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_truncate_qr_result() {
        // Create a test matrix
        let mut a = Tensor::from_fn((3, 3), |idx| {
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]][idx[0]][idx[1]]
        });
        
        // Perform RRQR
        let (qr, rank) = rrqr(&mut a, 1e-10);
        
        // Truncate to the effective rank
        let (q_trunc, r_trunc) = truncate_qr_result(&qr, rank);
        
        // Verify dimensions
        let q_shape = *q_trunc.shape();
        let r_shape = *r_trunc.shape();
        assert_eq!(q_shape.0, 3);
        assert_eq!(q_shape.1, rank);
        assert_eq!(r_shape.0, rank);
        assert_eq!(r_shape.1, 3);
        
        // Verify that Q_trunc is orthogonal (Q^T Q = I)
        // Compute Q^T * Q manually
        for i in 0..rank {
            for j in 0..rank {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += q_trunc[[k, i]] * q_trunc[[k, j]];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(sum, expected, epsilon = 1e-10);
            }
        }
    }
}
