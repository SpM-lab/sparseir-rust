//! QR truncation utilities

use ndarray::{Array2, s};
use crate::precision::Precision;
use super::rrqr::QRPivoted;

/// Truncate QR factorization result to effective rank k
/// 
/// Given a QR factorization with effective rank k, extracts the truncated
/// Q matrix (m × k) and R matrix (k × n).
pub fn truncate_qr_result<T: Precision>(
    qr: &QRPivoted<T>,
    k: usize,
) -> (Array2<T>, Array2<T>) {
    let m = qr.factors.nrows();
    let n = qr.factors.ncols();
    
    // Extract Q_trunc (m × k) - Initialize as identity
    let mut q_trunc = Array2::eye(m);
    let q_trunc = q_trunc.slice_mut(s![.., ..k]).to_owned();
    let mut q_trunc = q_trunc;
    
    // Apply Householder reflections using lmul (like libsparseir)
    let min_mn = m.min(k);
    for i in (0..min_mn).rev() {
        if qr.taus[i] != T::zero() {
            let tau = qr.taus[i];
            
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
    let mut r_trunc = Array2::zeros((k, n));
    for i in 0..k {
        for j in i..n {
            r_trunc[[i, j]] = qr.factors[[i, j]];
        }
    }
    
    (q_trunc, r_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qr::rrqr::rrqr;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_truncate_qr_result() {
        // Create a test matrix
        let mut a = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];
        
        // Perform RRQR
        let (qr, rank) = rrqr(&mut a, 1e-10);
        
        // Truncate to the effective rank
        let (q_trunc, r_trunc) = truncate_qr_result(&qr, rank);
        
        // Verify dimensions
        assert_eq!(q_trunc.nrows(), 3);
        assert_eq!(q_trunc.ncols(), rank);
        assert_eq!(r_trunc.nrows(), rank);
        assert_eq!(r_trunc.ncols(), 3);
        
        // Verify that Q_trunc is orthogonal (Q^T Q = I)
        let qtq = q_trunc.t().dot(&q_trunc);
        for i in 0..rank {
            for j in 0..rank {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(qtq[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }
}
