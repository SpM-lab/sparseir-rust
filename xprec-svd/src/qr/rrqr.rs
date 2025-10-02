//! Rank-Revealing QR with Column Pivoting (RRQR)

use ndarray::{Array1, Array2, ArrayView1, s};
use crate::precision::Precision;
// RRQR implementation

/// Result of QR factorization with column pivoting
#[derive(Debug, Clone)]
pub struct QRPivoted<T: Precision> {
    /// Packed QR factorization (Q and R stored together)
    pub factors: Array2<T>,
    /// Householder reflection coefficients
    pub taus: Array1<T>,
    /// Column pivot indices
    pub jpvt: Array1<usize>,
}

/// Find the index of the maximum element in a vector
fn argmax<T: Precision>(vec: ArrayView1<T>) -> usize {
    let mut max_idx = 0;
    let mut max_val = vec[0];
    
    for (i, &val) in vec.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    
    max_idx
}

/// Rank-Revealing QR with Column Pivoting
/// 
/// Performs QR factorization with column pivoting to reveal the numerical rank
/// of the matrix. The algorithm stops when the diagonal element becomes small
/// relative to the tolerance.
/// 
/// # Arguments
/// * `matrix` - Input matrix (modified in-place)
/// * `rtol` - Relative tolerance for rank determination
/// 
/// # Returns
/// * `QRPivoted` - QR factorization result with pivot information
/// * `usize` - Effective numerical rank
pub fn rrqr<T: Precision>(
    matrix: &mut Array2<T>,
    rtol: T,
) -> (QRPivoted<T>, usize) {
    let m = matrix.nrows();
    let n = matrix.ncols();
    let k = m.min(n);
    
    // Initialize pivot indices
    let mut jpvt: Array1<usize> = Array1::from_iter(0..n);
    
    // Initialize tau vector
    let mut taus = Array1::zeros(k);
    
    // Compute initial column norms
    let mut xnorms = Array1::zeros(n);
    let mut pnorms = Array1::zeros(n);
    
    for j in 0..n {
        let norm = crate::utils::norms::norm_2(matrix.column(j));
        xnorms[j] = norm;
        pnorms[j] = norm;
    }
    
    let sqrt_eps = Precision::sqrt(T::EPSILON);
    let mut effective_rank = k;
    
    for i in 0..k {
        // Find column with maximum norm
        let _remaining_cols = n - i;
        let max_idx = argmax(pnorms.slice(s![i..])) + i;
        
        // Swap columns if necessary
        if i != max_idx {
            jpvt.swap(i, max_idx);
            xnorms.swap(i, max_idx);
            pnorms.swap(i, max_idx);
            // Swap columns manually
            for row in 0..matrix.nrows() {
                let temp = matrix[[row, i]];
                matrix[[row, i]] = matrix[[row, max_idx]];
                matrix[[row, max_idx]] = temp;
            }
        }
        
        // Apply Householder reflection
        let col_i = matrix.slice_mut(s![i.., i]);
        let (tau, _) = super::householder::reflector(col_i);
        taus[i] = tau;
        
        // Apply reflection to remaining columns only if tau != 0
        if tau != T::zero() && i + 1 < n {
            let v = matrix.slice(s![i.., i]).to_owned();
            let block = matrix.slice_mut(s![i.., i+1..]);
            super::householder::reflector_apply(v.view(), tau, block);
        }
        
        // Update column norms
        for j in (i + 1)..n {
            let temp = Precision::abs(matrix[[i, j]]) / pnorms[j];
            let temp = (T::one() + temp) * (T::one() - temp);
            let temp = temp * (pnorms[j] / xnorms[j]) * (pnorms[j] / xnorms[j]);
            
            if temp < sqrt_eps {
                // Recompute norm to avoid numerical issues
                let recomputed = crate::utils::norms::norm_2(matrix.slice(s![i+1.., j]));
                pnorms[j] = recomputed;
                xnorms[j] = recomputed;
            } else {
                pnorms[j] = pnorms[j] * Precision::sqrt(temp);
            }
        }
        
        // Check rank condition - use a more robust criterion
        let diag_abs = Precision::abs(matrix[[i, i]]);
        let max_diag_abs = Precision::abs(matrix[[0, 0]]);
        
        // Check if the diagonal element is too small relative to the first diagonal element
        if diag_abs < rtol * max_diag_abs || diag_abs < T::EPSILON {
            // Zero out remaining columns
            for row in i..m {
                for col in i..n {
                    matrix[[row, col]] = T::zero();
                }
            }
            // Zero out remaining taus
            for j in i..k {
                taus[j] = T::zero();
            }
            effective_rank = i;
            break;
        }
    }
    
    (
        QRPivoted {
            factors: matrix.clone(),
            taus,
            jpvt,
        },
        effective_rank,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    // Test utilities
    
    #[test]
    fn test_rrqr_simple() {
        let mut a = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];
        
        let (qr, rank) = rrqr(&mut a, 1e-10);
        
        // The matrix should have rank 2 (third column is linear combination of first two)
        assert_eq!(rank, 2);
        
        // Check that pivot indices are valid
        assert_eq!(qr.jpvt.len(), 3);
        
        // Check that taus are computed
        assert_eq!(qr.taus.len(), 3);
    }
    
    #[test]
    fn test_rrqr_full_rank() {
        let mut a = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        
        let (qr, rank) = rrqr(&mut a, 1e-10);
        
        // Identity matrix should have full rank
        assert_eq!(rank, 3);
    }
    
    #[test]
    fn test_rrqr_rank_one() {
        let mut a = array![
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ];
        
        let (qr, rank) = rrqr(&mut a, 1e-10);
        
        // All columns are identical, so rank should be 1
        assert_eq!(rank, 1);
    }
}
