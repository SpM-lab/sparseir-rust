//! Rank-Revealing QR with Column Pivoting (RRQR)

use crate::precision::Precision;
use mdarray::Tensor;
// RRQR implementation

/// Result of QR factorization with column pivoting
#[derive(Debug, Clone)]
pub struct QRPivoted<T: Precision> {
    /// Packed QR factorization (Q and R stored together)
    pub factors: Tensor<T, (usize, usize)>,
    /// Householder reflection coefficients
    pub taus: Tensor<T, (usize,)>,
    /// Column pivot indices
    pub jpvt: Tensor<usize, (usize,)>,
}

/// Find the index of the maximum element in a vector
fn argmax<T: Precision>(vec: &Tensor<T, (usize,)>) -> usize {
    let mut max_idx = 0;
    let mut max_val = vec[[0]];

    let n = vec.len();
    for i in 0..n {
        let val = vec[[i]];
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
/// * `use_pivoting` - If true, use column pivoting; if false, use standard QR (default: true)
///
/// # Returns
/// * `QRPivoted` - QR factorization result with pivot information
/// * `usize` - Effective numerical rank
pub fn rrqr_with_options<T: Precision>(
    matrix: &mut Tensor<T, (usize, usize)>,
    rtol: T,
    use_pivoting: bool,
) -> (QRPivoted<T>, usize) {
    let shape = *matrix.shape();
    let (m, n) = shape;
    let k = m.min(n);

    // Initialize pivot indices
    let mut jpvt: Tensor<usize, (usize,)> = Tensor::from_fn((n,), |idx| idx[0]);

    // Initialize tau vector
    let mut taus = Tensor::from_elem((k,), T::zero());

    // Compute initial column norms
    let mut xnorms = Tensor::from_fn((n,), |idx| {
        let j = idx[0];
        // Compute column norm manually
        let mut sum = T::zero();
        for i in 0..m {
            let val = matrix[[i, j]];
            sum = sum + val * val;
        }
        Precision::sqrt(sum)
    });
    let mut pnorms = xnorms.clone();

    let sqrteps = Precision::sqrt(<T as Precision>::epsilon());
    let mut rk = k;

    for i in 0..k {
        // Find column with maximum norm and swap if pivoting is enabled
        if use_pivoting {
            // Find max in pnorms[i..]
            let mut pvt = i;
            let mut max_val = pnorms[[i]];
            for j in (i + 1)..n {
                if pnorms[[j]] > max_val {
                    max_val = pnorms[[j]];
                    pvt = j;
                }
            }

            // Swap columns if necessary
            if i != pvt {
                // Swap in jpvt
                let temp_jpvt = jpvt[[i]];
                jpvt[[i]] = jpvt[[pvt]];
                jpvt[[pvt]] = temp_jpvt;

                // Swap in xnorms and pnorms
                let temp_x = xnorms[[i]];
                xnorms[[i]] = xnorms[[pvt]];
                xnorms[[pvt]] = temp_x;

                let temp_p = pnorms[[i]];
                pnorms[[i]] = pnorms[[pvt]];
                pnorms[[pvt]] = temp_p;

                // Swap columns manually
                for row in 0..m {
                    let temp = matrix[[row, i]];
                    matrix[[row, i]] = matrix[[row, pvt]];
                    matrix[[row, pvt]] = temp;
                }
            }
        }

        // Apply Householder reflection
        // Extract column i from row i onwards
        let col_len = m - i;
        let mut col_i: Vec<T> = (0..col_len).map(|idx| matrix[[i + idx, i]]).collect();
        let (tau_i, _) = super::householder::reflector(&mut col_i);
        taus[[i]] = tau_i;

        // Write back the modified column
        for idx in 0..col_len {
            matrix[[i + idx, i]] = col_i[idx];
        }

        // Apply reflection to remaining columns only if tau_i != 0
        if tau_i != T::zero() && i + 1 < n {
            super::householder::reflector_apply_to_block(matrix, i, i, tau_i, i + 1, n);
        }

        // Update column norms
        for j in (i + 1)..n {
            let temp = Precision::abs(matrix[[i, j]]) / pnorms[[j]];
            let temp = (T::one() + temp) * (T::one() - temp);
            let temp = temp * (pnorms[[j]] / xnorms[[j]]) * (pnorms[[j]] / xnorms[[j]]);

            if temp < sqrteps {
                // Recompute norm to avoid numerical issues
                let mut sum = T::zero();
                for row in (i + 1)..m {
                    let val = matrix[[row, j]];
                    sum = sum + val * val;
                }
                let recomputed = Precision::sqrt(sum);
                pnorms[[j]] = recomputed;
                xnorms[[j]] = recomputed;
            } else {
                pnorms[[j]] = pnorms[[j]] * Precision::sqrt(temp);
            }
        }

        // Check rank condition - use a more robust criterion
        let diag_abs = Precision::abs(matrix[[i, i]]);
        let max_diag_abs = Precision::abs(matrix[[0, 0]]);

        // Check if the diagonal element is too small relative to the first diagonal element
        if diag_abs < rtol * max_diag_abs || diag_abs < <T as Precision>::epsilon() {
            // Zero out remaining columns
            for row in i..m {
                for col in i..n {
                    matrix[[row, col]] = T::zero();
                }
            }
            // Zero out remaining taus
            for j in i..k {
                taus[[j]] = T::zero();
            }
            rk = i;
            break;
        }
    }

    (
        QRPivoted {
            factors: matrix.clone(),
            taus,
            jpvt,
        },
        rk,
    )
}

/// Rank-Revealing QR with Column Pivoting (default: with pivoting)
///
/// This is a convenience wrapper that calls `rrqr_with_options` with pivoting enabled.
///
/// # Arguments
/// * `matrix` - Input matrix (modified in-place)
/// * `rtol` - Relative tolerance for rank determination
///
/// # Returns
/// * `QRPivoted` - QR factorization result with pivot information
/// * `usize` - Effective numerical rank
pub fn rrqr<T: Precision>(
    matrix: &mut Tensor<T, (usize, usize)>,
    rtol: T,
) -> (QRPivoted<T>, usize) {
    rrqr_with_options(matrix, rtol, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mdarray::tensor;
    // Test utilities

    #[test]
    fn test_rrqr_simple() {
        let mut a = Tensor::from_fn((3, 3), |idx| {
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]][idx[0]][idx[1]]
        });

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
        let mut a = Tensor::from_fn((3, 3), |idx| if idx[0] == idx[1] { 1.0 } else { 0.0 });

        let (qr, rank) = rrqr(&mut a, 1e-10);

        // Identity matrix should have full rank
        assert_eq!(rank, 3);
    }

    #[test]
    fn test_rrqr_rank_one() {
        let mut a = Tensor::from_fn((3, 3), |_| 1.0);

        let (qr, rank) = rrqr(&mut a, 1e-10);

        // All columns are identical, so rank should be 1
        assert_eq!(rank, 1);
    }
}
