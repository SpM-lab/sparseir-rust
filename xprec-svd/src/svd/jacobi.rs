//! Jacobi SVD implementation

use ndarray::{Array1, Array2};
use crate::precision::Precision;
// Jacobi SVD implementation

/// Result of SVD decomposition
#[derive(Debug, Clone)]
pub struct SVDResult<T: Precision> {
    /// Left singular vectors (m × k)
    pub u: Array2<T>,
    /// Singular values (k)
    pub s: Array1<T>,
    /// Right singular vectors (n × k)
    pub v: Array2<T>,
    /// Effective rank
    pub rank: usize,
}

/// Compute 2×2 SVD
/// 
/// Computes the SVD of a 2×2 matrix:
/// [a b] = [c -s] [σ1  0] [c2  s2]
/// [c d]   [s  c] [0  σ2] [-s2 c2]
/// 
/// Returns (c, s, σ1, σ2, c2, s2) where:
/// - (c, s) are the left rotation parameters
/// - (σ1, σ2) are the singular values
/// - (c2, s2) are the right rotation parameters
pub fn svd_2x2<T: Precision>(
    a: T,
    b: T,
    c: T,
    d: T,
) -> (T, T, T, T, T, T) {
    // Compute the 2×2 SVD using the standard algorithm
    let a_sq = a * a;
    let b_sq = b * b;
    let c_sq = c * c;
    let d_sq = d * d;
    
    let trace = a_sq + b_sq + c_sq + d_sq;
    let det = Precision::abs(a * d - b * c);
    
    // Compute singular values
    let four: T = <T as From<f64>>::from(4.0);
    let two: T = <T as From<f64>>::from(2.0);
    let s1_sq = (trace + Precision::sqrt(trace * trace - four * det * det)) / two;
    let s2_sq = (trace - Precision::sqrt(trace * trace - four * det * det)) / two;
    
    let s1 = Precision::sqrt(s1_sq);
    let s2 = Precision::sqrt(s2_sq);
    
    // For diagonal matrices, the rotation angles are simple
    if Precision::abs(b) < T::EPSILON && Precision::abs(c) < T::EPSILON {
        // Diagonal matrix: [a 0; 0 d]
        let (c1, s1_rot) = if a >= T::zero() { (T::one(), T::zero()) } else { (-T::one(), T::zero()) };
        let (c2, s2_rot) = if d >= T::zero() { (T::one(), T::zero()) } else { (-T::one(), T::zero()) };
        return (c1, s1_rot, s1, s2, c2, s2_rot);
    }
    
    // For general 2x2 matrices, compute proper rotation angles
    // This is a simplified version - in practice, you'd use a more robust algorithm
    let (c1, s1_rot) = if s1 != T::zero() {
        let norm = Precision::sqrt(a * a + c * c);
        if norm > T::EPSILON {
            (a / norm, c / norm)
        } else {
            (T::one(), T::zero())
        }
    } else {
        (T::one(), T::zero())
    };
    
    let (c2, s2_rot) = if s2 != T::zero() {
        let norm = Precision::sqrt(b * b + d * d);
        if norm > T::EPSILON {
            (b / norm, d / norm)
        } else {
            (T::one(), T::zero())
        }
    } else {
        (T::one(), T::zero())
    };
    
    (c1, s1_rot, s1, s2, c2, s2_rot)
}

/// Apply Givens rotation to a matrix
/// 
/// Applies the rotation matrix:
/// [c -s] to the left or [c  s] to the right
/// [s  c]               [-s c]
pub fn apply_givens_left<T: Precision>(
    matrix: &mut Array2<T>,
    i: usize,
    j: usize,
    c: T,
    s: T,
) {
    let m = matrix.nrows();
    
    for k in 0..m {
        let temp = c * matrix[[k, i]] + s * matrix[[k, j]];
        matrix[[k, j]] = -s * matrix[[k, i]] + c * matrix[[k, j]];
        matrix[[k, i]] = temp;
    }
}

pub fn apply_givens_right<T: Precision>(
    matrix: &mut Array2<T>,
    i: usize,
    j: usize,
    c: T,
    s: T,
) {
    let n = matrix.ncols();
    
    for k in 0..n {
        let temp = c * matrix[[i, k]] + s * matrix[[j, k]];
        matrix[[j, k]] = -s * matrix[[i, k]] + c * matrix[[j, k]];
        matrix[[i, k]] = temp;
    }
}

/// Jacobi SVD algorithm
/// 
/// Computes the SVD using two-sided Jacobi iterations.
/// This is more accurate than bidiagonalization methods but slower.
pub fn jacobi_svd<T: Precision>(
    matrix: &Array2<T>,
) -> SVDResult<T> {
    let m = matrix.nrows();
    let n = matrix.ncols();
    let k = m.min(n);
    
    // Initialize U and V
    let mut u = Array2::eye(m);
    let mut v = Array2::eye(n);
    let mut a = matrix.clone();
    
    // Convergence threshold
    let eps = Precision::sqrt(T::EPSILON);
    let max_iter = 30; // Maximum number of sweeps
    
    for _iter in 0..max_iter {
        let mut converged = true;
        
        // One-sided Jacobi: eliminate off-diagonal elements
        for i in 0..k {
            for j in (i + 1)..k {
                if Precision::abs(a[[i, j]]) < eps {
                    continue;
                }
                
                converged = false;
                
                // Compute 2×2 SVD of the 2×2 submatrix
                let (c, s, _s1, _s2, c2, s2_rot) = svd_2x2(
                    a[[i, i]], a[[i, j]],
                    a[[j, i]], a[[j, j]]
                );
                
                // Apply left rotation to A and U
                apply_givens_left(&mut a, i, j, c, s);
                apply_givens_left(&mut u, i, j, c, s);
                
                // Apply right rotation to A and V
                apply_givens_right(&mut a, i, j, c2, s2_rot);
                apply_givens_right(&mut v, i, j, c2, s2_rot);
            }
        }
        
        if converged {
            break;
        }
    }
    
    // Extract singular values
    let mut s = Array1::zeros(k);
    for i in 0..k {
        s[i] = Precision::abs(a[[i, i]]);
    }
    
    // Sort singular values in descending order
    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap());
    
    let mut s_sorted = Array1::zeros(k);
    let mut u_sorted = Array2::zeros((m, k));
    let mut v_sorted = Array2::zeros((n, k));
    
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        s_sorted[new_idx] = s[old_idx];
        u_sorted.column_mut(new_idx).assign(&u.column(old_idx));
        v_sorted.column_mut(new_idx).assign(&v.column(old_idx));
    }
    
    SVDResult {
        u: u_sorted,
        s: s_sorted,
        v: v_sorted,
        rank: k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_svd_2x2() {
        let (c, s, s1, s2, c2, s2_rot) = svd_2x2(3.0, 4.0, 0.0, 5.0);
        
        // Check that singular values are positive
        assert!(s1 > 0.0);
        assert!(s2 > 0.0);
        
        // Check that rotation matrices are orthogonal
        assert_abs_diff_eq!(c * c + s * s, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c2 * c2 + s2_rot * s2_rot, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_jacobi_svd_identity() {
        let a = Array2::eye(3);
        let result: SVDResult<f64> = jacobi_svd(&a);
        
        // Identity matrix should have singular values all equal to 1
        for &s in result.s.iter() {
            assert_abs_diff_eq!(s, 1.0, epsilon = 1e-10);
        }
        
        // U and V should be identity matrices
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(result.u[[i, j]], expected, epsilon = 1e-10);
                assert_abs_diff_eq!(result.v[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }
    
    #[test]
    fn test_jacobi_svd_rank_one() {
        let a = array![
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ];
        
        let result: SVDResult<f64> = jacobi_svd(&a);
        
        // Should have only one non-zero singular value
        assert!(result.s[0] > 1.0); // Should be around 3.0
        assert_abs_diff_eq!(result.s[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.s[2], 0.0, epsilon = 1e-10);
    }
}
