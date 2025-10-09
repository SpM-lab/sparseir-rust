//! Result validation utilities

use ndarray::{Array2, ArrayView1, ArrayView2};
use crate::precision::Precision;

/// Validate SVD result
/// 
/// Checks that the SVD decomposition A = U * S * V^T is correct
/// and that U and V are orthogonal matrices.
pub fn validate_svd<T: Precision>(
    original: ArrayView2<T>,
    u: ArrayView2<T>,
    s: ArrayView1<T>,
    v: ArrayView2<T>,
    tolerance: T,
) -> bool {
    let m = original.nrows();
    let n = original.ncols();
    let k = s.len();
    
    // Check dimensions
    if u.nrows() != m || u.ncols() != k {
        return false;
    }
    if v.nrows() != n || v.ncols() != k {
        return false;
    }
    
    // Check that U is orthogonal (U^T * U = I)
    if !is_orthogonal(u, tolerance) {
        return false;
    }
    
    // Check that V is orthogonal (V^T * V = I)
    if !is_orthogonal(v, tolerance) {
        return false;
    }
    
    // Check that singular values are non-negative and sorted
    if !is_singular_values_valid(s, tolerance) {
        return false;
    }
    
    // Check reconstruction: A ≈ U * S * V^T
    if !is_reconstruction_valid(original, u, s, v, tolerance) {
        return false;
    }
    
    true
}

/// Check if a matrix is orthogonal
fn is_orthogonal<T: Precision>(matrix: ArrayView2<T>, tolerance: T) -> bool {
    let k = matrix.ncols();
    let mut utu = Array2::zeros((k, k));
    
    // Compute U^T * U
    for i in 0..k {
        for j in 0..k {
            let mut sum = T::zero();
            for row in 0..matrix.nrows() {
                sum = sum + matrix[[row, i]] * matrix[[row, j]];
            }
            utu[[i, j]] = sum;
        }
    }
    
    // Check that U^T * U = I
    for i in 0..k {
        for j in 0..k {
            let expected = if i == j { T::one() } else { T::zero() };
            if Precision::abs(utu[[i, j]] - expected) > tolerance {
                return false;
            }
        }
    }
    
    true
}

/// Check if singular values are valid
fn is_singular_values_valid<T: Precision>(s: ArrayView1<T>, tolerance: T) -> bool {
    // Check that all singular values are non-negative
    for &val in s.iter() {
        if val < -tolerance {
            return false;
        }
    }
    
    // Check that singular values are sorted in descending order
    for i in 0..(s.len() - 1) {
        if s[i] < s[i + 1] - tolerance {
            return false;
        }
    }
    
    true
}

/// Check if reconstruction is valid
fn is_reconstruction_valid<T: Precision>(
    original: ArrayView2<T>,
    u: ArrayView2<T>,
    s: ArrayView1<T>,
    v: ArrayView2<T>,
    tolerance: T,
) -> bool {
    let m = original.nrows();
    let n = original.ncols();
    let k = s.len();
    
    // Compute reconstructed matrix
    let mut reconstructed = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for l in 0..k {
                sum = sum + u[[i, l]] * s[l] * v[[j, l]];
            }
            reconstructed[[i, j]] = sum;
        }
    }
    
    // Check that ||A - U*S*V^T||_F < tolerance * ||A||_F
    let mut diff_norm_sq = T::zero();
    let mut orig_norm_sq = T::zero();
    
    for i in 0..m {
        for j in 0..n {
            let diff = original[[i, j]] - reconstructed[[i, j]];
            diff_norm_sq = diff_norm_sq + diff * diff;
            orig_norm_sq = orig_norm_sq + original[[i, j]] * original[[i, j]];
        }
    }
    
    let diff_norm = Precision::sqrt(diff_norm_sq);
    let orig_norm = Precision::sqrt(orig_norm_sq);
    
    if orig_norm == T::zero() {
        diff_norm < tolerance
    } else {
        diff_norm < tolerance * orig_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_validate_svd_identity() {
        let original = Array2::eye(3);
        let u = Array2::eye(3);
        let s = array![1.0, 1.0, 1.0];
        let v = Array2::eye(3);
        
        assert!(validate_svd(original.view(), u.view(), s.view(), v.view(), 1e-10));
    }
    
    #[test]
    fn test_is_orthogonal() {
        let u = Array2::eye(3);
        assert!(is_orthogonal(u.view(), 1e-10));
        
        let u = array![[1.0, 0.0], [0.0, 1.0]];
        assert!(is_orthogonal(u.view(), 1e-10));
        
        let u = array![[1.0, 1.0], [0.0, 1.0]]; // Not orthogonal
        assert!(!is_orthogonal(u.view(), 1e-10));
    }
    
    #[test]
    fn test_is_singular_values_valid() {
        let s = array![3.0, 2.0, 1.0]; // Valid: positive and sorted
        assert!(is_singular_values_valid(s.view(), 1e-10));
        
        let s = array![1.0, 2.0, 3.0]; // Invalid: not sorted
        assert!(!is_singular_values_valid(s.view(), 1e-10));
        
        let s = array![3.0, -1.0, 1.0]; // Invalid: negative
        assert!(!is_singular_values_valid(s.view(), 1e-10));
    }
}
