//! Result validation utilities

use crate::precision::Precision;
use mdarray::Tensor;

/// Validate SVD result
///
/// Checks that the SVD decomposition A = U * S * V^T is correct
/// and that U and V are orthogonal matrices.
pub fn validate_svd<T: Precision>(
    original: &Tensor<T, (usize, usize)>,
    u: &Tensor<T, (usize, usize)>,
    s: &Tensor<T, (usize,)>,
    v: &Tensor<T, (usize, usize)>,
    tolerance: T,
) -> bool {
    let orig_shape = *original.shape();
    let (m, n) = orig_shape;
    let k = s.len();

    // Check dimensions
    let u_shape = *u.shape();
    let v_shape = *v.shape();
    if u_shape.0 != m || u_shape.1 != k {
        return false;
    }
    if v_shape.0 != n || v_shape.1 != k {
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
fn is_orthogonal<T: Precision>(matrix: &Tensor<T, (usize, usize)>, tolerance: T) -> bool {
    let shape = *matrix.shape();
    let k = shape.1;

    // Compute U^T * U and check against identity
    for i in 0..k {
        for j in 0..k {
            let mut sum = T::zero();
            for row in 0..shape.0 {
                sum = sum + matrix[[row, i]] * matrix[[row, j]];
            }
            let expected = if i == j { T::one() } else { T::zero() };
            if Precision::abs(sum - expected) > tolerance {
                return false;
            }
        }
    }

    true
}

/// Check if singular values are valid
fn is_singular_values_valid<T: Precision>(s: &Tensor<T, (usize,)>, tolerance: T) -> bool {
    let k = s.len();

    // Check that all singular values are non-negative
    for i in 0..k {
        if s[[i]] < T::zero() - tolerance {
            return false;
        }
    }

    // Check that singular values are sorted in descending order
    if k > 1 {
        for i in 0..(k - 1) {
            if s[[i]] < s[[i + 1]] - tolerance {
                return false;
            }
        }
    }

    true
}

/// Check if reconstruction is valid
fn is_reconstruction_valid<T: Precision>(
    original: &Tensor<T, (usize, usize)>,
    u: &Tensor<T, (usize, usize)>,
    s: &Tensor<T, (usize,)>,
    v: &Tensor<T, (usize, usize)>,
    tolerance: T,
) -> bool {
    let orig_shape = *original.shape();
    let (m, n) = orig_shape;
    let k = s.len();

    // Check that ||A - U*S*V^T||_F < tolerance * ||A||_F
    let mut diff_norm_sq = T::zero();
    let mut orig_norm_sq = T::zero();

    for i in 0..m {
        for j in 0..n {
            // Compute reconstructed element
            let mut reconstructed = T::zero();
            for l in 0..k {
                reconstructed = reconstructed + u[[i, l]] * s[[l]] * v[[j, l]];
            }

            let diff = original[[i, j]] - reconstructed;
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
    use mdarray::tensor;

    #[test]
    fn test_validate_svd_identity() {
        let original = Tensor::from_fn((3, 3), |idx| if idx[0] == idx[1] { 1.0 } else { 0.0 });
        let u = Tensor::from_fn((3, 3), |idx| if idx[0] == idx[1] { 1.0 } else { 0.0 });
        let s = Tensor::from_fn((3,), |_| 1.0);
        let v = Tensor::from_fn((3, 3), |idx| if idx[0] == idx[1] { 1.0 } else { 0.0 });

        assert!(validate_svd(&original, &u, &s, &v, 1e-10));
    }

    #[test]
    fn test_is_orthogonal() {
        let u = Tensor::from_fn((3, 3), |idx| if idx[0] == idx[1] { 1.0 } else { 0.0 });
        assert!(is_orthogonal(&u, 1e-10));

        let u = Tensor::from_fn((2, 2), |idx| if idx[0] == idx[1] { 1.0 } else { 0.0 });
        assert!(is_orthogonal(&u, 1e-10));

        let u = Tensor::from_fn((2, 2), |idx| [[1.0, 1.0], [0.0, 1.0]][idx[0]][idx[1]]);
        assert!(!is_orthogonal(&u, 1e-10));
    }

    #[test]
    fn test_is_singular_values_valid() {
        let s = Tensor::from_fn((3,), |idx| [3.0, 2.0, 1.0][idx[0]]);
        assert!(is_singular_values_valid(&s, 1e-10));

        let s = Tensor::from_fn((3,), |idx| [1.0, 2.0, 3.0][idx[0]]);
        assert!(!is_singular_values_valid(&s, 1e-10));

        let s = Tensor::from_fn((3,), |idx| [3.0, -1.0, 1.0][idx[0]]);
        assert!(!is_singular_values_valid(&s, 1e-10));
    }
}
