//! Column pivoting utilities

use mdarray::Tensor;
use crate::precision::Precision;

/// Apply column permutation to a matrix
/// 
/// Given a permutation vector `p` and a matrix, permutes the columns
/// according to the permutation.
pub fn apply_column_permutation<T: Precision>(
    matrix: &mut Tensor<T, (usize, usize)>,
    permutation: &Tensor<usize, (usize,)>,
) {
    let shape = *matrix.shape();
    let (m, n) = shape;
    assert_eq!(permutation.len(), n);
    
    // Create a copy for reading
    let original = matrix.clone();
    
    // Apply permutation
    for j in 0..n {
        let new_j = permutation[[j]];
        for i in 0..m {
            matrix[[i, j]] = original[[i, new_j]];
        }
    }
}

/// Create permutation matrix from permutation vector
/// 
/// Given a permutation vector `p`, creates the corresponding
/// permutation matrix P such that P[i, p[i]] = 1.
pub fn permutation_matrix<T: Precision>(
    permutation: &Tensor<usize, (usize,)>,
) -> Tensor<T, (usize, usize)> {
    let n = permutation.len();
    Tensor::from_fn((n, n), |idx| {
        if idx[1] == permutation[[idx[0]]] {
            T::one()
        } else {
            T::zero()
        }
    })
}

/// Invert a permutation vector
/// 
/// Given a permutation vector `p`, computes the inverse permutation
/// such that inv_p[p[i]] = i for all i.
pub fn invert_permutation(permutation: &Tensor<usize, (usize,)>) -> Tensor<usize, (usize,)> {
    let n = permutation.len();
    let mut inv_p = vec![0; n];
    
    for i in 0..n {
        inv_p[permutation[[i]]] = i;
    }
    
    Tensor::from_fn((n,), |idx| inv_p[idx[0]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_apply_column_permutation() {
        let mut m = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ];
        let p = array![2, 0, 1]; // Swap columns 0<->2, 1<->0, 2<->1
        
        apply_column_permutation(&mut m, &p);
        
        let expected = array![
            [3.0, 1.0, 2.0],
            [6.0, 4.0, 5.0]
        ];
        
        for i in 0..2 {
            for j in 0..3 {
                assert_abs_diff_eq!(m[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }
    
    #[test]
    fn test_permutation_matrix() {
        let p = array![2, 0, 1];
        let p_matrix = permutation_matrix::<f64>(&p);
        
        let expected = array![
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ];
        
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(p_matrix[[i, j]], expected[[i, j]], epsilon = 1e-10);
            }
        }
    }
    
    #[test]
    fn test_invert_permutation() {
        let p = array![2, 0, 1];
        let inv_p = invert_permutation(&p);
        
        let expected = array![1, 2, 0];
        
        for i in 0..3 {
            assert_eq!(inv_p[i], expected[i]);
        }
    }
}
