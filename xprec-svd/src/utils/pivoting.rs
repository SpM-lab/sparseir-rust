//! Column pivoting utilities

use mdarray::Tensor;
use crate::precision::Precision;

/// Apply column permutation to a matrix
/// 
/// Given a permutation vector `p` and a matrix, permutes the columns
/// according to the permutation.
pub fn apply_column_permutation<T: Precision>(
    matrix: &mut ndarray::Array2<T>,
    permutation: &Array1<usize>,
) {
    let n = matrix.ncols();
    assert_eq!(permutation.len(), n);
    
    // Create a copy for reading
    let original = matrix.clone();
    
    // Apply permutation
    for j in 0..n {
        let new_j = permutation[j];
        matrix.column_mut(j).assign(&original.column(new_j));
    }
}

/// Create permutation matrix from permutation vector
/// 
/// Given a permutation vector `p`, creates the corresponding
/// permutation matrix P such that P[i, p[i]] = 1.
pub fn permutation_matrix<T: Precision>(
    permutation: &Array1<usize>,
) -> ndarray::Array2<T> {
    let n = permutation.len();
    let mut p = ndarray::Array2::zeros((n, n));
    
    for i in 0..n {
        p[[i, permutation[i]]] = T::one();
    }
    
    p
}

/// Invert a permutation vector
/// 
/// Given a permutation vector `p`, computes the inverse permutation
/// such that inv_p[p[i]] = i for all i.
pub fn invert_permutation(permutation: &Array1<usize>) -> Array1<usize> {
    let n = permutation.len();
    let mut inv_p = Array1::zeros(n);
    
    for i in 0..n {
        inv_p[permutation[i]] = i;
    }
    
    inv_p
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
