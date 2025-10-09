//! Vector and matrix norm computations

use mdarray::Tensor;
use crate::precision::Precision;

/// Compute the 2-norm (Euclidean norm) of a vector
pub fn norm_2<T: Precision>(vec: &Tensor<T, (usize,)>) -> T {
    let mut sum = T::zero();
    let n = vec.len();
    for i in 0..n {
        let val = vec[[i]];
        sum = sum + val * val;
    }
    Precision::sqrt(sum)
}

/// Compute the Frobenius norm of a matrix
pub fn norm_frobenius<T: Precision>(mat: &Tensor<T, (usize, usize)>) -> T {
    let shape = *mat.shape();
    let (m, n) = shape;
    let mut sum = T::zero();
    for i in 0..m {
        for j in 0..n {
            let val = mat[[i, j]];
            sum = sum + val * val;
        }
    }
    Precision::sqrt(sum)
}

/// Compute the maximum absolute value in a vector
pub fn norm_inf<T: Precision>(vec: &Tensor<T, (usize,)>) -> T {
    let mut max_val = T::zero();
    let n = vec.len();
    for i in 0..n {
        let abs_val = Precision::abs(vec[[i]]);
        if abs_val > max_val {
            max_val = abs_val;
        }
    }
    max_val
}

/// Compute the maximum absolute value in a matrix
pub fn norm_max<T: Precision>(mat: &Tensor<T, (usize, usize)>) -> T {
    let shape = *mat.shape();
    let (m, n) = shape;
    let mut max_val = T::zero();
    for i in 0..m {
        for j in 0..n {
            let abs_val = Precision::abs(mat[[i, j]]);
            if abs_val > max_val {
                max_val = abs_val;
            }
        }
    }
    max_val
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_norm_2() {
        let v = array![3.0, 4.0, 0.0];
        let norm = norm_2(v.view());
        assert_abs_diff_eq!(norm, 5.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_norm_frobenius() {
        let m = array![[3.0, 4.0], [0.0, 5.0]];
        let norm = norm_frobenius(m.view());
        assert_abs_diff_eq!(norm, (9.0 + 16.0 + 0.0 + 25.0).sqrt(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_norm_inf() {
        let v = array![1.0, -3.0, 2.0];
        let norm = norm_inf(v.view());
        assert_abs_diff_eq!(norm, 3.0, epsilon = 1e-10);
    }
}
