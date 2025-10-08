//! Compatibility layer between ndarray and mdarray
//!
//! This module provides conversion functions to bridge between ndarray and mdarray
//! during the migration period. Once migration is complete, this module can be removed.

use ndarray::{Array1, Array2};
use mdarray::Tensor;

/// Convert ndarray Array1 to mdarray Tensor
pub fn array1_to_tensor<T: Clone>(arr: &Array1<T>) -> Tensor<T, (usize,)> {
    let n = arr.len();
    Tensor::from_fn((n,), |idx| arr[idx[0]].clone())
}

/// Convert mdarray Tensor to ndarray Array1
pub fn tensor_to_array1<T: Clone>(tensor: &Tensor<T, (usize,)>) -> Array1<T> {
    let n = tensor.len();
    Array1::from_vec((0..n).map(|i| tensor[[i]].clone()).collect())
}

/// Convert ndarray Array2 to mdarray Tensor
pub fn array2_to_tensor<T: Clone>(arr: &Array2<T>) -> Tensor<T, (usize, usize)> {
    let (m, n) = (arr.nrows(), arr.ncols());
    Tensor::from_fn((m, n), |idx| arr[[idx[0], idx[1]]].clone())
}

/// Convert mdarray Tensor to ndarray Array2
pub fn tensor_to_array2<T: Clone>(tensor: &Tensor<T, (usize, usize)>) -> Array2<T> {
    let shape = *tensor.shape();
    let (m, n) = shape;
    let mut vec = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            vec.push(tensor[[i, j]].clone());
        }
    }
    Array2::from_shape_vec((m, n), vec).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use mdarray::tensor;

    #[test]
    fn test_array1_to_tensor() {
        let arr = array![1.0, 2.0, 3.0];
        let t = array1_to_tensor(&arr);
        
        assert_eq!(t.len(), 3);
        assert_eq!(t[[0]], 1.0);
        assert_eq!(t[[1]], 2.0);
        assert_eq!(t[[2]], 3.0);
    }

    #[test]
    fn test_tensor_to_array1() {
        let t = tensor![1.0, 2.0, 3.0];
        let arr = tensor_to_array1(&t);
        
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], 1.0);
        assert_eq!(arr[1], 2.0);
        assert_eq!(arr[2], 3.0);
    }

    #[test]
    fn test_array2_to_tensor() {
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let t = array2_to_tensor(&arr);
        
        assert_eq!(*t.shape(), (2, 2));
        assert_eq!(t[[0, 0]], 1.0);
        assert_eq!(t[[0, 1]], 2.0);
        assert_eq!(t[[1, 0]], 3.0);
        assert_eq!(t[[1, 1]], 4.0);
    }

    #[test]
    fn test_tensor_to_array2() {
        let t = tensor![[1.0, 2.0], [3.0, 4.0]];
        let arr = tensor_to_array2(&t);
        
        assert_eq!(arr.nrows(), 2);
        assert_eq!(arr.ncols(), 2);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[0, 1]], 2.0);
        assert_eq!(arr[[1, 0]], 3.0);
        assert_eq!(arr[[1, 1]], 4.0);
    }

    #[test]
    fn test_roundtrip_array1() {
        let arr = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let t = array1_to_tensor(&arr);
        let arr2 = tensor_to_array1(&t);
        
        assert_eq!(arr, arr2);
    }

    #[test]
    fn test_roundtrip_array2() {
        let arr = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let t = array2_to_tensor(&arr);
        let arr2 = tensor_to_array2(&t);
        
        assert_eq!(arr, arr2);
    }
}

