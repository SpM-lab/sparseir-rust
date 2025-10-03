use approx::assert_abs_diff_eq;
use ndarray::{array, Array2};
use xprec_svd::svd::jacobi::jacobi_svd;

#[test]
fn test_jacobi_svd_identity() {
    // Test with identity matrix
    let a = Array2::<f64>::eye(3);
    let result = jacobi_svd(&a);
    
    // Identity matrix should have singular values all equal to 1
    for &s in result.s.iter() {
        assert_abs_diff_eq!(s, 1.0, epsilon = 1e-10);
    }
    
    // U and V should be identity matrices (or permutations of identity)
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(result.u[[i, j]].abs(), expected, epsilon = 1e-10);
            assert_abs_diff_eq!(result.v[[i, j]].abs(), expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_jacobi_svd_rank_one() {
    // Test with rank-1 matrix (all ones)
    let a = array![
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ];
    
    let result = jacobi_svd(&a);
    
    // Should have only one non-zero singular value (around 3.0)
    assert!(result.s[0] > 1.0);
    assert_abs_diff_eq!(result.s[1], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result.s[2], 0.0, epsilon = 1e-10);
}

#[test]
fn test_jacobi_svd_2x2_rank_one() {
    // Test with 2x2 rank-1 matrix
    let a = array![
        [1.0, 1.0],
        [1.0, 1.0]
    ];
    
    let result = jacobi_svd(&a);
    
    // First singular value should be around 2.0
    assert_abs_diff_eq!(result.s[0], 2.0, epsilon = 1e-9);
    
    // Second singular value should be zero
    assert_abs_diff_eq!(result.s[1], 0.0, epsilon = 1e-10);
}

#[test]
fn test_jacobi_svd_diagonal() {
    // Test with diagonal matrix
    let a = array![
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0]
    ];
    
    let result = jacobi_svd(&a);
    
    // Singular values should be [3, 2, 1] in descending order
    assert_abs_diff_eq!(result.s[0], 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result.s[1], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result.s[2], 1.0, epsilon = 1e-10);
}

#[test]
fn test_jacobi_svd_orthogonality() {
    // Test that U and V are orthogonal
    let a = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];
    
    let result = jacobi_svd(&a);
    
    // Check U^T * U = I
    let utu = result.u.t().dot(&result.u);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(utu[[i, j]], expected, epsilon = 1e-10);
        }
    }
    
    // Check V^T * V = I
    let vtv = result.v.t().dot(&result.v);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(vtv[[i, j]], expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_jacobi_svd_zero_matrix() {
    // Test with zero matrix
    let a = Array2::<f64>::zeros((3, 3));
    let result = jacobi_svd(&a);
    
    // All singular values should be zero
    for i in 0..3 {
        assert_abs_diff_eq!(result.s[i], 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_jacobi_svd_single_element() {
    // Test with 1x1 matrix
    let a = array![[5.0]];
    let result = jacobi_svd(&a);
    
    // Singular value should be 5.0
    assert_abs_diff_eq!(result.s[0], 5.0, epsilon = 1e-10);
    
    // U and V should be [[1]] or [[-1]]
    assert_abs_diff_eq!(result.u[[0, 0]].abs(), 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result.v[[0, 0]].abs(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_jacobi_svd_singular_values_positive() {
    // Test that singular values are non-negative
    let a = array![
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ];
    
    let result = jacobi_svd(&a);
    
    // Check that singular values are positive and in descending order
    assert!(result.s[0] >= result.s[1]);
    for i in 0..2 {
        assert!(result.s[i] >= 0.0);
    }
}
