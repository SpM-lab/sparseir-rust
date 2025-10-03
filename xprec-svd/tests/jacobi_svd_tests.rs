use approx::{assert_abs_diff_eq, AbsDiffEq};
use ndarray::Array2;
use xprec_svd::svd::jacobi::jacobi_svd;
use xprec_svd::precision::{Precision, TwoFloatPrecision};

/// Convert f64 matrix to TwoFloatPrecision matrix
fn to_twofloat_matrix(matrix: &ndarray::Array2<f64>) -> ndarray::Array2<TwoFloatPrecision> {
    let (m, n) = matrix.dim();
    let mut result = ndarray::Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = TwoFloatPrecision::from_f64(matrix[[i, j]]);
        }
    }
    result
}

/// Convert TwoFloatPrecision matrix back to f64 for comparison
fn to_f64_matrix(matrix: &ndarray::Array2<TwoFloatPrecision>) -> ndarray::Array2<f64> {
    let (m, n) = matrix.dim();
    let mut result = ndarray::Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = matrix[[i, j]].to_f64();
        }
    }
    result
}

/// Template function for Jacobi SVD identity matrix test
fn test_jacobi_svd_identity_template<T: Precision + 'static>() 
where 
    T: From<f64> + Into<f64>,
{
    // Create identity matrix directly in target precision
    let mut a = ndarray::Array2::zeros((3, 3));
    for i in 0..3 {
        a[[i, i]] = <T as From<f64>>::from(1.0);
    }
    
    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);
    
    // Identity matrix should have singular values all equal to 1
    let epsilon = <T as Precision>::epsilon().into();
    for &s in result.s.iter() {
        let s_f64: f64 = s.into();
        assert_abs_diff_eq!(s_f64, 1.0, epsilon = 100.0 * epsilon);
    }
    
    // U and V should be identity matrices (or permutations of identity)
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let u_val: f64 = result.u[[i, j]].into();
            let v_val: f64 = result.v[[i, j]].into();
            assert_abs_diff_eq!(u_val.abs(), expected, epsilon = 100.0 * epsilon);
            assert_abs_diff_eq!(v_val.abs(), expected, epsilon = 100.0 * epsilon);
        }
    }
}

#[test]
fn test_jacobi_svd_identity_f64() {
    test_jacobi_svd_identity_template::<f64>();
}

#[test]
fn test_jacobi_svd_identity_twofloat() {
    test_jacobi_svd_identity_template::<TwoFloatPrecision>();
}

/// Template function for Jacobi SVD rank one matrix test
fn test_jacobi_svd_rank_one_template<T: Precision + 'static>() 
where 
    T: From<f64> + Into<f64>,
{
    // Create rank-one matrix directly in target precision
    let mut a = ndarray::Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            a[[i, j]] = <T as From<f64>>::from(1.0);
        }
    }
    
    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);
    
    // Should have only one non-zero singular value (around 3.0)
    let epsilon = <T as Precision>::epsilon().into();
    let s0: f64 = result.s[0].into();
    assert!(s0 > 1.0_f64);
    let s1: f64 = result.s[1].into();
    let s2: f64 = result.s[2].into();
    assert_abs_diff_eq!(s1, 0.0, epsilon = 100.0 * epsilon);
    assert_abs_diff_eq!(s2, 0.0, epsilon = 100.0 * epsilon);
}

#[test]
fn test_jacobi_svd_rank_one_f64() {
    test_jacobi_svd_rank_one_template::<f64>();
}

#[test]
fn test_jacobi_svd_rank_one_twofloat() {
    test_jacobi_svd_rank_one_template::<TwoFloatPrecision>();
}

/// Template function for Jacobi SVD 2x2 rank one matrix test
fn test_jacobi_svd_2x2_rank_one_template<T: Precision + 'static>() 
where 
    T: From<f64> + Into<f64> + AbsDiffEq<Epsilon = T>,
{
    // Create 2x2 rank-one matrix directly in target precision
    let mut a = ndarray::Array2::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            a[[i, j]] = <T as From<f64>>::from(1.0);
        }
    }
    
    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);
    
    // First singular value should be around 2.0
    let epsilon = <T as Precision>::epsilon().into();
    let s0: f64 = result.s[0].into();
    assert_abs_diff_eq!(s0, 2.0, epsilon = 100.0 * epsilon);
    
    // Second singular value should be zero
    let s1: f64 = result.s[1].into();
    assert_abs_diff_eq!(s1, 0.0, epsilon = 100.0 * epsilon);
    
    // Check reconstruction: A = U * S * V^T
    let s_diag = Array2::from_diag(&result.s);
    let reconstructed = result.u.dot(&s_diag).dot(&result.v.t());
    
    for i in 0..2 {
        for j in 0..2 {
            let orig_val = <T as From<f64>>::from(1.0); // All elements are 1.0
            let recon_val = reconstructed[[i, j]];
            // Use T's epsilon for comparison
            let tolerance = <T as Precision>::epsilon() * <T as From<f64>>::from(100.0);
            assert!(recon_val.abs_diff_eq(&orig_val, tolerance));
        }
    }
}

#[test]
fn test_jacobi_svd_2x2_rank_one_f64() {
    test_jacobi_svd_2x2_rank_one_template::<f64>();
}

#[test]
fn test_jacobi_svd_2x2_rank_one_twofloat() {
    test_jacobi_svd_2x2_rank_one_template::<TwoFloatPrecision>();
}

/// Template function for Jacobi SVD diagonal matrix test
fn test_jacobi_svd_diagonal_template<T: Precision + 'static>() 
where 
    T: From<f64> + Into<f64> + AbsDiffEq<Epsilon = T>,
{
    // Create diagonal matrix directly in target precision
    let mut a = ndarray::Array2::zeros((3, 3));
    a[[0, 0]] = <T as From<f64>>::from(3.0);
    a[[1, 1]] = <T as From<f64>>::from(2.0);
    a[[2, 2]] = <T as From<f64>>::from(1.0);
    
    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);
    
    // Singular values should be [3, 2, 1] in descending order
    let epsilon = <T as Precision>::epsilon().into();
    let s0: f64 = result.s[0].into();
    let s1: f64 = result.s[1].into();
    let s2: f64 = result.s[2].into();
    assert_abs_diff_eq!(s0, 3.0, epsilon = 100.0 * epsilon);
    assert_abs_diff_eq!(s1, 2.0, epsilon = 100.0 * epsilon);
    assert_abs_diff_eq!(s2, 1.0, epsilon = 100.0 * epsilon);
    
    // Check reconstruction: A = U * S * V^T
    let s_diag = Array2::from_diag(&result.s);
    let reconstructed = result.u.dot(&s_diag).dot(&result.v.t());
    
    for i in 0..3 {
        for j in 0..3 {
            let orig_val = if i == j {
                match i {
                    0 => <T as From<f64>>::from(3.0),
                    1 => <T as From<f64>>::from(2.0),
                    2 => <T as From<f64>>::from(1.0),
                    _ => <T as From<f64>>::from(0.0),
                }
            } else {
                <T as From<f64>>::from(0.0)
            };
            let recon_val = reconstructed[[i, j]];
            // Use T's epsilon for comparison
            let tolerance = <T as Precision>::epsilon() * <T as From<f64>>::from(100.0);
            assert!(recon_val.abs_diff_eq(&orig_val, tolerance));
        }
    }
}

#[test]
fn test_jacobi_svd_diagonal_f64() {
    test_jacobi_svd_diagonal_template::<f64>();
}

#[test]
fn test_jacobi_svd_diagonal_twofloat() {
    test_jacobi_svd_diagonal_template::<TwoFloatPrecision>();
}


/// Template function for Jacobi SVD orthogonality test
fn test_jacobi_svd_orthogonality_template<T: Precision + 'static>() 
where 
    T: From<f64> + Into<f64>,
{
    // Create test matrix directly in target precision
    let mut a = ndarray::Array2::zeros((3, 3));
    let values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    for i in 0..3 {
        for j in 0..3 {
            a[[i, j]] = <T as From<f64>>::from(values[i][j]);
        }
    }
    
    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);
    
    let epsilon = <T as Precision>::epsilon().into();
    
    // Check U^T * U = I
    let utu = result.u.t().dot(&result.u);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let utu_val: f64 = utu[[i, j]].into();
            assert_abs_diff_eq!(utu_val, expected, epsilon = 100.0 * epsilon);
        }
    }
    
    // Check V^T * V = I
    let vtv = result.v.t().dot(&result.v);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let vtv_val: f64 = vtv[[i, j]].into();
            assert_abs_diff_eq!(vtv_val, expected, epsilon = 100.0 * epsilon);
        }
    }
}

#[test]
fn test_jacobi_svd_orthogonality_f64() {
    test_jacobi_svd_orthogonality_template::<f64>();
}

#[test]
fn test_jacobi_svd_orthogonality_twofloat() {
    test_jacobi_svd_orthogonality_template::<TwoFloatPrecision>();
}

/// Template function for Jacobi SVD zero matrix test
fn test_jacobi_svd_zero_matrix_template<T: Precision + 'static>() 
where 
    T: From<f64> + Into<f64>,
{
    // Create zero matrix directly in target precision
    let a = ndarray::Array2::zeros((3, 3));
    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);
    
    let epsilon = <T as Precision>::epsilon().into();
    
    // All singular values should be zero
    for i in 0..3 {
        let s_val: f64 = result.s[i].into();
        assert_abs_diff_eq!(s_val, 0.0, epsilon = 100.0 * epsilon);
    }
}

#[test]
fn test_jacobi_svd_zero_matrix_f64() {
    test_jacobi_svd_zero_matrix_template::<f64>();
}

#[test]
fn test_jacobi_svd_zero_matrix_twofloat() {
    test_jacobi_svd_zero_matrix_template::<TwoFloatPrecision>();
}

/// Template function for Jacobi SVD single element test
fn test_jacobi_svd_single_element_template<T: Precision + 'static>() 
where 
    T: From<f64> + Into<f64>,
{
    // Create 1x1 matrix directly in target precision
    let mut a = ndarray::Array2::zeros((1, 1));
    a[[0, 0]] = <T as From<f64>>::from(5.0);
    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);
    
    let epsilon = <T as Precision>::epsilon().into();
    
    // Singular value should be 5.0
    let s_val: f64 = result.s[0].into();
    assert_abs_diff_eq!(s_val, 5.0, epsilon = 100.0 * epsilon);
    
    // U and V should be [[1]] or [[-1]]
    let u_val: f64 = <T as Precision>::abs(result.u[[0, 0]]).into();
    let v_val: f64 = <T as Precision>::abs(result.v[[0, 0]]).into();
    assert_abs_diff_eq!(u_val, 1.0, epsilon = 100.0 * epsilon);
    assert_abs_diff_eq!(v_val, 1.0, epsilon = 100.0 * epsilon);
}

#[test]
fn test_jacobi_svd_single_element_f64() {
    test_jacobi_svd_single_element_template::<f64>();
}

#[test]
fn test_jacobi_svd_single_element_twofloat() {
    test_jacobi_svd_single_element_template::<TwoFloatPrecision>();
}

/// Template function for Jacobi SVD singular values positive test
fn test_jacobi_svd_singular_values_positive_template<T: Precision + 'static>() 
where 
    T: From<f64> + Into<f64> + AbsDiffEq<Epsilon = T>,
{
    // Create test matrix directly in target precision
    let mut a = ndarray::Array2::zeros((3, 2));
    let values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    for i in 0..3 {
        for j in 0..2 {
            a[[i, j]] = <T as From<f64>>::from(values[i][j]);
        }
    }
    
    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);
    
    let epsilon = <T as Precision>::epsilon().into();
    
    // Check that singular values are positive and in descending order
    let s0: f64 = result.s[0].into();
    let s1: f64 = result.s[1].into();
    assert!(s0 >= s1);
    for i in 0..2 {
        let s_val: f64 = result.s[i].into();
        assert!(s_val >= 0.0);
    }
    
    // Check reconstruction: A = U * S * V^T
    let s_diag = Array2::from_diag(&result.s);
    let reconstructed = result.u.dot(&s_diag).dot(&result.v.t());
    
    for i in 0..3 {
        for j in 0..2 {
            let orig_val = a[[i, j]];
            let recon_val = reconstructed[[i, j]];
            // Use T's epsilon for comparison
            let tolerance = <T as Precision>::epsilon() * <T as From<f64>>::from(100.0);
            assert!(recon_val.abs_diff_eq(&orig_val, tolerance));
        }
    }
}

#[test]
fn test_jacobi_svd_singular_values_positive_f64() {
    test_jacobi_svd_singular_values_positive_template::<f64>();
}

#[test]
fn test_jacobi_svd_singular_values_positive_twofloat() {
    test_jacobi_svd_singular_values_positive_template::<TwoFloatPrecision>();
}

/// Template function for Jacobi SVD reconstruction test
fn test_jacobi_svd_reconstruction_template<T: Precision + 'static>() 
where 
    T: From<f64> + Into<f64>,
{
    let epsilon = <T as Precision>::epsilon().into();
    
    // Test 1: General 3x3 matrix
    let mut a1 = ndarray::Array2::zeros((3, 3));
    let values1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    for i in 0..3 {
        for j in 0..3 {
            a1[[i, j]] = <T as From<f64>>::from(values1[i][j]);
        }
    }
    let result1 = jacobi_svd(&a1);
    let s_diag1 = Array2::from_diag(&result1.s);
    let reconstructed1 = result1.u.dot(&s_diag1).dot(&result1.v.t());
    
    for i in 0..3 {
        for j in 0..3 {
            let orig_val: f64 = a1[[i, j]].into();
            let recon_val: f64 = reconstructed1[[i, j]].into();
            assert_abs_diff_eq!(recon_val, orig_val, epsilon = 100.0 * epsilon);
        }
    }
    
    // Test 2: Symmetric matrix
    let mut a2 = ndarray::Array2::zeros((3, 3));
    let values2 = [[4.0, 2.0, 1.0], [2.0, 3.0, 1.0], [1.0, 1.0, 2.0]];
    for i in 0..3 {
        for j in 0..3 {
            a2[[i, j]] = <T as From<f64>>::from(values2[i][j]);
        }
    }
    let result2 = jacobi_svd(&a2);
    let s_diag2 = Array2::from_diag(&result2.s);
    let reconstructed2 = result2.u.dot(&s_diag2).dot(&result2.v.t());
    
    for i in 0..3 {
        for j in 0..3 {
            let orig_val: f64 = a2[[i, j]].into();
            let recon_val: f64 = reconstructed2[[i, j]].into();
            assert_abs_diff_eq!(recon_val, orig_val, epsilon = 100.0 * epsilon);
        }
    }
    
    // Test 3: Off-diagonal 2x2 matrix
    let mut a3 = ndarray::Array2::zeros((2, 2));
    let values3 = [[0.0, 1.0], [1.0, 0.0]];
    for i in 0..2 {
        for j in 0..2 {
            a3[[i, j]] = <T as From<f64>>::from(values3[i][j]);
        }
    }
    let result3 = jacobi_svd(&a3);
    let s_diag3 = Array2::from_diag(&result3.s);
    let reconstructed3 = result3.u.dot(&s_diag3).dot(&result3.v.t());
    
    for i in 0..2 {
        for j in 0..2 {
            let orig_val: f64 = a3[[i, j]].into();
            let recon_val: f64 = reconstructed3[[i, j]].into();
            assert_abs_diff_eq!(recon_val, orig_val, epsilon = 100.0 * epsilon);
        }
    }
}

#[test]
fn test_jacobi_svd_reconstruction_f64() {
    test_jacobi_svd_reconstruction_template::<f64>();
}

#[test]
fn test_jacobi_svd_reconstruction_twofloat() {
    test_jacobi_svd_reconstruction_template::<TwoFloatPrecision>();
}

/// Template function for Jacobi SVD rectangular 3x2 test
fn test_jacobi_svd_rectangular_3x2_template<T: Precision + 'static>() 
where 
    T: From<f64> + Into<f64> + AbsDiffEq<Epsilon = T>,
{
    // Create 3x2 rectangular matrix directly in target precision
    let mut a = ndarray::Array2::zeros((3, 2));
    let values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    for i in 0..3 {
        for j in 0..2 {
            a[[i, j]] = <T as From<f64>>::from(values[i][j]);
        }
    }
    
    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);
    
    let epsilon = <T as Precision>::epsilon().into();
    
    // Check that singular values are in descending order
    let s0: f64 = result.s[0].into();
    let s1: f64 = result.s[1].into();
    assert!(s0 >= s1);
    
    // Check that singular values are positive
    for i in 0..2 {
        let s_val: f64 = result.s[i].into();
        assert!(s_val >= 0.0);
    }
    
    // Check reconstruction: A = U * S * V^T
    let s_diag = Array2::from_diag(&result.s);
    let reconstructed = result.u.dot(&s_diag).dot(&result.v.t());
    
    for i in 0..3 {
        for j in 0..2 {
            let orig_val = a[[i, j]];
            let recon_val = reconstructed[[i, j]];
            // Use T's epsilon for comparison
            let tolerance = <T as Precision>::epsilon() * <T as From<f64>>::from(100.0);
            assert!(recon_val.abs_diff_eq(&orig_val, tolerance));
        }
    }
    
    // Check orthogonality: U^T * U = I
    let utu = result.u.t().dot(&result.u);
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let utu_val: f64 = utu[[i, j]].into();
            assert_abs_diff_eq!(utu_val, expected, epsilon = 100.0 * epsilon);
        }
    }
    
    // Check orthogonality: V^T * V = I
    let vtv = result.v.t().dot(&result.v);
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let vtv_val: f64 = vtv[[i, j]].into();
            assert_abs_diff_eq!(vtv_val, expected, epsilon = 100.0 * epsilon);
        }
    }
}

#[test]
fn test_jacobi_svd_rectangular_3x2_f64() {
    test_jacobi_svd_rectangular_3x2_template::<f64>();
}

#[test]
fn test_jacobi_svd_rectangular_3x2_twofloat() {
    test_jacobi_svd_rectangular_3x2_template::<TwoFloatPrecision>();
}

/// Template function for Jacobi SVD rectangular 4x2 test
fn test_jacobi_svd_rectangular_4x2_template<T: Precision + 'static>() 
where 
    T: From<f64> + Into<f64> + AbsDiffEq<Epsilon = T>,
{
    // Create 4x2 rectangular matrix directly in target precision
    let mut a = ndarray::Array2::zeros((4, 2));
    let values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    for i in 0..4 {
        for j in 0..2 {
            a[[i, j]] = <T as From<f64>>::from(values[i][j]);
        }
    }
    
    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);
    
    let epsilon = <T as Precision>::epsilon().into();
    
    // Check reconstruction
    let s_diag = Array2::from_diag(&result.s);
    let reconstructed = result.u.dot(&s_diag).dot(&result.v.t());
    
    for i in 0..4 {
        for j in 0..2 {
            let orig_val = a[[i, j]];
            let recon_val = reconstructed[[i, j]];
            // Use T's epsilon for comparison
            let tolerance = <T as Precision>::epsilon() * <T as From<f64>>::from(100.0);
            assert!(recon_val.abs_diff_eq(&orig_val, tolerance));
        }
    }
}

#[test]
fn test_jacobi_svd_rectangular_4x2_f64() {
    test_jacobi_svd_rectangular_4x2_template::<f64>();
}

#[test]
fn test_jacobi_svd_rectangular_4x2_twofloat() {
    test_jacobi_svd_rectangular_4x2_template::<TwoFloatPrecision>();
}








