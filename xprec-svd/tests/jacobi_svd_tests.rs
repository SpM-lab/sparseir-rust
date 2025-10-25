use approx::{AbsDiffEq, assert_abs_diff_eq};
use mdarray::Tensor;
use mdarray_linalg::matmul::{MatMul, MatMulBuilder};
use mdarray_linalg_faer::Faer;
use xprec_svd::precision::{Df64Precision, Precision};
use xprec_svd::svd::jacobi::jacobi_svd;

/// Convert f64 matrix to Df64Precision matrix
fn to_df64_matrix(matrix: &Tensor<f64, (usize, usize)>) -> Tensor<Df64Precision, (usize, usize)> {
    let shape = *matrix.shape();
    Tensor::from_fn(shape, |idx| {
        Df64Precision::from_f64(matrix[[idx[0], idx[1]]])
    })
}

/// Convert Df64Precision matrix back to f64 for comparison
fn to_f64_matrix(matrix: &Tensor<Df64Precision, (usize, usize)>) -> Tensor<f64, (usize, usize)> {
    let shape = *matrix.shape();
    Tensor::from_fn(shape, |idx| matrix[[idx[0], idx[1]]].to_f64())
}

/// Create diagonal matrix from vector
fn diag_matrix<T: Precision>(v: &Tensor<T, (usize,)>) -> Tensor<T, (usize, usize)> {
    let n = v.len();
    Tensor::from_fn((n, n), |idx| {
        if idx[0] == idx[1] {
            v[[idx[0]]]
        } else {
            T::zero()
        }
    })
}

/// Matrix multiplication: C = A * B (using Faer backend)
fn matmul_f64(
    a: &Tensor<f64, (usize, usize)>,
    b: &Tensor<f64, (usize, usize)>,
) -> Tensor<f64, (usize, usize)> {
    Faer.matmul(a, b).eval()
}

/// Matrix multiplication for generic types (fallback)
fn matmul<T: Precision>(
    a: &Tensor<T, (usize, usize)>,
    b: &Tensor<T, (usize, usize)>,
) -> Tensor<T, (usize, usize)> {
    let a_shape = *a.shape();
    let b_shape = *b.shape();
    assert_eq!(a_shape.1, b_shape.0);

    Tensor::from_fn((a_shape.0, b_shape.1), |idx| {
        let mut sum = T::zero();
        for k in 0..a_shape.1 {
            sum = sum + a[[idx[0], k]] * b[[k, idx[1]]];
        }
        sum
    })
}

/// Matrix transpose
fn transpose<T: Precision>(m: &Tensor<T, (usize, usize)>) -> Tensor<T, (usize, usize)> {
    let shape = *m.shape();
    Tensor::from_fn((shape.1, shape.0), |idx| m[[idx[1], idx[0]]])
}

/// Reconstruct matrix from SVD: A = U * S * V^T (for f64 using Faer)
fn reconstruct_svd_f64(
    u: &Tensor<f64, (usize, usize)>,
    s: &Tensor<f64, (usize,)>,
    v: &Tensor<f64, (usize, usize)>,
) -> Tensor<f64, (usize, usize)> {
    // U * diag(S)
    let u_shape = *u.shape();
    let us = Tensor::from_fn(u_shape, |idx| u[[idx[0], idx[1]]] * s[[idx[1]]]);
    // (U*S) * V^T
    let v_t = transpose(v);
    matmul_f64(&us, &v_t)
}

/// Reconstruct matrix from SVD: A = U * S * V^T (generic version)
/// Note: v is V matrix (not V^T), so we need v[[idx[1], k]] for V^T[k,j] = V[j,k]
fn reconstruct_svd<T: Precision>(
    u: &Tensor<T, (usize, usize)>,
    s: &Tensor<T, (usize,)>,
    v: &Tensor<T, (usize, usize)>,
) -> Tensor<T, (usize, usize)> {
    let u_shape = *u.shape();
    let v_shape = *v.shape();

    Tensor::from_fn((u_shape.0, v_shape.0), |idx| {
        let mut sum = T::zero();
        for k in 0..s.len() {
            // A[i,j] = Σ_k U[i,k] * S[k] * V^T[k,j] = Σ_k U[i,k] * S[k] * V[j,k]
            sum = sum + u[[idx[0], k]] * s[[k]] * v[[idx[1], k]];
        }
        sum
    })
}

/// Template function for Jacobi SVD identity matrix test
fn test_jacobi_svd_identity_template<T: Precision + 'static>()
where
    T: From<f64> + Into<f64>,
{
    // Create identity matrix directly in target precision
    let a = Tensor::from_fn((3, 3), |idx| {
        if idx[0] == idx[1] {
            <T as From<f64>>::from(1.0)
        } else {
            <T as From<f64>>::from(0.0)
        }
    });

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
fn test_jacobi_svd_identity_df64() {
    test_jacobi_svd_identity_template::<Df64Precision>();
}

/// Template function for Jacobi SVD rank one matrix test
fn test_jacobi_svd_rank_one_template<T: Precision + 'static>()
where
    T: From<f64> + Into<f64>,
{
    // Create rank-one matrix directly in target precision
    let a = Tensor::from_fn((3, 3), |_idx| <T as From<f64>>::from(1.0));

    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);

    // Should have only one non-zero singular value (around 3.0)
    let epsilon = <T as Precision>::epsilon().into();
    let s0: f64 = result.s[[0]].into();
    assert!(s0 > 1.0_f64);
    let s1: f64 = result.s[[1]].into();
    let s2: f64 = result.s[[2]].into();
    assert_abs_diff_eq!(s1, 0.0, epsilon = 100.0 * epsilon);
    assert_abs_diff_eq!(s2, 0.0, epsilon = 100.0 * epsilon);
}

#[test]
fn test_jacobi_svd_rank_one_f64() {
    test_jacobi_svd_rank_one_template::<f64>();
}

#[test]
fn test_jacobi_svd_rank_one_df64() {
    test_jacobi_svd_rank_one_template::<Df64Precision>();
}

/// Template function for Jacobi SVD 2x2 rank one matrix test
fn test_jacobi_svd_2x2_rank_one_template<T: Precision + 'static>()
where
    T: From<f64> + Into<f64> + AbsDiffEq<Epsilon = T>,
{
    // Create 2x2 rank-one matrix directly in target precision
    let mut a = Tensor::from_elem((2, 2), T::zero());
    for i in 0..2 {
        for j in 0..2 {
            a[[i, j]] = <T as From<f64>>::from(1.0);
        }
    }

    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);

    // First singular value should be around 2.0
    let epsilon = <T as Precision>::epsilon().into();
    let s0: f64 = result.s[[0]].into();
    assert_abs_diff_eq!(s0, 2.0, epsilon = 100.0 * epsilon);

    // Second singular value should be zero
    let s1: f64 = result.s[[1]].into();
    assert_abs_diff_eq!(s1, 0.0, epsilon = 100.0 * epsilon);

    // Check reconstruction: A = U * S * V^T

    let reconstructed = reconstruct_svd(&result.u, &result.s, &result.v);

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
fn test_jacobi_svd_2x2_rank_one_df64() {
    test_jacobi_svd_2x2_rank_one_template::<Df64Precision>();
}

/// Template function for Jacobi SVD diagonal matrix test
fn test_jacobi_svd_diagonal_template<T: Precision + 'static>()
where
    T: From<f64> + Into<f64> + AbsDiffEq<Epsilon = T>,
{
    // Create diagonal matrix directly in target precision
    let mut a = Tensor::from_elem((3, 3), T::zero());
    a[[0, 0]] = <T as From<f64>>::from(3.0);
    a[[1, 1]] = <T as From<f64>>::from(2.0);
    a[[2, 2]] = <T as From<f64>>::from(1.0);

    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);

    // Singular values should be [3, 2, 1] in descending order
    let epsilon = <T as Precision>::epsilon().into();
    let s0: f64 = result.s[[0]].into();
    let s1: f64 = result.s[[1]].into();
    let s2: f64 = result.s[[2]].into();
    assert_abs_diff_eq!(s0, 3.0, epsilon = 100.0 * epsilon);
    assert_abs_diff_eq!(s1, 2.0, epsilon = 100.0 * epsilon);
    assert_abs_diff_eq!(s2, 1.0, epsilon = 100.0 * epsilon);

    // Check reconstruction: A = U * S * V^T

    let reconstructed = reconstruct_svd(&result.u, &result.s, &result.v);

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
fn test_jacobi_svd_diagonal_df64() {
    test_jacobi_svd_diagonal_template::<Df64Precision>();
}

/// Template function for Jacobi SVD orthogonality test
fn test_jacobi_svd_orthogonality_template<T: Precision + 'static>()
where
    T: From<f64> + Into<f64>,
{
    // Create test matrix directly in target precision
    let mut a = Tensor::from_elem((3, 3), T::zero());
    let values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    for i in 0..3 {
        for j in 0..3 {
            a[[i, j]] = <T as From<f64>>::from(values[i][j]);
        }
    }

    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);

    let epsilon = <T as Precision>::epsilon().into();

    // Check U^T * U = I
    let utu = matmul(&transpose(&result.u), &result.u);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let utu_val: f64 = utu[[i, j]].into();
            assert_abs_diff_eq!(utu_val, expected, epsilon = 100.0 * epsilon);
        }
    }

    // Check V^T * V = I
    let vtv = matmul(&transpose(&result.v), &result.v);
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
fn test_jacobi_svd_orthogonality_df64() {
    test_jacobi_svd_orthogonality_template::<Df64Precision>();
}

/// Template function for Jacobi SVD zero matrix test
fn test_jacobi_svd_zero_matrix_template<T: Precision + 'static>()
where
    T: From<f64> + Into<f64>,
{
    // Create zero matrix directly in target precision
    let a = Tensor::from_elem((3, 3), T::zero());
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
fn test_jacobi_svd_zero_matrix_df64() {
    test_jacobi_svd_zero_matrix_template::<Df64Precision>();
}

/// Template function for Jacobi SVD single element test
fn test_jacobi_svd_single_element_template<T: Precision + 'static>()
where
    T: From<f64> + Into<f64>,
{
    // Create 1x1 matrix directly in target precision
    let mut a = Tensor::from_elem((1, 1), T::zero());
    a[[0, 0]] = <T as From<f64>>::from(5.0);
    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);

    let epsilon = <T as Precision>::epsilon().into();

    // Singular value should be 5.0
    let s_val: f64 = result.s[[0]].into();
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
fn test_jacobi_svd_single_element_df64() {
    test_jacobi_svd_single_element_template::<Df64Precision>();
}

/// Template function for Jacobi SVD singular values positive test
fn test_jacobi_svd_singular_values_positive_template<T: Precision + 'static>()
where
    T: From<f64> + Into<f64> + AbsDiffEq<Epsilon = T>,
{
    // Create test matrix directly in target precision
    let mut a = Tensor::from_elem((3, 2), T::zero());
    let values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    for i in 0..3 {
        for j in 0..2 {
            a[[i, j]] = <T as From<f64>>::from(values[i][j]);
        }
    }

    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);

    let epsilon = <T as Precision>::epsilon().into();

    // Check that singular values are positive and in descending order
    let s0: f64 = result.s[[0]].into();
    let s1: f64 = result.s[[1]].into();
    assert!(s0 >= s1);
    for i in 0..2 {
        let s_val: f64 = result.s[i].into();
        assert!(s_val >= 0.0);
    }

    // Check reconstruction: A = U * S * V^T

    let reconstructed = reconstruct_svd(&result.u, &result.s, &result.v);

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
fn test_jacobi_svd_singular_values_positive_df64() {
    test_jacobi_svd_singular_values_positive_template::<Df64Precision>();
}

/// Template function for Jacobi SVD reconstruction test
fn test_jacobi_svd_reconstruction_template<T: Precision + 'static>()
where
    T: From<f64> + Into<f64>,
{
    let epsilon = <T as Precision>::epsilon().into();

    // Test 1: General 3x3 matrix
    let mut a1 = Tensor::from_elem((3, 3), T::zero());
    let values1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    for i in 0..3 {
        for j in 0..3 {
            a1[[i, j]] = <T as From<f64>>::from(values1[i][j]);
        }
    }
    let result1 = jacobi_svd(&a1);
    let reconstructed1 = reconstruct_svd(&result1.u, &result1.s, &result1.v);

    for i in 0..3 {
        for j in 0..3 {
            let orig_val: f64 = a1[[i, j]].into();
            let recon_val: f64 = reconstructed1[[i, j]].into();
            assert_abs_diff_eq!(recon_val, orig_val, epsilon = 100.0 * epsilon);
        }
    }

    // Test 2: Symmetric matrix
    let mut a2 = Tensor::from_elem((3, 3), T::zero());
    let values2 = [[4.0, 2.0, 1.0], [2.0, 3.0, 1.0], [1.0, 1.0, 2.0]];
    for i in 0..3 {
        for j in 0..3 {
            a2[[i, j]] = <T as From<f64>>::from(values2[i][j]);
        }
    }
    let result2 = jacobi_svd(&a2);
    let reconstructed2 = reconstruct_svd(&result2.u, &result2.s, &result2.v);

    for i in 0..3 {
        for j in 0..3 {
            let orig_val: f64 = a2[[i, j]].into();
            let recon_val: f64 = reconstructed2[[i, j]].into();
            assert_abs_diff_eq!(recon_val, orig_val, epsilon = 100.0 * epsilon);
        }
    }

    // Test 3: Off-diagonal 2x2 matrix
    let mut a3 = Tensor::from_elem((2, 2), T::zero());
    let values3 = [[0.0, 1.0], [1.0, 0.0]];
    for i in 0..2 {
        for j in 0..2 {
            a3[[i, j]] = <T as From<f64>>::from(values3[i][j]);
        }
    }
    let result3 = jacobi_svd(&a3);
    let reconstructed3 = reconstruct_svd(&result3.u, &result3.s, &result3.v);

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
fn test_jacobi_svd_reconstruction_df64() {
    test_jacobi_svd_reconstruction_template::<Df64Precision>();
}

/// Template function for Jacobi SVD rectangular 3x2 test
fn test_jacobi_svd_rectangular_3x2_template<T: Precision + 'static>()
where
    T: From<f64> + Into<f64> + AbsDiffEq<Epsilon = T>,
{
    // Create 3x2 rectangular matrix directly in target precision
    let mut a = Tensor::from_elem((3, 2), T::zero());
    let values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    for i in 0..3 {
        for j in 0..2 {
            a[[i, j]] = <T as From<f64>>::from(values[i][j]);
        }
    }

    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);

    let epsilon = <T as Precision>::epsilon().into();

    // Check that singular values are in descending order
    let s0: f64 = result.s[[0]].into();
    let s1: f64 = result.s[[1]].into();
    assert!(s0 >= s1);

    // Check that singular values are positive
    for i in 0..2 {
        let s_val: f64 = result.s[i].into();
        assert!(s_val >= 0.0);
    }

    // Check reconstruction: A = U * S * V^T

    let reconstructed = reconstruct_svd(&result.u, &result.s, &result.v);

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
    let utu = matmul(&transpose(&result.u), &result.u);
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let utu_val: f64 = utu[[i, j]].into();
            assert_abs_diff_eq!(utu_val, expected, epsilon = 100.0 * epsilon);
        }
    }

    // Check orthogonality: V^T * V = I
    let vtv = matmul(&transpose(&result.v), &result.v);
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
fn test_jacobi_svd_rectangular_3x2_df64() {
    test_jacobi_svd_rectangular_3x2_template::<Df64Precision>();
}

/// Template function for Jacobi SVD rectangular 4x2 test
fn test_jacobi_svd_rectangular_4x2_template<T: Precision + 'static>()
where
    T: From<f64> + Into<f64> + AbsDiffEq<Epsilon = T>,
{
    // Create 4x2 rectangular matrix directly in target precision
    let mut a = Tensor::from_elem((4, 2), T::zero());
    let values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    for i in 0..4 {
        for j in 0..2 {
            a[[i, j]] = <T as From<f64>>::from(values[i][j]);
        }
    }

    let result: xprec_svd::svd::SVDResult<T> = jacobi_svd(&a);

    let epsilon = <T as Precision>::epsilon().into();

    // Check reconstruction

    let reconstructed = reconstruct_svd(&result.u, &result.s, &result.v);

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
fn test_jacobi_svd_rectangular_4x2_df64() {
    test_jacobi_svd_rectangular_4x2_template::<Df64Precision>();
}
