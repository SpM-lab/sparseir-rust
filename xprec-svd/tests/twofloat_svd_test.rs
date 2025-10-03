use xprec_svd::precision::TwoFloatPrecision;
use xprec_svd::{tsvd_twofloat, tsvd_twofloat_from_f64};
use ndarray::array;

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

#[test]
fn test_twofloat_svd_identity() {
    let matrix_f64 = array![
        [1.0, 0.0],
        [0.0, 1.0]
    ];
    
    // Test with f64 input (conversion)
    let result = tsvd_twofloat_from_f64(&matrix_f64, TwoFloatPrecision::epsilon().to_f64()).unwrap();
    
    assert_eq!(result.rank, 2);
    
    // Check singular values
    let s_f64: Vec<f64> = result.s.iter().map(|&x| x.to_f64()).collect();
    assert!((s_f64[0] - 1.0).abs() < 100.0 * f64::EPSILON);
    assert!((s_f64[1] - 1.0).abs() < 100.0 * f64::EPSILON);
    
    // Check orthogonality of U
    let u_f64 = to_f64_matrix(&result.u);
    let utu = u_f64.t().dot(&u_f64);
    let identity: ndarray::Array2<f64> = ndarray::Array2::eye(2);
    
    for i in 0..2 {
        for j in 0..2 {
            let diff = (utu[[i, j]] - identity[[i, j]]).abs();
            assert!(diff < 100.0 * f64::EPSILON, "U^T * U should be identity, got diff = {}", diff);
        }
    }
}

#[test]
fn test_twofloat_svd_rank_one() {
    let matrix_f64 = array![
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ];
    
    let result = tsvd_twofloat_from_f64(&matrix_f64, TwoFloatPrecision::epsilon().to_f64()).unwrap();
    
    assert_eq!(result.rank, 1);
    
    // Check that only one singular value is significant
    let s_f64: Vec<f64> = result.s.iter().map(|&x| x.to_f64()).collect();
    assert!(s_f64[0] > 1.0); // First singular value should be large
    
    // Check remaining singular values if they exist
    for i in 1..s_f64.len() {
        assert!(s_f64[i] < 100.0 * f64::EPSILON, "Singular value {} should be very small: {}", i, s_f64[i]);
    }
}

#[test]
fn test_twofloat_svd_precision_comparison() {
    // Test matrix with small values that might cause precision issues
    let matrix = array![
        [1e-10, 2e-10, 3e-10],
        [2e-10, 4e-10, 6e-10],
        [1e-15, 2e-15, 3e-15]  // Very small values
    ];
    
    // Test with f64
    let result_f64 = xprec_svd::tsvd_f64(&matrix, f64::EPSILON).unwrap();
    
    // Test with TwoFloat
    let result_tf = tsvd_twofloat_from_f64(&matrix, TwoFloatPrecision::epsilon().to_f64()).unwrap();
    
    // Both should give same rank
    assert_eq!(result_f64.rank, result_tf.rank);
    
    // Compare singular values
    let s_f64: Vec<f64> = result_f64.s.iter().map(|&x| x).collect();
    let s_tf: Vec<f64> = result_tf.s.iter().map(|&x| x.to_f64()).collect();
    
    for i in 0..s_f64.len() {
        let diff = (s_f64[i] - s_tf[i]).abs();
        // TwoFloat should be more accurate, so difference should be small
        assert!(diff < 100.0 * f64::EPSILON, "Singular value difference at {}: {}", i, diff);
    }
}

#[test]
fn test_twofloat_svd_reconstruction() {
    let matrix_f64 = array![
        [1.0, 2.0],
        [3.0, 4.0]
    ];
    
    let result = tsvd_twofloat_from_f64(&matrix_f64, TwoFloatPrecision::epsilon().to_f64()).unwrap();
    
    // Reconstruct matrix: A = U * S * V^T
    let u_f64 = to_f64_matrix(&result.u);
    let v_f64 = to_f64_matrix(&result.v);
    let s_f64: Vec<f64> = result.s.iter().map(|&x| x.to_f64()).collect();
    
    let mut reconstructed: ndarray::Array2<f64> = ndarray::Array2::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..result.rank {
                reconstructed[[i, j]] += u_f64[[i, k]] * s_f64[k] * v_f64[[j, k]];
            }
        }
    }
    
    // Check reconstruction accuracy
    for i in 0..2 {
        for j in 0..2 {
            let diff = (matrix_f64[[i, j]] - reconstructed[[i, j]]).abs();
            assert!(diff < 100.0 * f64::EPSILON, "Reconstruction error at [{}, {}]: {}", i, j, diff);
        }
    }
}

#[test]
fn test_twofloat_svd_direct_input() {
    // Test with direct TwoFloat matrix input
    let matrix_f64 = array![
        [1.0, 2.0],
        [3.0, 4.0]
    ];
    
    // Convert to TwoFloat matrix
    let matrix_tf = to_twofloat_matrix(&matrix_f64);
    let rtol_tf = TwoFloatPrecision::epsilon();
    
    // Test direct TwoFloat input
    let result = tsvd_twofloat(&matrix_tf, rtol_tf).unwrap();
    
    assert_eq!(result.rank, 2);
    
    // Check singular values
    let s_f64: Vec<f64> = result.s.iter().map(|&x| x.to_f64()).collect();
    assert!(s_f64[0] > s_f64[1]); // Singular values should be in descending order
    assert!(s_f64[0] > 0.0); // All singular values should be positive
    assert!(s_f64[1] > 0.0);
    
    // Check orthogonality of U and V
    let u_f64 = to_f64_matrix(&result.u);
    let v_f64 = to_f64_matrix(&result.v);
    
    // U^T * U should be identity
    let utu = u_f64.t().dot(&u_f64);
    let identity: ndarray::Array2<f64> = ndarray::Array2::eye(2);
    
    for i in 0..2 {
        for j in 0..2 {
            let diff = (utu[[i, j]] - identity[[i, j]]).abs();
            assert!(diff < 100.0 * f64::EPSILON, "U^T * U should be identity, got diff = {}", diff);
        }
    }
    
    // V^T * V should be identity
    let vtv = v_f64.t().dot(&v_f64);
    
    for i in 0..2 {
        for j in 0..2 {
            let diff = (vtv[[i, j]] - identity[[i, j]]).abs();
            assert!(diff < 100.0 * f64::EPSILON, "V^T * V should be identity, got diff = {}", diff);
        }
    }
}
