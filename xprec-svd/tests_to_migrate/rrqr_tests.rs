//! Parameterized RRQR tests for f64 and TwoFloat precision

use xprec_svd::*;
use ndarray::array;
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

/// Template function for RRQR identity matrix test
fn test_rrqr_identity_matrix_template<T: Precision>() 
where 
    T: From<f64> + Into<f64>,
{
    let matrix_f64 = array![
        [1.0, 0.0],
        [0.0, 1.0]
    ];
    
    // Convert to target precision
    let (m, n) = matrix_f64.dim();
    let mut a = ndarray::Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            a[[i, j]] = <T as From<f64>>::from(matrix_f64[[i, j]]);
        }
    }
    
    let rtol = <T as Precision>::epsilon();
    let (qr, rank) = rrqr(&mut a, rtol);
    
    assert_eq!(rank, 2);
    assert_eq!(qr.jpvt[0], 0);
    assert_eq!(qr.jpvt[1], 1);
}

/// Template function for RRQR simple matrix test (f64 version)
fn test_rrqr_simple_matrix_template_f64() {
    let mut a = array![
        [2.0, 1.0],
        [4.0, 3.0]
    ];
    
    let (qr, rank) = rrqr(&mut a, f64::EPSILON);
    assert_eq!(rank, 2);
    
    // Test reconstruction: A * P = Q * R
    let (q, r) = truncate_qr_result(&qr, rank);
    
    // Create permutation matrix P
    let mut p: ndarray::Array2<f64> = ndarray::Array2::zeros((a.ncols(), a.ncols()));
    for (i, &j) in qr.jpvt.iter().enumerate() {
        p[[j, i]] = 1.0;
    }
    
    let a_original = array![
        [2.0, 1.0],
        [4.0, 3.0]
    ];
    
    let ap = a_original.dot(&p);
    let qr_reconstructed = q.dot(&r);
    
    let error = &ap - &qr_reconstructed;
    let max_error = error.iter().map(|x| x.abs()).fold(0.0, f64::max);
    
    assert!(max_error < 100.0 * f64::EPSILON, "Reconstruction error too large: {}", max_error);
}

/// Template function for RRQR simple matrix test (TwoFloat version)
fn test_rrqr_simple_matrix_template_twofloat() {
    let matrix_f64 = array![
        [2.0, 1.0],
        [4.0, 3.0]
    ];
    
    let mut a = to_twofloat_matrix(&matrix_f64);
    let rtol = TwoFloatPrecision::epsilon();
    
    let (qr, rank) = rrqr(&mut a, rtol);
    assert_eq!(rank, 2);
    
    // Test reconstruction: A * P = Q * R
    let (q, r) = truncate_qr_result(&qr, rank);
    
    // Create permutation matrix P
    let mut p: ndarray::Array2<f64> = ndarray::Array2::zeros((a.ncols(), a.ncols()));
    for (i, &j) in qr.jpvt.iter().enumerate() {
        p[[j, i]] = 1.0;
    }
    
    // Convert Q and R to f64 for reconstruction
    let q_f64 = to_f64_matrix(&q);
    let r_f64 = to_f64_matrix(&r);
    
    let ap = matrix_f64.dot(&p);
    let qr_reconstructed = q_f64.dot(&r_f64);
    
    let error = &ap - &qr_reconstructed;
    let max_error = error.iter().map(|x| x.abs()).fold(0.0, f64::max);
    
    assert!(max_error < 100.0 * f64::EPSILON, "Reconstruction error too large: {}", max_error);
}

/// Template function for RRQR rank deficient matrix test
fn test_rrqr_rank_deficient_template<T: Precision>() 
where 
    T: From<f64> + Into<f64>,
{
    let matrix_f64 = array![
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ];
    
    // Convert to target precision
    let (m, n) = matrix_f64.dim();
    let mut a = ndarray::Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            a[[i, j]] = <T as From<f64>>::from(matrix_f64[[i, j]]);
        }
    }
    
    let rtol = <T as Precision>::epsilon();
    let (_qr, rank) = rrqr(&mut a, rtol);
    
    assert_eq!(rank, 1);
}

/// Template function for RRQR zero matrix test
fn test_rrqr_zero_matrix_template<T: Precision>() 
where 
    T: From<f64> + Into<f64>,
{
    let matrix_f64 = array![
        [0.0, 0.0],
        [0.0, 0.0]
    ];
    
    // Convert to target precision
    let (m, n) = matrix_f64.dim();
    let mut a = ndarray::Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            a[[i, j]] = <T as From<f64>>::from(matrix_f64[[i, j]]);
        }
    }
    
    let rtol = <T as Precision>::epsilon();
    let (_qr, rank) = rrqr(&mut a, rtol);
    
    assert_eq!(rank, 0);
}

/// Template function for RRQR single element test
fn test_rrqr_single_element_template<T: Precision>() 
where 
    T: From<f64> + Into<f64>,
{
    let matrix_f64 = array![[5.0]];
    
    // Convert to target precision
    let (m, n) = matrix_f64.dim();
    let mut a = ndarray::Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            a[[i, j]] = <T as From<f64>>::from(matrix_f64[[i, j]]);
        }
    }
    
    let rtol = <T as Precision>::epsilon();
    let (qr, rank) = rrqr(&mut a, rtol);
    
    assert_eq!(rank, 1);
    assert_eq!(qr.jpvt[0], 0);
}

/// Template function for RRQR orthogonal Q test (f64 version)
fn test_rrqr_orthogonal_q_template_f64() {
    let mut a = array![
        [3.0, 4.0],
        [4.0, 3.0]
    ];
    
    let (qr, rank) = rrqr(&mut a, f64::EPSILON);
    let (q, _r) = truncate_qr_result(&qr, rank);
    
    // Q should be orthogonal: Q^T * Q = I
    let qtq = q.t().dot(&q);
    let identity: ndarray::Array2<f64> = ndarray::Array2::eye(q.ncols());
    
    let error = &qtq - &identity;
    let max_error = error.iter().map(|x| x.abs()).fold(0.0, f64::max);
    
    assert!(max_error < 1e-10, "Q is not orthogonal, max error: {}", max_error);
}

/// Template function for RRQR orthogonal Q test (TwoFloat version)
fn test_rrqr_orthogonal_q_template_twofloat() {
    let matrix_f64 = array![
        [3.0, 4.0],
        [4.0, 3.0]
    ];
    
    let mut a = to_twofloat_matrix(&matrix_f64);
    let rtol = TwoFloatPrecision::epsilon();
    
    let (qr, rank) = rrqr(&mut a, rtol);
    let (q, _r) = truncate_qr_result(&qr, rank);
    
    // Convert Q to f64 for orthogonality check
    let q_f64 = to_f64_matrix(&q);
    
    // Q should be orthogonal: Q^T * Q = I
    let qtq = q_f64.t().dot(&q_f64);
    let identity: ndarray::Array2<f64> = ndarray::Array2::eye(q_f64.ncols());
    
    let error = &qtq - &identity;
    let max_error = error.iter().map(|x| x.abs()).fold(0.0, f64::max);
    
    assert!(max_error < 1e-10, "Q is not orthogonal, max error: {}", max_error);
}

// ===== f64 Tests =====

#[test]
fn test_rrqr_identity_matrix_f64() {
    test_rrqr_identity_matrix_template::<f64>();
}

#[test]
fn test_rrqr_simple_matrix_f64() {
    test_rrqr_simple_matrix_template_f64();
}

#[test]
fn test_rrqr_rank_deficient_f64() {
    test_rrqr_rank_deficient_template::<f64>();
}

#[test]
fn test_rrqr_zero_matrix_f64() {
    test_rrqr_zero_matrix_template::<f64>();
}

#[test]
fn test_rrqr_single_element_f64() {
    test_rrqr_single_element_template::<f64>();
}

#[test]
fn test_rrqr_orthogonal_q_f64() {
    test_rrqr_orthogonal_q_template_f64();
}

// ===== TwoFloat Tests =====

#[test]
fn test_rrqr_identity_matrix_twofloat() {
    test_rrqr_identity_matrix_template::<TwoFloatPrecision>();
}

#[test]
fn test_rrqr_simple_matrix_twofloat() {
    test_rrqr_simple_matrix_template_twofloat();
}

#[test]
fn test_rrqr_rank_deficient_twofloat() {
    test_rrqr_rank_deficient_template::<TwoFloatPrecision>();
}

#[test]
fn test_rrqr_zero_matrix_twofloat() {
    test_rrqr_zero_matrix_template::<TwoFloatPrecision>();
}

#[test]
fn test_rrqr_single_element_twofloat() {
    test_rrqr_single_element_template::<TwoFloatPrecision>();
}

#[test]
fn test_rrqr_orthogonal_q_twofloat() {
    test_rrqr_orthogonal_q_template_twofloat();
}

// ===== Comparison Tests =====

#[test]
fn test_rrqr_precision_comparison() {
    let matrix_f64 = array![
        [1e-10, 2e-10, 3e-10],
        [2e-10, 4e-10, 6e-10],
        [1e-15, 2e-15, 3e-15]  // Very small values
    ];
    
    // Test with f64
    let mut matrix_f64_copy = matrix_f64.clone();
    let (qr_f64, rank_f64) = rrqr(&mut matrix_f64_copy, f64::EPSILON);
    
    // Test with TwoFloat
    let mut matrix_tf = to_twofloat_matrix(&matrix_f64);
    let rtol_tf = TwoFloatPrecision::epsilon();
    let (qr_tf, rank_tf) = rrqr(&mut matrix_tf, rtol_tf);
    
    // Both should give same rank
    assert_eq!(rank_f64, rank_tf);
    
    // Compare Q matrices (should be very close)
    let (q_f64, _) = truncate_qr_result(&qr_f64, rank_f64);
    let (q_tf, _) = truncate_qr_result(&qr_tf, rank_tf);
    let q_tf_f64 = to_f64_matrix(&q_tf);
    
    for i in 0..q_f64.nrows() {
        for j in 0..q_f64.ncols() {
            let diff = (q_f64[[i, j]] - q_tf_f64[[i, j]]).abs();
            // TwoFloat should be more accurate, so difference should be small
            assert!(diff < 100.0 * f64::EPSILON, "Q matrix difference at [{}, {}]: {}", i, j, diff);
        }
    }
}