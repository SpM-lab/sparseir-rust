//! Parameterized RRQR tests for f64 and TwoFloat precision

use xprec_svd::*;
use mdarray::Tensor;
use xprec_svd::precision::{Precision, TwoFloatPrecision};

// ===== Helper Functions =====

/// Create identity matrix
fn eye<T: Copy + From<f64>>(n: usize) -> Tensor<T, (usize, usize)> {
    Tensor::from_fn((n, n), |idx| {
        if idx[0] == idx[1] {
            <T as From<f64>>::from(1.0)
        } else {
            <T as From<f64>>::from(0.0)
        }
    })
}

/// Matrix transpose
fn transpose<T: Copy>(matrix: &Tensor<T, (usize, usize)>) -> Tensor<T, (usize, usize)> {
    let (m, n) = *matrix.shape();
    Tensor::from_fn((n, m), |idx| matrix[[idx[1], idx[0]]])
}

/// Matrix multiplication
fn matmul<T>(a: &Tensor<T, (usize, usize)>, b: &Tensor<T, (usize, usize)>) -> Tensor<T, (usize, usize)>
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + From<f64>,
{
    let (m, k1) = *a.shape();
    let (k2, n) = *b.shape();
    assert_eq!(k1, k2, "Matrix dimensions incompatible for multiplication");
    
    Tensor::from_fn((m, n), |idx| {
        let i = idx[0];
        let j = idx[1];
        let mut sum = <T as From<f64>>::from(0.0);
        for k in 0..k1 {
            sum = sum + a[[i, k]] * b[[k, j]];
        }
        sum
    })
}

/// Convert f64 matrix to TwoFloatPrecision matrix
fn to_twofloat_matrix(matrix: &Tensor<f64, (usize, usize)>) -> Tensor<TwoFloatPrecision, (usize, usize)> {
    let (m, n) = *matrix.shape();
    Tensor::from_fn((m, n), |idx| {
        TwoFloatPrecision::from_f64(matrix[[idx[0], idx[1]]])
    })
}

/// Convert TwoFloatPrecision matrix back to f64 for comparison
fn to_f64_matrix(matrix: &Tensor<TwoFloatPrecision, (usize, usize)>) -> Tensor<f64, (usize, usize)> {
    let (m, n) = *matrix.shape();
    Tensor::from_fn((m, n), |idx| {
        matrix[[idx[0], idx[1]]].to_f64()
    })
}

/// Find maximum absolute value in matrix
fn max_abs<T>(matrix: &Tensor<T, (usize, usize)>) -> T
where
    T: Copy + PartialOrd + From<f64> + std::ops::Sub<Output = T>,
{
    let (m, n) = *matrix.shape();
    let mut max_val = <T as From<f64>>::from(0.0);
    let zero = <T as From<f64>>::from(0.0);
    for i in 0..m {
        for j in 0..n {
            let val = matrix[[i, j]];
            let abs_val = if val >= zero { val } else { zero - val };
            if abs_val > max_val {
                max_val = abs_val;
            }
        }
    }
    max_val
}

/// Matrix subtraction
fn mat_sub<T>(a: &Tensor<T, (usize, usize)>, b: &Tensor<T, (usize, usize)>) -> Tensor<T, (usize, usize)>
where
    T: Copy + std::ops::Sub<Output = T>,
{
    let shape_a = *a.shape();
    let shape_b = *b.shape();
    assert_eq!(shape_a, shape_b, "Matrix dimensions must match for subtraction");
    
    Tensor::from_fn(shape_a, |idx| {
        a[[idx[0], idx[1]]] - b[[idx[0], idx[1]]]
    })
}

// ===== Test Templates =====

/// Template function for RRQR identity matrix test
fn test_rrqr_identity_matrix_template<T: Precision>() 
where 
    T: Copy + From<f64>,
{
    let matrix = Tensor::from_fn((2, 2), |idx| {
        if idx[0] == idx[1] {
            <T as From<f64>>::from(1.0)
        } else {
            <T as From<f64>>::from(0.0)
        }
    });
    
    let mut a = matrix.clone();
    let rtol = <T as Precision>::epsilon();
    let (qr, rank) = rrqr(&mut a, rtol);
    
    assert_eq!(rank, 2);
    assert_eq!(qr.jpvt[0], 0);
    assert_eq!(qr.jpvt[1], 1);
}

/// Template function for RRQR simple matrix test (f64 version)
fn test_rrqr_simple_matrix_template_f64() {
    let mut a = Tensor::from_fn((2, 2), |idx| {
        match (idx[0], idx[1]) {
            (0, 0) => 2.0,
            (0, 1) => 1.0,
            (1, 0) => 4.0,
            (1, 1) => 3.0,
            _ => unreachable!(),
        }
    });
    
    let (qr, rank) = rrqr(&mut a, f64::EPSILON);
    assert_eq!(rank, 2);
    
    // Test reconstruction: A * P = Q * R
    let (q, r) = truncate_qr_result(&qr, rank);
    
    // Create permutation matrix P
    let mut p = Tensor::from_elem((2, 2), 0.0);
    for (i, &j) in qr.jpvt.iter().enumerate() {
        p[[j, i]] = 1.0;
    }
    
    let a_original = Tensor::from_fn((2, 2), |idx| {
        match (idx[0], idx[1]) {
            (0, 0) => 2.0,
            (0, 1) => 1.0,
            (1, 0) => 4.0,
            (1, 1) => 3.0,
            _ => unreachable!(),
        }
    });
    
    let ap = matmul(&a_original, &p);
    let qr_reconstructed = matmul(&q, &r);
    
    let error = mat_sub(&ap, &qr_reconstructed);
    let max_error = max_abs(&error);
    
    assert!(max_error < 100.0 * f64::EPSILON, "Reconstruction error too large: {}", max_error);
}

/// Template function for RRQR simple matrix test (TwoFloat version)
fn test_rrqr_simple_matrix_template_twofloat() {
    let matrix_f64 = Tensor::from_fn((2, 2), |idx| {
        match (idx[0], idx[1]) {
            (0, 0) => 2.0,
            (0, 1) => 1.0,
            (1, 0) => 4.0,
            (1, 1) => 3.0,
            _ => unreachable!(),
        }
    });
    
    let mut a = to_twofloat_matrix(&matrix_f64);
    let rtol = TwoFloatPrecision::epsilon();
    
    let (qr, rank) = rrqr(&mut a, rtol);
    assert_eq!(rank, 2);
    
    // Test reconstruction: A * P = Q * R
    let (q, r) = truncate_qr_result(&qr, rank);
    
    // Create permutation matrix P
    let mut p = Tensor::from_elem((2, 2), 0.0);
    for (i, &j) in qr.jpvt.iter().enumerate() {
        p[[j, i]] = 1.0;
    }
    
    // Convert Q and R to f64 for reconstruction
    let q_f64 = to_f64_matrix(&q);
    let r_f64 = to_f64_matrix(&r);
    
    let ap = matmul(&matrix_f64, &p);
    let qr_reconstructed = matmul(&q_f64, &r_f64);
    
    let error = mat_sub(&ap, &qr_reconstructed);
    let max_error = max_abs(&error);
    
    assert!(max_error < 100.0 * f64::EPSILON, "Reconstruction error too large: {}", max_error);
}

/// Template function for RRQR rank deficient matrix test
fn test_rrqr_rank_deficient_template<T: Precision>() 
where 
    T: Copy + From<f64>,
{
    let matrix = Tensor::from_fn((3, 3), |idx| {
        <T as From<f64>>::from((idx[0] + 1) as f64 * (idx[1] + 1) as f64)
    });
    
    let mut a = matrix.clone();
    let rtol = <T as Precision>::epsilon();
    let (_qr, rank) = rrqr(&mut a, rtol);
    
    assert_eq!(rank, 1);
}

/// Template function for RRQR zero matrix test
fn test_rrqr_zero_matrix_template<T: Precision>() 
where 
    T: Copy + From<f64>,
{
    let mut a = Tensor::from_elem((2, 2), <T as From<f64>>::from(0.0));
    let rtol = <T as Precision>::epsilon();
    let (_qr, rank) = rrqr(&mut a, rtol);
    
    assert_eq!(rank, 0);
}

/// Template function for RRQR single element test
fn test_rrqr_single_element_template<T: Precision>() 
where 
    T: Copy + From<f64>,
{
    let mut a = Tensor::from_elem((1, 1), <T as From<f64>>::from(5.0));
    let rtol = <T as Precision>::epsilon();
    let (qr, rank) = rrqr(&mut a, rtol);
    
    assert_eq!(rank, 1);
    assert_eq!(qr.jpvt[0], 0);
}

/// Template function for RRQR orthogonal Q test (f64 version)
fn test_rrqr_orthogonal_q_template_f64() {
    let mut a = Tensor::from_fn((2, 2), |idx| {
        match (idx[0], idx[1]) {
            (0, 0) => 3.0,
            (0, 1) => 4.0,
            (1, 0) => 4.0,
            (1, 1) => 3.0,
            _ => unreachable!(),
        }
    });
    
    let (qr, rank) = rrqr(&mut a, f64::EPSILON);
    let (q, _r) = truncate_qr_result(&qr, rank);
    
    // Q should be orthogonal: Q^T * Q = I
    let qt = transpose(&q);
    let qtq = matmul(&qt, &q);
    let identity: Tensor<f64, (usize, usize)> = eye(qtq.shape().1);
    
    let error = mat_sub(&qtq, &identity);
    let max_error = max_abs(&error);
    
    assert!(max_error < 1e-10, "Q is not orthogonal, max error: {}", max_error);
}

/// Template function for RRQR orthogonal Q test (TwoFloat version)
fn test_rrqr_orthogonal_q_template_twofloat() {
    let matrix_f64 = Tensor::from_fn((2, 2), |idx| {
        match (idx[0], idx[1]) {
            (0, 0) => 3.0,
            (0, 1) => 4.0,
            (1, 0) => 4.0,
            (1, 1) => 3.0,
            _ => unreachable!(),
        }
    });
    
    let mut a = to_twofloat_matrix(&matrix_f64);
    let rtol = TwoFloatPrecision::epsilon();
    
    let (qr, rank) = rrqr(&mut a, rtol);
    let (q, _r) = truncate_qr_result(&qr, rank);
    
    // Convert Q to f64 for orthogonality check
    let q_f64 = to_f64_matrix(&q);
    
    // Q should be orthogonal: Q^T * Q = I
    let qt = transpose(&q_f64);
    let qtq = matmul(&qt, &q_f64);
    let identity: Tensor<f64, (usize, usize)> = eye(qtq.shape().1);
    
    let error = mat_sub(&qtq, &identity);
    let max_error = max_abs(&error);
    
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
    let matrix_f64 = Tensor::from_fn((3, 3), |idx| {
        match (idx[0], idx[1]) {
            (0, _) => (idx[1] + 1) as f64 * 1e-10,
            (1, _) => (idx[1] + 1) as f64 * 2e-10,
            (2, _) => (idx[1] + 1) as f64 * 1e-15,
            _ => unreachable!(),
        }
    });
    
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
    
    let (m, n) = *q_f64.shape();
    for i in 0..m {
        for j in 0..n {
            let diff = (q_f64[[i, j]] - q_tf_f64[[i, j]]).abs();
            // TwoFloat should be more accurate, so difference should be small
            assert!(diff < 100.0 * f64::EPSILON, "Q matrix difference at [{}, {}]: {}", i, j, diff);
        }
    }
}

