//! Basic RRQR tests

use xprec_svd::*;
use ndarray::array;
// use approx::assert_abs_diff_eq;

#[test]
fn test_rrqr_identity_matrix() {
    let mut a = array![
        [1.0, 0.0],
        [0.0, 1.0]
    ];
    
    let (qr, rank) = rrqr(&mut a, 1e-10);
    
    // Identity matrix should have rank 2
    assert_eq!(rank, 2);
    
    // Check pivot indices (should be [0, 1] for identity)
    assert_eq!(qr.jpvt[0], 0);
    assert_eq!(qr.jpvt[1], 1);
}

#[test]
fn test_rrqr_simple_matrix() {
    let mut a = array![
        [2.0, 1.0],
        [4.0, 3.0]
    ];
    
    let (qr, rank) = rrqr(&mut a, 1e-10);
    
    // Should have full rank
    assert_eq!(rank, 2);
    
    // Test reconstruction: A * P = Q * R
    let (q, r) = truncate_qr_result(&qr, rank);
    
    // Create permutation matrix P
    let mut p: ndarray::Array2<f64> = ndarray::Array2::zeros((a.ncols(), a.ncols()));
    for (i, &j) in qr.jpvt.iter().enumerate() {
        p[[j, i]] = 1.0;
    }
    
    // Use original matrix for reconstruction
    let a_original = array![
        [2.0, 1.0],
        [4.0, 3.0]
    ];
    
    let ap = a_original.dot(&p);
    let qr_reconstructed = q.dot(&r);
    
    // Check reconstruction error
    let error = &ap - &qr_reconstructed;
    let max_error = error.iter().map(|x| x.abs()).fold(0.0, f64::max);
    
    assert!(max_error < 1e-10, "Reconstruction error too large: {}", max_error);
}

#[test]
fn test_rrqr_rank_deficient() {
    let mut a = array![
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ];
    
    let (qr, rank) = rrqr(&mut a, 1e-10);
    
    // Should have rank 1 (all rows are multiples of [1, 2, 3])
    assert_eq!(rank, 1);
}

#[test]
fn test_rrqr_zero_matrix() {
    let mut a = array![
        [0.0, 0.0],
        [0.0, 0.0]
    ];
    
    let (qr, rank) = rrqr(&mut a, 1e-10);
    
    // Zero matrix should have rank 0
    assert_eq!(rank, 0);
}

#[test]
fn test_rrqr_single_element() {
    let mut a = array![[5.0]];
    
    let (qr, rank) = rrqr(&mut a, 1e-10);
    
    // Single non-zero element should have rank 1
    assert_eq!(rank, 1);
    assert_eq!(qr.jpvt[0], 0);
}

#[test]
fn test_rrqr_orthogonal_q() {
    let mut a = array![
        [3.0, 4.0],
        [4.0, 3.0]
    ];
    
    let (qr, rank) = rrqr(&mut a, 1e-10);
    let (q, _r) = truncate_qr_result(&qr, rank);
    
    // Q should be orthogonal: Q^T * Q = I
    let qtq = q.t().dot(&q);
    let identity: ndarray::Array2<f64> = ndarray::Array2::eye(q.ncols());
    
    let error = &qtq - &identity;
    let max_error = error.iter().map(|x| x.abs()).fold(0.0, f64::max);
    
    assert!(max_error < 1e-10, "Q is not orthogonal, max error: {}", max_error);
}
