use xprec_svd::precision::TwoFloatPrecision;
use xprec_svd::qr::{rrqr, truncate_qr_result};
use ndarray::{Array2, array};

/// Convert f64 matrix to TwoFloatPrecision matrix
fn to_twofloat_matrix(matrix: &Array2<f64>) -> Array2<TwoFloatPrecision> {
    let (m, n) = matrix.dim();
    let mut result = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = TwoFloatPrecision::from_f64(matrix[[i, j]]);
        }
    }
    result
}

/// Convert TwoFloatPrecision matrix back to f64 for comparison
fn to_f64_matrix(matrix: &Array2<TwoFloatPrecision>) -> Array2<f64> {
    let (m, n) = matrix.dim();
    let mut result = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = matrix[[i, j]].to_f64();
        }
    }
    result
}

#[test]
fn test_twofloat_rrqr_identity() {
    let matrix_f64 = Array2::eye(3);
    let mut matrix = to_twofloat_matrix(&matrix_f64);
    
    let rtol = TwoFloatPrecision::epsilon();
    let (qr, rank) = rrqr(&mut matrix, rtol);
    
    assert_eq!(rank, 3);
    
    // Check that Q is orthogonal
    let (q, _r) = truncate_qr_result(&qr, rank);
    let qtq = q.t().dot(&q);
    let identity: Array2<f64> = Array2::eye(3);
    let qtq_f64 = to_f64_matrix(&qtq);
    
    for i in 0..3 {
        for j in 0..3 {
            let diff = (qtq_f64[[i, j]] - identity[[i, j]]).abs() as f64;
            assert!(diff < 100.0 * f64::EPSILON, "Q^T * Q should be identity, got diff = {}", diff);
        }
    }
}

#[test]
fn test_twofloat_rrqr_rank_one() {
    let matrix_f64 = array![
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ];
    let mut matrix = to_twofloat_matrix(&matrix_f64);
    
    let rtol = TwoFloatPrecision::epsilon();
    let (qr, rank) = rrqr(&mut matrix, rtol);
    
    assert_eq!(rank, 1);
    
    // Check reconstruction: A = Q * R * P^T
    let (q, r) = truncate_qr_result(&qr, rank);
    let p = &qr.jpvt;
    
    // Reconstruct A with permutation
    let mut reconstructed = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..rank {
                reconstructed[[i, j]] += q[[i, k]] * r[[k, j]];
            }
        }
    }
    
    // Apply permutation
    let mut permuted = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            permuted[[i, p[j]]] = reconstructed[[i, j]];
        }
    }
    
    let original_f64 = to_f64_matrix(&to_twofloat_matrix(&matrix_f64));
    let reconstructed_f64 = to_f64_matrix(&permuted);
    
    for i in 0..3 {
        for j in 0..3 {
            let diff = (original_f64[[i, j]] - reconstructed_f64[[i, j]]).abs();
            assert!(diff < 100.0 * f64::EPSILON, "Reconstruction error at [{}, {}]: {}", i, j, diff);
        }
    }
}

#[test]
fn test_twofloat_rrqr_precision_comparison() {
    // Test matrix with small values that might cause precision issues
    let matrix_f64 = array![
        [1e-10, 2e-10, 3e-10],
        [2e-10, 4e-10, 6e-10],
        [1e-15, 2e-15, 3e-15]  // Very small values
    ];
    
    // Test with f64
    let mut matrix_f64_copy = matrix_f64.clone();
    let rtol_f64 = f64::EPSILON;
    let (qr_f64, rank_f64) = xprec_svd::qr::rrqr(&mut matrix_f64_copy, rtol_f64);
    
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

#[test]
fn test_twofloat_rrqr_eps_precision() {
    // Test with EPS-level precision requirements
    let matrix_f64 = array![
        [1.0, 1e-15, 1e-16],
        [1e-15, 1.0, 1e-15],
        [1e-16, 1e-15, 1.0]
    ];
    
    let mut matrix = to_twofloat_matrix(&matrix_f64);
    let rtol = TwoFloatPrecision::epsilon(); // EPSILON tolerance
    
    let (qr, rank) = rrqr(&mut matrix, rtol);
    
    // Should detect rank 3 (all diagonal elements are significant)
    assert_eq!(rank, 3);
    
    // Check orthogonality with high precision
    let (q, _r) = truncate_qr_result(&qr, rank);
    let qtq = q.t().dot(&q);
    let identity: Array2<f64> = Array2::eye(3);
    let qtq_f64 = to_f64_matrix(&qtq);
    
    for i in 0..3 {
        for j in 0..3 {
            let diff = (qtq_f64[[i, j]] - identity[[i, j]]).abs() as f64;
            // TwoFloat should achieve better than machine epsilon precision
            assert!(diff < 1e-15, "Q^T * Q orthogonality error at [{}, {}]: {}", i, j, diff);
        }
    }
}
