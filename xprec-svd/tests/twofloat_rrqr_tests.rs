use xprec_svd::precision::TwoFloatPrecision;
use xprec_svd::qr::{rrqr, truncate_qr_result};
use mdarray::Tensor;

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

#[test]
fn test_twofloat_rrqr_identity() {
    let matrix_f64: Tensor<f64, (usize, usize)> = eye(3);
    let mut matrix = to_twofloat_matrix(&matrix_f64);
    
    let rtol = TwoFloatPrecision::epsilon();
    let (qr, rank) = rrqr(&mut matrix, rtol);
    
    assert_eq!(rank, 3);
    
    // Check that Q is orthogonal
    let (q, _r) = truncate_qr_result(&qr, rank);
    let qt = transpose(&q);
    let qtq = matmul(&qt, &q);
    let identity: Tensor<f64, (usize, usize)> = eye(3);
    let qtq_f64 = to_f64_matrix(&qtq);
    
    for i in 0..3 {
        for j in 0..3 {
            let diff = (qtq_f64[[i, j]] - identity[[i, j]]).abs();
            assert!(diff < 100.0 * f64::EPSILON, "Q^T * Q should be identity, got diff = {}", diff);
        }
    }
}

#[test]
fn test_twofloat_rrqr_rank_one() {
    let matrix_f64 = Tensor::from_fn((3, 3), |idx| {
        match (idx[0], idx[1]) {
            (0, 0) => 1.0,
            (0, 1) => 2.0,
            (0, 2) => 3.0,
            (1, 0) => 2.0,
            (1, 1) => 4.0,
            (1, 2) => 6.0,
            (2, 0) => 3.0,
            (2, 1) => 6.0,
            (2, 2) => 9.0,
            _ => unreachable!(),
        }
    });
    let mut matrix = to_twofloat_matrix(&matrix_f64);
    
    let rtol = TwoFloatPrecision::epsilon();
    let (qr, rank) = rrqr(&mut matrix, rtol);
    
    assert_eq!(rank, 1);
    
    // Check reconstruction: A = Q * R * P^T
    let (q, r) = truncate_qr_result(&qr, rank);
    let p = &qr.jpvt;
    
    // Reconstruct A with permutation
    let mut reconstructed = Tensor::from_elem((3, 3), TwoFloatPrecision::from_f64(0.0));
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..rank {
                reconstructed[[i, j]] = reconstructed[[i, j]] + q[[i, k]] * r[[k, j]];
            }
        }
    }
    
    // Apply permutation
    let mut permuted = Tensor::from_elem((3, 3), TwoFloatPrecision::from_f64(0.0));
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
    let matrix_f64 = Tensor::from_fn((3, 3), |idx| {
        match (idx[0], idx[1]) {
            (0, 0) => 1e-10,
            (0, 1) => 2e-10,
            (0, 2) => 3e-10,
            (1, 0) => 2e-10,
            (1, 1) => 4e-10,
            (1, 2) => 6e-10,
            (2, 0) => 1e-15,
            (2, 1) => 2e-15,
            (2, 2) => 3e-15,
            _ => unreachable!(),
        }
    });
    
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
    
    let q_f64_shape = *q_f64.shape();
    for i in 0..q_f64_shape.0 {
        for j in 0..q_f64_shape.1 {
            let diff = (q_f64[[i, j]] - q_tf_f64[[i, j]]).abs();
            // TwoFloat should be more accurate, so difference should be small
            assert!(diff < 100.0 * f64::EPSILON, "Q matrix difference at [{}, {}]: {}", i, j, diff);
        }
    }
}

#[test]
fn test_twofloat_rrqr_eps_precision() {
    // Test with EPS-level precision requirements
    let matrix_f64 = Tensor::from_fn((3, 3), |idx| {
        match (idx[0], idx[1]) {
            (0, 0) => 1.0,
            (0, 1) => 1e-15,
            (0, 2) => 1e-16,
            (1, 0) => 1e-15,
            (1, 1) => 1.0,
            (1, 2) => 1e-15,
            (2, 0) => 1e-16,
            (2, 1) => 1e-15,
            (2, 2) => 1.0,
            _ => unreachable!(),
        }
    });
    
    let mut matrix = to_twofloat_matrix(&matrix_f64);
    let rtol = TwoFloatPrecision::epsilon(); // EPSILON tolerance
    
    let (qr, rank) = rrqr(&mut matrix, rtol);
    
    // Should detect rank 3 (all diagonal elements are significant)
    assert_eq!(rank, 3);
    
    // Check orthogonality with high precision
    let (q, _r) = truncate_qr_result(&qr, rank);
    let qt = transpose(&q);
    let qtq = matmul(&qt, &q);
    let identity: Tensor<f64, (usize, usize)> = eye(3);
    let qtq_f64 = to_f64_matrix(&qtq);
    
    for i in 0..3 {
        for j in 0..3 {
            let diff = (qtq_f64[[i, j]] - identity[[i, j]]).abs();
            // TwoFloat should achieve better than machine epsilon precision
            assert!(diff < 1e-15, "Q^T * Q orthogonality error at [{}, {}]: {}", i, j, diff);
        }
    }
}

