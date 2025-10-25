//! Hilbert matrix reconstruction tests for xprec-svd
//!
//! Tests SVD reconstruction accuracy using Hilbert matrices, which are known
//! to be ill-conditioned and thus good test cases for SVD accuracy.

use mdarray::Tensor;
use xprec_svd::{tsvd_df64_from_f64, tsvd_f64};

/// Create Hilbert matrix of size n x n
/// H[i,j] = 1 / (i + j + 1)
fn create_hilbert_matrix(n: usize) -> Tensor<f64, (usize, usize)> {
    Tensor::from_fn((n, n), |idx| 1.0 / ((idx[0] + idx[1] + 1) as f64))
}

/// Reconstruct matrix from SVD: A = U * S * V^T
fn reconstruct_matrix_f64(
    u: &Tensor<f64, (usize, usize)>,
    s: &Tensor<f64, (usize,)>,
    v: &Tensor<f64, (usize, usize)>,
) -> Tensor<f64, (usize, usize)> {
    let u_shape = *u.shape();
    let v_shape = *v.shape();
    let k = s.len();

    Tensor::from_fn((u_shape.0, v_shape.0), |idx| {
        let mut sum = 0.0;
        for l in 0..k {
            sum += u[[idx[0], l]] * s[[l]] * v[[idx[1], l]];
        }
        sum
    })
}

/// Calculate Frobenius norm of matrix
fn frobenius_norm(matrix: &Tensor<f64, (usize, usize)>) -> f64 {
    let (m, n) = *matrix.shape();
    let mut sum = 0.0;
    for i in 0..m {
        for j in 0..n {
            let val = matrix[[i, j]];
            sum += val * val;
        }
    }
    sum.sqrt()
}

fn matrix_sub(
    a: &Tensor<f64, (usize, usize)>,
    b: &Tensor<f64, (usize, usize)>,
) -> Tensor<f64, (usize, usize)> {
    let shape = *a.shape();
    Tensor::from_fn(shape, |idx| a[[idx[0], idx[1]]] - b[[idx[0], idx[1]]])
}

#[test]
fn test_hilbert_5x5_f64_reconstruction() {
    let n = 5;
    let h = create_hilbert_matrix(n);

    println!("Testing Hilbert {}x{} with f64 precision", n, n);
    println!("Original matrix norm: {:.6e}", frobenius_norm(&h));

    // Compute SVD with very loose tolerance to get all singular values
    let rtol = 1e-16;
    let result = tsvd_f64(&h, rtol).expect("SVD failed");

    println!("SVD computed: rank = {} (expected {})", result.rank, n);
    println!("Singular values:");
    for i in 0..result.s.len() {
        println!("  s[{}] = {:.6e}", i, result.s[[i]]);
    }

    // Reconstruct matrix
    let h_reconstructed = reconstruct_matrix_f64(&result.u, &result.s, &result.v);

    // Compute reconstruction error
    let diff = matrix_sub(&h, &h_reconstructed);
    let error_norm = frobenius_norm(&diff);
    let relative_error = error_norm / frobenius_norm(&h);

    println!("Reconstruction error:");
    println!("  Absolute (Frobenius): {:.6e}", error_norm);
    println!("  Relative: {:.6e}", relative_error);

    // Check sample elements
    println!("Sample elements comparison:");
    println!(
        "  H[0,0]: orig={:.6e}, recon={:.6e}, diff={:.6e}",
        h[[0, 0]],
        h_reconstructed[[0, 0]],
        h[[0, 0]] - h_reconstructed[[0, 0]]
    );
    println!(
        "  H[0,1]: orig={:.6e}, recon={:.6e}, diff={:.6e}",
        h[[0, 1]],
        h_reconstructed[[0, 1]],
        h[[0, 1]] - h_reconstructed[[0, 1]]
    );
    println!(
        "  H[2,3]: orig={:.6e}, recon={:.6e}, diff={:.6e}",
        h[[2, 3]],
        h_reconstructed[[2, 3]],
        h[[2, 3]] - h_reconstructed[[2, 3]]
    );

    // Assert reconstruction is accurate
    assert!(
        relative_error < 1e-12,
        "Reconstruction error too large: {:.6e}",
        relative_error
    );
}

#[test]
fn test_hilbert_5x5_df64_reconstruction() {
    let n = 5;
    let h = create_hilbert_matrix(n);

    println!("Testing Hilbert {}x{} with Df64 precision", n, n);
    println!("Original matrix norm: {:.6e}", frobenius_norm(&h));

    // Compute SVD with Df64
    let rtol = 1e-14;
    let result = tsvd_df64_from_f64(&h, rtol).expect("SVD failed");

    println!("SVD computed: rank = {}", result.rank);
    println!("Singular values:");
    for i in 0..result.s.len() {
        println!("  s[{}] = {:.6e}", i, result.s[[i]].to_f64());
    }

    // Reconstruct matrix (convert Df64 to f64 for comparison)
    let u_shape = *result.u.shape();
    let u_f64 = Tensor::from_fn(u_shape, |idx| result.u[[idx[0], idx[1]]].to_f64());
    let s_f64 = Tensor::from_fn((result.s.len(),), |idx| result.s[[idx[0]]].to_f64());
    let v_shape = *result.v.shape();
    let v_f64 = Tensor::from_fn(v_shape, |idx| result.v[[idx[0], idx[1]]].to_f64());

    let h_reconstructed = reconstruct_matrix_f64(&u_f64, &s_f64, &v_f64);

    // Compute reconstruction error
    let diff = matrix_sub(&h, &h_reconstructed);
    let error_norm = frobenius_norm(&diff);
    let relative_error = error_norm / frobenius_norm(&h);

    println!("Reconstruction error:");
    println!("  Absolute (Frobenius): {:.6e}", error_norm);
    println!("  Relative: {:.6e}", relative_error);

    // Check sample elements
    println!("Sample elements comparison:");
    println!(
        "  H[0,0]: orig={:.6e}, recon={:.6e}, diff={:.6e}",
        h[[0, 0]],
        h_reconstructed[[0, 0]],
        h[[0, 0]] - h_reconstructed[[0, 0]]
    );
    println!(
        "  H[0,1]: orig={:.6e}, recon={:.6e}, diff={:.6e}",
        h[[0, 1]],
        h_reconstructed[[0, 1]],
        h[[0, 1]] - h_reconstructed[[0, 1]]
    );
    println!(
        "  H[2,3]: orig={:.6e}, recon={:.6e}, diff={:.6e}",
        h[[2, 3]],
        h_reconstructed[[2, 3]],
        h[[2, 3]] - h_reconstructed[[2, 3]]
    );

    // Df64 should give better accuracy than f64
    assert!(
        relative_error < 1e-14,
        "Reconstruction error too large: {:.6e}",
        relative_error
    );
}

#[test]
fn test_hilbert_10x10_f64_reconstruction() {
    let n = 10;
    let h = create_hilbert_matrix(n);

    println!("Testing Hilbert {}x{} with f64 precision", n, n);

    let rtol = 1e-14;
    let result = tsvd_f64(&h, rtol).expect("SVD failed");

    println!("SVD rank: {}", result.rank);
    println!("First 5 singular values:");
    for i in 0..5.min(result.s.len()) {
        println!("  s[{}] = {:.6e}", i, result.s[[i]]);
    }

    let h_reconstructed = reconstruct_matrix_f64(&result.u, &result.s, &result.v);

    let diff = matrix_sub(&h, &h_reconstructed);
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&h);

    println!("Relative reconstruction error: {:.6e}", relative_error);

    // Hilbert matrices are ill-conditioned, so we need a looser tolerance
    assert!(
        relative_error < 1e-7,
        "Reconstruction error too large: {:.6e}",
        relative_error
    );
}

#[test]
fn test_hilbert_10x10_df64_reconstruction() {
    let n = 10;
    let h = create_hilbert_matrix(n);

    println!("Testing Hilbert {}x{} with Df64 precision", n, n);

    let rtol = 1e-14;
    let result = tsvd_df64_from_f64(&h, rtol).expect("SVD failed");

    println!("SVD rank: {}", result.rank);
    println!("First 5 singular values:");
    for i in 0..5.min(result.s.len()) {
        println!("  s[{}] = {:.6e}", i, result.s[[i]].to_f64());
    }

    let u_shape = *result.u.shape();
    let u_f64 = Tensor::from_fn(u_shape, |idx| result.u[[idx[0], idx[1]]].to_f64());
    let s_f64 = Tensor::from_fn((result.s.len(),), |idx| result.s[[idx[0]]].to_f64());
    let v_shape = *result.v.shape();
    let v_f64 = Tensor::from_fn(v_shape, |idx| result.v[[idx[0], idx[1]]].to_f64());

    let h_reconstructed = reconstruct_matrix_f64(&u_f64, &s_f64, &v_f64);

    let diff = matrix_sub(&h, &h_reconstructed);
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&h);

    println!("Relative reconstruction error: {:.6e}", relative_error);

    // Df64 should give better accuracy, but still limited by condition number
    assert!(
        relative_error < 1e-7,
        "Reconstruction error too large: {:.6e}",
        relative_error
    );
}
