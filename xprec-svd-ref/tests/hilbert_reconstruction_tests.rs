//! Hilbert matrix reconstruction tests for xprec-svd
//!
//! Tests SVD reconstruction accuracy using Hilbert matrices, which are known
//! to be ill-conditioned and thus good test cases for SVD accuracy.

use ndarray::Array2;
use xprec_svd::{tsvd_f64, tsvd_twofloat_from_f64};

/// Create Hilbert matrix of size n x n
/// H[i,j] = 1 / (i + j + 1)
fn create_hilbert_matrix(n: usize) -> Array2<f64> {
    let mut h = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            h[[i, j]] = 1.0 / ((i + j + 1) as f64);
        }
    }
    h
}

/// Reconstruct matrix from SVD: A = U * S * V^T
fn reconstruct_matrix_f64(
    u: &Array2<f64>,
    s: &ndarray::Array1<f64>,
    v: &Array2<f64>,
) -> Array2<f64> {
    let m = u.nrows();
    let n = v.nrows();
    let k = s.len();
    
    let mut reconstructed = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += u[[i, l]] * s[l] * v[[j, l]];
            }
            reconstructed[[i, j]] = sum;
        }
    }
    reconstructed
}

/// Calculate Frobenius norm of matrix
fn frobenius_norm(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|&x| x * x).sum::<f64>().sqrt()
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
    for (i, &s) in result.s.iter().enumerate() {
        println!("  s[{}] = {:.6e}", i, s);
    }
    
    // Reconstruct matrix
    let h_reconstructed = reconstruct_matrix_f64(&result.u, &result.s, &result.v);
    
    // Compute reconstruction error
    let mut diff = h.clone();
    diff -= &h_reconstructed;
    let error_norm = frobenius_norm(&diff);
    let relative_error = error_norm / frobenius_norm(&h);
    
    println!("Reconstruction error:");
    println!("  Absolute (Frobenius): {:.6e}", error_norm);
    println!("  Relative: {:.6e}", relative_error);
    
    // Check sample elements
    println!("Sample elements comparison:");
    println!("  H[0,0]: orig={:.6e}, recon={:.6e}, diff={:.6e}", 
             h[[0,0]], h_reconstructed[[0,0]], h[[0,0]] - h_reconstructed[[0,0]]);
    println!("  H[0,1]: orig={:.6e}, recon={:.6e}, diff={:.6e}", 
             h[[0,1]], h_reconstructed[[0,1]], h[[0,1]] - h_reconstructed[[0,1]]);
    println!("  H[2,3]: orig={:.6e}, recon={:.6e}, diff={:.6e}", 
             h[[2,3]], h_reconstructed[[2,3]], h[[2,3]] - h_reconstructed[[2,3]]);
    
    // Assert reconstruction is accurate
    assert!(
        relative_error < 1e-12,
        "Reconstruction error too large: {:.6e}",
        relative_error
    );
}

#[test]
fn test_hilbert_5x5_twofloat_reconstruction() {
    let n = 5;
    let h = create_hilbert_matrix(n);
    
    println!("Testing Hilbert {}x{} with TwoFloat precision", n, n);
    println!("Original matrix norm: {:.6e}", frobenius_norm(&h));
    
    // Compute SVD with TwoFloat
    let rtol = 1e-14;
    let result = tsvd_twofloat_from_f64(&h, rtol).expect("SVD failed");
    
    println!("SVD computed: rank = {}", result.rank);
    println!("Singular values:");
    for (i, &s) in result.s.iter().enumerate() {
        println!("  s[{}] = {:.6e}", i, s.to_f64());
    }
    
    // Reconstruct matrix (convert TwoFloat to f64 for comparison)
    let u_f64 = result.u.map(|&x| x.to_f64());
    let s_f64 = result.s.map(|&x| x.to_f64());
    let v_f64 = result.v.map(|&x| x.to_f64());
    
    let h_reconstructed = reconstruct_matrix_f64(&u_f64, &s_f64, &v_f64);
    
    // Compute reconstruction error
    let mut diff = h.clone();
    diff -= &h_reconstructed;
    let error_norm = frobenius_norm(&diff);
    let relative_error = error_norm / frobenius_norm(&h);
    
    println!("Reconstruction error:");
    println!("  Absolute (Frobenius): {:.6e}", error_norm);
    println!("  Relative: {:.6e}", relative_error);
    
    // Check sample elements
    println!("Sample elements comparison:");
    println!("  H[0,0]: orig={:.6e}, recon={:.6e}, diff={:.6e}", 
             h[[0,0]], h_reconstructed[[0,0]], h[[0,0]] - h_reconstructed[[0,0]]);
    println!("  H[0,1]: orig={:.6e}, recon={:.6e}, diff={:.6e}", 
             h[[0,1]], h_reconstructed[[0,1]], h[[0,1]] - h_reconstructed[[0,1]]);
    println!("  H[2,3]: orig={:.6e}, recon={:.6e}, diff={:.6e}", 
             h[[2,3]], h_reconstructed[[2,3]], h[[2,3]] - h_reconstructed[[2,3]]);
    
    // TwoFloat should give better accuracy than f64
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
        println!("  s[{}] = {:.6e}", i, result.s[i]);
    }
    
    let h_reconstructed = reconstruct_matrix_f64(&result.u, &result.s, &result.v);
    
    let mut diff = h.clone();
    diff -= &h_reconstructed;
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
fn test_hilbert_10x10_twofloat_reconstruction() {
    let n = 10;
    let h = create_hilbert_matrix(n);
    
    println!("Testing Hilbert {}x{} with TwoFloat precision", n, n);
    
    let rtol = 1e-14;
    let result = tsvd_twofloat_from_f64(&h, rtol).expect("SVD failed");
    
    println!("SVD rank: {}", result.rank);
    println!("First 5 singular values:");
    for i in 0..5.min(result.s.len()) {
        println!("  s[{}] = {:.6e}", i, result.s[i].to_f64());
    }
    
    let u_f64 = result.u.map(|&x| x.to_f64());
    let s_f64 = result.s.map(|&x| x.to_f64());
    let v_f64 = result.v.map(|&x| x.to_f64());
    
    let h_reconstructed = reconstruct_matrix_f64(&u_f64, &s_f64, &v_f64);
    
    let mut diff = h.clone();
    diff -= &h_reconstructed;
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&h);
    
    println!("Relative reconstruction error: {:.6e}", relative_error);
    
    // TwoFloat should give better accuracy, but still limited by condition number
    assert!(
        relative_error < 1e-7,
        "Reconstruction error too large: {:.6e}",
        relative_error
    );
}

