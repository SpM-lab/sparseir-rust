//! Comprehensive SVD accuracy tests for various matrix types

use ndarray::Array2;
use xprec_svd::{tsvd_f64, tsvd_twofloat_from_f64};

fn create_hilbert_matrix(m: usize, n: usize) -> Array2<f64> {
    let mut h = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            h[[i, j]] = 1.0 / ((i + j + 1) as f64);
        }
    }
    h
}

fn frobenius_norm(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

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

#[test]
fn test_hilbert_10x10_f64() {
    let h = create_hilbert_matrix(10, 10);
    
    println!("Testing 10x10 Hilbert matrix with f64 precision");
    
    let rtol = 1e-16;
    let result = tsvd_f64(&h, rtol).expect("SVD failed");
    
    assert_eq!(result.rank, 10);
    
    // Check singular values are in descending order
    for i in 0..result.s.len()-1 {
        assert!(result.s[i] >= result.s[i+1], 
                "Singular values not in descending order");
    }
    
    // Reconstruct and check error
    let h_reconstructed = reconstruct_matrix_f64(&result.u, &result.s, &result.v);
    let mut diff = h.clone();
    diff -= &h_reconstructed;
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&h);
    
    println!("  Rank: {}", result.rank);
    println!("  First singular value: {:.6e}", result.s[0]);
    println!("  Last singular value: {:.6e}", result.s[result.s.len()-1]);
    println!("  Condition number: {:.6e}", result.s[0] / result.s[result.s.len()-1]);
    println!("  Reconstruction error: {:.6e}", relative_error);
    
    assert!(relative_error < 1e-13, 
            "Reconstruction error too large: {:.6e}", relative_error);
}

#[test]
fn test_hilbert_10x10_twofloat() {
    let h = create_hilbert_matrix(10, 10);
    
    println!("Testing 10x10 Hilbert matrix with TwoFloat precision");
    
    let rtol = 1e-16;
    let result = tsvd_twofloat_from_f64(&h, rtol).expect("SVD failed");
    
    assert_eq!(result.rank, 10);
    
    // Convert to f64 for comparison
    let u_f64 = result.u.map(|&x| x.to_f64());
    let s_f64 = result.s.map(|&x| x.to_f64());
    let v_f64 = result.v.map(|&x| x.to_f64());
    
    let h_reconstructed = reconstruct_matrix_f64(&u_f64, &s_f64, &v_f64);
    let mut diff = h.clone();
    diff -= &h_reconstructed;
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&h);
    
    println!("  Rank: {}", result.rank);
    println!("  First singular value: {:.6e}", s_f64[0]);
    println!("  Last singular value: {:.6e}", s_f64[s_f64.len()-1]);
    println!("  Condition number: {:.6e}", s_f64[0] / s_f64[s_f64.len()-1]);
    println!("  Reconstruction error: {:.6e}", relative_error);
    
    // TwoFloat should give better or similar accuracy
    assert!(relative_error < 1e-13, 
            "Reconstruction error too large: {:.6e}", relative_error);
}

#[test]
fn test_hilbert_10x15_f64() {
    let h = create_hilbert_matrix(10, 15);
    
    println!("Testing 10x15 Hilbert matrix (wide) with f64 precision");
    
    let rtol = 1e-16;
    let result = tsvd_f64(&h, rtol).expect("SVD failed");
    
    // Rank should be at most min(m, n) = 10
    assert!(result.rank <= 10);
    
    let h_reconstructed = reconstruct_matrix_f64(&result.u, &result.s, &result.v);
    let mut diff = h.clone();
    diff -= &h_reconstructed;
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&h);
    
    println!("  Matrix shape: {}x{}", h.nrows(), h.ncols());
    println!("  Rank: {}", result.rank);
    println!("  U shape: {}x{}", result.u.nrows(), result.u.ncols());
    println!("  S length: {}", result.s.len());
    println!("  V shape: {}x{}", result.v.nrows(), result.v.ncols());
    println!("  First singular value: {:.6e}", result.s[0]);
    println!("  Last singular value: {:.6e}", result.s[result.s.len()-1]);
    println!("  Reconstruction error: {:.6e}", relative_error);
    
    assert!(relative_error < 1e-13, 
            "Reconstruction error too large: {:.6e}", relative_error);
}

#[test]
fn test_hilbert_15x10_f64() {
    let h = create_hilbert_matrix(15, 10);
    
    println!("Testing 15x10 Hilbert matrix (tall) with f64 precision");
    
    let rtol = 1e-16;
    let result = tsvd_f64(&h, rtol).expect("SVD failed");
    
    // Rank should be at most min(m, n) = 10
    assert!(result.rank <= 10);
    
    let h_reconstructed = reconstruct_matrix_f64(&result.u, &result.s, &result.v);
    let mut diff = h.clone();
    diff -= &h_reconstructed;
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&h);
    
    println!("  Matrix shape: {}x{}", h.nrows(), h.ncols());
    println!("  Rank: {}", result.rank);
    println!("  U shape: {}x{}", result.u.nrows(), result.u.ncols());
    println!("  S length: {}", result.s.len());
    println!("  V shape: {}x{}", result.v.nrows(), result.v.ncols());
    println!("  First singular value: {:.6e}", result.s[0]);
    println!("  Last singular value: {:.6e}", result.s[result.s.len()-1]);
    println!("  Reconstruction error: {:.6e}", relative_error);
    
    assert!(relative_error < 1e-13, 
            "Reconstruction error too large: {:.6e}", relative_error);
}

#[test]
fn test_hilbert_10x15_twofloat() {
    let h = create_hilbert_matrix(10, 15);
    
    println!("Testing 10x15 Hilbert matrix (wide) with TwoFloat precision");
    
    let rtol = 1e-16;
    let result = tsvd_twofloat_from_f64(&h, rtol).expect("SVD failed");
    
    assert!(result.rank <= 10);
    
    let u_f64 = result.u.map(|&x| x.to_f64());
    let s_f64 = result.s.map(|&x| x.to_f64());
    let v_f64 = result.v.map(|&x| x.to_f64());
    
    let h_reconstructed = reconstruct_matrix_f64(&u_f64, &s_f64, &v_f64);
    let mut diff = h.clone();
    diff -= &h_reconstructed;
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&h);
    
    println!("  Matrix shape: {}x{}", h.nrows(), h.ncols());
    println!("  Rank: {}", result.rank);
    println!("  Reconstruction error: {:.6e}", relative_error);
    
    assert!(relative_error < 1e-13, 
            "Reconstruction error too large: {:.6e}", relative_error);
}

#[test]
fn test_singular_values_comparison_10x10() {
    let h = create_hilbert_matrix(10, 10);
    
    println!("Comparing f64 vs TwoFloat singular values for 10x10 Hilbert");
    
    let rtol = 1e-16;
    let result_f64 = tsvd_f64(&h, rtol).expect("f64 SVD failed");
    let result_tf = tsvd_twofloat_from_f64(&h, rtol).expect("TwoFloat SVD failed");
    
    assert_eq!(result_f64.s.len(), result_tf.s.len(), "Different number of singular values");
    
    println!("  i   f64              TwoFloat         Rel Diff");
    for i in 0..result_f64.s.len() {
        let s_tf = result_tf.s[i].to_f64();
        let rel_diff = ((result_f64.s[i] - s_tf) / result_f64.s[i]).abs();
        println!("  {:2}  {:.6e}     {:.6e}     {:.2e}", 
                 i, result_f64.s[i], s_tf, rel_diff);
        
        // Allow larger relative error for very small singular values
        // (below machine epsilon * first singular value)
        let tolerance = if result_f64.s[i] < 1e-12 * result_f64.s[0] {
            1e-3  // 0.1% for very small singular values
        } else {
            1e-6  // 0.0001% for larger ones
        };
        
        assert!(rel_diff < tolerance, 
                "Singular value {} differs too much: {:.2e} (tolerance: {:.2e})", 
                i, rel_diff, tolerance);
    }
}

