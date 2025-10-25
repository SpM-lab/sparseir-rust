//! Comprehensive SVD accuracy tests for various matrix types

use mdarray::Tensor;
use xprec_svd::{tsvd_df64_from_f64, tsvd_f64};

fn create_hilbert_matrix(m: usize, n: usize) -> Tensor<f64, (usize, usize)> {
    Tensor::from_fn((m, n), |idx| 1.0 / ((idx[0] + idx[1] + 1) as f64))
}

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

fn matrix_sub(
    a: &Tensor<f64, (usize, usize)>,
    b: &Tensor<f64, (usize, usize)>,
) -> Tensor<f64, (usize, usize)> {
    let shape = *a.shape();
    Tensor::from_fn(shape, |idx| a[[idx[0], idx[1]]] - b[[idx[0], idx[1]]])
}

#[test]
fn test_hilbert_10x10_f64() {
    let h = create_hilbert_matrix(10, 10);

    println!("Testing 10x10 Hilbert matrix with f64 precision");

    let rtol = 1e-16;
    let result = tsvd_f64(&h, rtol).expect("SVD failed");

    assert_eq!(result.rank, 10);

    // Check singular values are in descending order
    for i in 0..result.s.len() - 1 {
        assert!(
            result.s[[i]] >= result.s[[i + 1]],
            "Singular values not in descending order"
        );
    }

    // Reconstruct and check error
    let h_reconstructed = reconstruct_matrix_f64(&result.u, &result.s, &result.v);
    let diff = matrix_sub(&h, &h_reconstructed);
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&h);

    println!("  Rank: {}", result.rank);
    println!("  First singular value: {:.6e}", result.s[[0]]);
    println!(
        "  Last singular value: {:.6e}",
        result.s[[result.s.len() - 1]]
    );
    println!(
        "  Condition number: {:.6e}",
        result.s[[0]] / result.s[[result.s.len() - 1]]
    );
    println!("  Reconstruction error: {:.6e}", relative_error);

    assert!(
        relative_error < 1e-13,
        "Reconstruction error too large: {:.6e}",
        relative_error
    );
}

#[test]
fn test_hilbert_10x10_df64() {
    let h = create_hilbert_matrix(10, 10);

    println!("Testing 10x10 Hilbert matrix with Df64 precision");

    let rtol = 1e-16;
    let result = tsvd_df64_from_f64(&h, rtol).expect("SVD failed");

    assert_eq!(result.rank, 10);

    // Convert to f64 for comparison
    let u_shape = *result.u.shape();
    let u_f64 = Tensor::from_fn(u_shape, |idx| result.u[[idx[0], idx[1]]].to_f64());
    let s_f64 = Tensor::from_fn((result.s.len(),), |idx| result.s[[idx[0]]].to_f64());
    let v_shape = *result.v.shape();
    let v_f64 = Tensor::from_fn(v_shape, |idx| result.v[[idx[0], idx[1]]].to_f64());

    let h_reconstructed = reconstruct_matrix_f64(&u_f64, &s_f64, &v_f64);
    let diff = matrix_sub(&h, &h_reconstructed);
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&h);

    println!("  Rank: {}", result.rank);
    println!("  First singular value: {:.6e}", s_f64[[0]]);
    println!("  Last singular value: {:.6e}", s_f64[[s_f64.len() - 1]]);
    println!(
        "  Condition number: {:.6e}",
        s_f64[[0]] / s_f64[[s_f64.len() - 1]]
    );
    println!("  Reconstruction error: {:.6e}", relative_error);

    // Df64 should give better or similar accuracy
    assert!(
        relative_error < 1e-13,
        "Reconstruction error too large: {:.6e}",
        relative_error
    );
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
    let diff = matrix_sub(&h, &h_reconstructed);
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&h);

    let h_shape = *h.shape();
    let u_shape = *result.u.shape();
    let v_shape = *result.v.shape();

    println!("  Matrix shape: {}x{}", h_shape.0, h_shape.1);
    println!("  Rank: {}", result.rank);
    println!("  U shape: {}x{}", u_shape.0, u_shape.1);
    println!("  S length: {}", result.s.len());
    println!("  V shape: {}x{}", v_shape.0, v_shape.1);
    println!("  First singular value: {:.6e}", result.s[[0]]);
    println!(
        "  Last singular value: {:.6e}",
        result.s[[result.s.len() - 1]]
    );
    println!("  Reconstruction error: {:.6e}", relative_error);

    assert!(
        relative_error < 1e-13,
        "Reconstruction error too large: {:.6e}",
        relative_error
    );
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
    let diff = matrix_sub(&h, &h_reconstructed);
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&h);

    let h_shape = *h.shape();
    let u_shape = *result.u.shape();
    let v_shape = *result.v.shape();

    println!("  Matrix shape: {}x{}", h_shape.0, h_shape.1);
    println!("  Rank: {}", result.rank);
    println!("  U shape: {}x{}", u_shape.0, u_shape.1);
    println!("  S length: {}", result.s.len());
    println!("  V shape: {}x{}", v_shape.0, v_shape.1);
    println!("  First singular value: {:.6e}", result.s[[0]]);
    println!(
        "  Last singular value: {:.6e}",
        result.s[[result.s.len() - 1]]
    );
    println!("  Reconstruction error: {:.6e}", relative_error);

    assert!(
        relative_error < 1e-13,
        "Reconstruction error too large: {:.6e}",
        relative_error
    );
}

#[test]
fn test_hilbert_10x15_df64() {
    let h = create_hilbert_matrix(10, 15);

    println!("Testing 10x15 Hilbert matrix (wide) with Df64 precision");

    let rtol = 1e-16;
    let result = tsvd_df64_from_f64(&h, rtol).expect("SVD failed");

    assert!(result.rank <= 10);

    let u_shape = *result.u.shape();
    let u_f64 = Tensor::from_fn(u_shape, |idx| result.u[[idx[0], idx[1]]].to_f64());
    let s_f64 = Tensor::from_fn((result.s.len(),), |idx| result.s[[idx[0]]].to_f64());
    let v_shape = *result.v.shape();
    let v_f64 = Tensor::from_fn(v_shape, |idx| result.v[[idx[0], idx[1]]].to_f64());

    let h_reconstructed = reconstruct_matrix_f64(&u_f64, &s_f64, &v_f64);
    let diff = matrix_sub(&h, &h_reconstructed);
    let relative_error = frobenius_norm(&diff) / frobenius_norm(&h);

    let h_shape = *h.shape();

    println!("  Matrix shape: {}x{}", h_shape.0, h_shape.1);
    println!("  Rank: {}", result.rank);
    println!("  Reconstruction error: {:.6e}", relative_error);

    assert!(
        relative_error < 1e-13,
        "Reconstruction error too large: {:.6e}",
        relative_error
    );
}

#[test]
fn test_singular_values_comparison_10x10() {
    let h = create_hilbert_matrix(10, 10);

    println!("Comparing f64 vs Df64 singular values for 10x10 Hilbert");

    let rtol = 1e-16;
    let result_f64 = tsvd_f64(&h, rtol).expect("f64 SVD failed");
    let result_tf = tsvd_df64_from_f64(&h, rtol).expect("Df64 SVD failed");

    assert_eq!(
        result_f64.s.len(),
        result_tf.s.len(),
        "Different number of singular values"
    );

    println!("  i   f64              Df64         Rel Diff");
    for i in 0..result_f64.s.len() {
        let s_tf = result_tf.s[[i]].to_f64();
        let rel_diff = ((result_f64.s[[i]] - s_tf) / result_f64.s[[i]]).abs();
        println!(
            "  {:2}  {:.6e}     {:.6e}     {:.2e}",
            i,
            result_f64.s[[i]],
            s_tf,
            rel_diff
        );

        // Allow larger relative error for very small singular values
        // (below machine epsilon * first singular value)
        let tolerance = if result_f64.s[[i]] < 1e-12 * result_f64.s[[0]] {
            1e-3 // 0.1% for very small singular values
        } else {
            1e-6 // 0.0001% for larger ones
        };

        assert!(
            rel_diff < tolerance,
            "Singular value {} differs too much: {:.2e} (tolerance: {:.2e})",
            i,
            rel_diff,
            tolerance
        );
    }
}
